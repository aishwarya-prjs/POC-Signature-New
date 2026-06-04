

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from loguru import logger


# ── Quality Metrics ───────────────────────────────────────────────────────────

@dataclass
class QualityReport:
    blur_score:        float   # Laplacian variance  (higher = sharper)
    noise_score:       float   # estimated SNR in dB
    contrast_score:    float   # RMS contrast 0-1
    brightness_score:  float   # mean luminance 0-1
    resolution_ok:     bool    # >= 150 DPI equivalent
    overall_score:     float   # 0-100 composite
    needs_enhancement: bool


def assess_quality(img: np.ndarray, min_score: float = 55.0) -> QualityReport:
    """
    Compute image quality metrics.
    Returns QualityReport with needs_enhancement flag.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    # 1. Blur  (Laplacian variance)
    lap_var    = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(100.0, lap_var / 5.0)          # normalise ≈ 0-100

    # 2. Noise  (simple SNR estimate via local std dev)
    ksize       = 5
    mean, std   = cv2.meanStdDev(gray)
    local_std   = cv2.blur((gray.astype(np.float32) - mean[0][0]) ** 2, (ksize, ksize))
    noise_std   = float(np.sqrt(np.mean(local_std)))
    snr_db      = 20 * np.log10((mean[0][0] + 1e-6) / (noise_std + 1e-6))
    noise_score = float(np.clip(snr_db, 0, 40) / 40 * 100)

    # 3. Contrast  (RMS)
    contrast_score = float(std[0][0] / 128.0 * 100)

    # 4. Brightness (penalise very dark or very bright)
    lum            = float(mean[0][0]) / 255.0
    brightness_ok  = 1.0 - abs(lum - 0.5) * 2          # peaks at 0.5
    brightness_score = brightness_ok * 100

    # 5. Resolution proxy  (short side ≥ 300 px ≈ 150 DPI on A4)
    h, w       = gray.shape[:2]
    res_ok     = min(h, w) >= 300

    # Composite
    overall = (
        blur_score      * 0.35 +
        noise_score     * 0.25 +
        contrast_score  * 0.20 +
        brightness_score* 0.20
    )
    if not res_ok:
        overall *= 0.7   # penalty for low-res

    return QualityReport(
        blur_score        = round(blur_score,       2),
        noise_score       = round(noise_score,      2),
        contrast_score    = round(contrast_score,   2),
        brightness_score  = round(brightness_score, 2),
        resolution_ok     = res_ok,
        overall_score     = round(overall,          2),
        needs_enhancement = overall < min_score,
    )


# ── Enhancement Pipeline ──────────────────────────────────────────────────────

def denoise(img: np.ndarray) -> np.ndarray:
    """Non-local means denoising (colour-aware)."""
    if img.ndim == 2:
        return cv2.fastNlMeansDenoising(img, h=10)
    return cv2.fastNlMeansDenoisingColored(img, h=10, hColor=10,
                                           templateWindowSize=7,
                                           searchWindowSize=21)


def sharpen(img: np.ndarray) -> np.ndarray:
    """Unsharp masking."""
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)


def clahe_enhance(img: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement (works on L channel in LAB)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if img.ndim == 2:
        return clahe.apply(img)
    lab        = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b    = cv2.split(lab)
    l_eq       = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def deskew(img: np.ndarray) -> np.ndarray:
    """Correct small rotation angles via Hough lines on binarised image."""
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords   = np.column_stack(np.where(bw > 0))
    if len(coords) < 10:
        return img
    angle    = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle

    if abs(angle) < 0.5:   # skip tiny corrections
        return img

    h, w     = img.shape[:2]
    M        = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))


def super_resolve(img: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Simple bicubic upscaling as SR fallback.
    For production, swap with ESRGAN / Real-ESRGAN.
    """
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def enhance(img: np.ndarray, report: QualityReport) -> np.ndarray:
    """
    Apply a targeted pipeline based on quality report.
    Returns enhanced image.
    """
    out = img.copy()

    if not report.resolution_ok:
        logger.debug("  → super-resolving (low res)")
        out = super_resolve(out)

    if report.blur_score < 30:
        logger.debug("  → denoising + sharpening (blurry)")
        out = denoise(out)
        out = sharpen(out)

    if report.noise_score < 50:
        logger.debug("  → denoising (noisy)")
        out = denoise(out)

    if report.contrast_score < 40:
        logger.debug("  → CLAHE (low contrast)")
        out = clahe_enhance(out)

    if report.brightness_score < 30:
        logger.debug("  → deskewing")
        out = deskew(out)

    # Always apply mild CLAHE at the end for document clarity
    out = clahe_enhance(out)

    return out


# ── Public API ────────────────────────────────────────────────────────────────

def process_image(img: np.ndarray,
                  quality_threshold: float = 55.0
                  ) -> tuple[np.ndarray, QualityReport]:
    """
    Assess quality, enhance if necessary.
    Returns (final_image, quality_report).
    """
    report = assess_quality(img, min_score=quality_threshold)
    logger.info(f"Quality score: {report.overall_score:.1f}/100 "
                f"{'— ENHANCING' if report.needs_enhancement else '— OK'}")

    if report.needs_enhancement:
        img = enhance(img, report)

    return img, report
