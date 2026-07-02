import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from SRC.auth.oauth2 import verify_user
from SRC.core.dependencies import storage
from SRC.services.validation_service import validate_image
from SRC.services.enhancement_service import assess_quality, enhance_image
from SRC.services.detection_service import classify_document, detect_signatures
from SRC.utils.file_validator import verify_file_format
from SRC.services.storage_service import store_accepted, store_enhanced, store_rejected

# Import the newly updated function
from SRC.training.pdf_utils import pdf_to_images 

router = APIRouter(tags=["Detection"])

@router.post("/detect")
async def detect(
    file: UploadFile = File(...),
    username: str = Depends(verify_user),
):
    contents = await file.read()

    # File format verification
    fmt = verify_file_format(contents, file.content_type)
    if not fmt["valid"]:
        return JSONResponse(status_code=400, content={
            "error":  "Invalid file format",
            "reason": fmt["reason"],
        })

    # PDF vs Image Routing logic
    if fmt["actual"] == "pdf":
        try:
            # Get the list of all pages
            images = pdf_to_images(contents)
            # Use the first page as our primary representative image
            image = images[0] 
        except Exception as e:
            raise HTTPException(422, f"Could not process PDF: {str(e)}")
    else:
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Wrap single images in a list so the detection loop below works
        images = [image]

    if image is None:
        raise HTTPException(422, "Could not decode image")

    # Validation (Checks Page 1)
    validation = validate_image(image, len(contents))
    if not validation["passed"]:
        rejected_path = store_rejected(
            storage, image, file.filename, validation["reason"]
        )
        return JSONResponse(content={
            "document":   file.filename,
            "validation": {
                "passed": False,
                "reason": validation["reason"],
            },
            "stored": {"rejected": rejected_path},
        })

    # Quality assessment before (Checks Page 1)
    before = assess_quality(image)

    # Enhancement (Applied to Page 1)
    needs_enhancement = before.overall_score < 65
    enhanced_image    = image
    enhancements      = []

    if needs_enhancement:
        enhanced_image, enhancements = enhance_image(image, before)

    # Quality assessment after
    after = assess_quality(enhanced_image)

    # Classify document type (Based on Page 1)
    doc_info = classify_document(enhanced_image)

    # --- THE NEW MULTI-PAGE DETECTOR ---
    all_detections = []
    
    # Loop through every single page extracted from the PDF
    for page_num, img in enumerate(images):
        # If it's the first page and it needed enhancement, use the enhanced version.
        # Otherwise, use the raw page.
        target_img = enhanced_image if page_num == 0 else img
        
        # Run YOLO on this specific page
        page_detections = detect_signatures(target_img, storage)
        all_detections.extend(page_detections)

    # Reassign our combined list back to the detections variable
    detections = all_detections
    # -----------------------------------

    # Store in MinIO (Stores Page 1 as the main document)
    orig_path     = store_accepted(storage, image, file.filename)
    enhanced_path = store_enhanced(storage, enhanced_image, file.filename) \
                    if needs_enhancement else None

    return JSONResponse(content={
        "document":     file.filename,
        "requested_by": username,
        "validation": {
            "passed": True,
            "reason": "Image quality acceptable",
        },
        "quality": {
            "before": {
                "blur_score":    before.blur_score,
                "brightness":    before.brightness,
                "contrast":      before.contrast,
                "overall_score": before.overall_score,
                "grade":         before.grade,
            },
            "after": {
                "blur_score":    after.blur_score,
                "brightness":    after.brightness,
                "contrast":      after.contrast,
                "overall_score": after.overall_score,
                "grade":         after.grade,
            },
            "enhanced":             needs_enhancement,
            "enhancements_applied": enhancements,
        },
        "document_type":       doc_info["document_type"],
        "document_confidence": doc_info["document_confidence"],
        "signature_found":     len(detections) > 0,
        "signature_verified":  any(d["verified"] for d in detections),
        "total_signatures":    len(detections),
        "detections":          detections,
        "stored": {
            "accepted":   orig_path, 
            "enhanced":   enhanced_path,
            "signatures": [d["crop_url"] for d in detections],
        },
    })