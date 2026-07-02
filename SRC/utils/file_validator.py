MAGIC_BYTES = {
    "jpeg": [b"\xff\xd8\xff"],
    "png":  [b"\x89PNG\r\n\x1a\n"],
    "pdf":  [b"%PDF-"],
    "bmp":  [b"BM"],
}

ALLOWED_TYPES = ["jpeg", "png", "pdf"]

def detect_file_format(file_bytes: bytes) -> str:
  
    if file_bytes[0:4] == b"RIFF" and file_bytes[8:12] == b"WEBP":
        return "webp"
        
  
    for fmt, signatures in MAGIC_BYTES.items():
        for sig in signatures:
            if file_bytes[:len(sig)] == sig:
                return fmt
                
    return "unknown"

def verify_file_format(file_bytes: bytes, declared_content_type: str) -> dict:
    actual_format = detect_file_format(file_bytes)

    if actual_format not in ALLOWED_TYPES:
        return {
            "valid":  False,
            "reason": f"Format '{actual_format}' not allowed. Only JPEG, PNG, and PDF accepted.",
            "actual": actual_format,
        }

    declared = declared_content_type.lower()
    mismatch = False

   
    if actual_format == "jpeg" and "jpeg" not in declared and "jpg" not in declared:
        mismatch = True
    elif actual_format == "png" and "png" not in declared:
        mismatch = True
    elif actual_format == "pdf" and "pdf" not in declared:
        mismatch = True

    if mismatch:
        return {
            "valid":  False,
            "reason": f"File mismatch. File is actually {actual_format} but declared as {declared_content_type}.",
            "actual": actual_format,
        }

    return {
        "valid":  True,
        "reason": f"Valid {actual_format.upper()} file",
        "actual": actual_format,
    }