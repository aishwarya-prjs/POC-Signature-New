import uuid
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dependencies import storage, classifier, detector

router = APIRouter(tags=["Detection"])


@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    if classifier is None or detector is None:
        raise HTTPException(503, "Models not loaded — check api/models/ folder")

    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    image    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(422, "Could not decode image")

    h, w = image.shape[:2]

    cls_result  = classifier.predict(image, verbose=False)
    probs       = cls_result[0].probs
    top1_idx    = int(probs.top1)
    top1_conf   = float(probs.top1conf)
    class_names = cls_result[0].names
    doc_type    = class_names[top1_idx] if top1_conf >= 0.5 else "others"

    det_result = detector.predict(image, conf=0.3, verbose=False)
    detections = []

    for i, box in enumerate(det_result[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf            = float(box.conf[0])

        pad  = 10
        cx1  = max(0, x1 - pad)
        cy1  = max(0, y1 - pad)
        cx2  = min(w, x2 + pad)
        cy2  = min(h, y2 + pad)
        crop = image[cy1:cy2, cx1:cx2]

        run_id      = str(uuid.uuid4())[:8]
        crop_object = f"signatures/{run_id}_sig_{i}.jpg"
        storage.upload_image(crop, crop_object)
        crop_url = storage.presigned_url(crop_object, expires_hours=24)

        detections.append({
            "signature_id":  i + 1,
            "confidence":    round(conf, 4),
            "verified":      conf >= 0.5,
            "bounding_box": {
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "width":  x2 - x1,
                "height": y2 - y1,
            },
            "crop_url": crop_url,
        })

    run_id      = str(uuid.uuid4())[:8]
    orig_object = f"uploads/{run_id}_{file.filename}"
    storage.upload_image(image, orig_object)

    return JSONResponse(content={
        "document":            file.filename,
        "document_type":       doc_type,
        "document_confidence": round(top1_conf, 4),
        "signature_found":     len(detections) > 0,
        "signature_verified":  any(d["verified"] for d in detections),
        "total_signatures":    len(detections),
        "detections":          detections,
        "stored_as":           orig_object,
    })
