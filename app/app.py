from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from PIL import Image
import io

# from schemas import PredictionResponse, Detection, BoundingBox
from .schemas import PredictionResponse, Detection, BoundingBox


app = FastAPI(
    title="Bangla Note Detection API",
    description="YOLOv11 based Bangla currency detection",
    version="1.0"
)

# Load trained YOLO model
MODEL_PATH = "model/best.pt"
model = YOLO(MODEL_PATH)
CLASS_NAMES = ['5', '10', '20', '50', '100', '500', '1000']

def run_inference(image: Image.Image):
    results = model.predict(image, conf=0.25, save=False)

    detections = []

    for result in results:
        boxes = result.boxes
        names = result.names

        if boxes is None:
            continue

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = names[class_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=round(confidence, 3),
                    bbox=BoundingBox(
                        x1=x1, y1=y1, x2=x2, y2=y2
                    )
                )
            )

    return detections

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Validate file existence
    if not file:
        raise HTTPException(
            status_code=400,
            detail="No image file provided"
        )

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=415,
            detail="Invalid file type. Only JPEG and PNG are supported."
        )

    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    detections = run_inference(image)

    return PredictionResponse(detections=detections)
