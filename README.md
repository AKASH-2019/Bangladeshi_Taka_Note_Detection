# 💰 Bangla Taka Note Detection API

YOLOv11 + FastAPI + Docker Deployment

### 📌 Project Overview

This project is a Bangladeshi Currency Note Detection System built using YOLOv11 (Ultralytics) for object detection.
The trained model is served via a FastAPI REST API and fully containerized using Docker for deployment.

The system detects Bangla currency denominations from input images and returns:

✅ Detected denomination names

✅ Confidence scores

✅ Bounding box coordinates

### 🎯 Objectives

Train a YOLO model for Bangla note detection

Implement single-image inference

Develop a REST API using FastAPI

Dockerize the application

Serve the model via /predict endpoint

### 🧠 Model Details

Model: YOLOv11 (Ultralytics)

Weights: best.pt

Task: Object Detection

Classes: Bangla currency denominations (e.g., 10, 20, 50, 100, etc.)

Framework: PyTorch

### 📂 Project Structure
bangla_note_api/
* │
* ├── app/
* │   ├── app.py              # FastAPI application
* │   ├── schemas.py          # Pydantic response models
* │
* ├── model/
* │   └── best.pt             # Trained YOLO model weights
* │
* ├── requirements.txt
* ├── Dockerfile
* └── README.md

### 🚀 API Specification
🔹 Endpoint
POST /predict

🔹 Input

Image file (JPEG or PNG)

Form-data key: file

🔹 Output (JSON)
{
  "detections": [
    {
      "class_name": "100",
      "confidence": 0.92,
      "bbox": {
        "x1": 118,
        "y1": 75,
        "x2": 455,
        "y2": 305
      }
    }
  ]
}

### ⚙️ Installation (Without Docker)
1️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run API
uvicorn app.app:app --reload

4️⃣ Open Swagger UI
http://localhost:8000/docs

### 🐳 Docker Setup
🔹 Build Docker Image
docker build -t bangla-note-api .

🔹 Run Container
docker run -d -p 8000:8000 --name bangla_note_container bangla-note-api

🔹 Check Running Container
docker ps

🔹 View Logs
docker logs bangla_note_container

🔹 Access API
http://localhost:8000/docs

### 📦 Dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#### Copy app folder as package
* COPY app ./app
* COPY model ./model

EXPOSE 8000

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]

### 📋 Requirements
fastapi
uvicorn
ultralytics
pillow
python-multipart
numpy
torch

### 🛡 Error Handling

The API handles:

❌ Missing file → 400 Bad Request

❌ Invalid file type → 415 Unsupported Media Type

❌ Corrupt image → 400 Bad Request

### ✅ Successful detection → 200 OK

### 📊 Technologies Used

Python 3.10

YOLOv11 (Ultralytics)

PyTorch

FastAPI

Pydantic

Docker

Uvicorn

### 🎓 Learning Outcomes

Through this project, I learned:

Object detection using YOLO

Model inference pipeline design

REST API development

Docker containerization

Model deployment workflow
