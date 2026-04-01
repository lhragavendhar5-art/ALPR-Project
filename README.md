# 🚗 ALPR — Automatic License Plate Recognition System

> A real-time Automatic License Plate Recognition (ALPR) system built for gate access control, using **YOLOv11** for plate detection and a custom-trained **CNN** for character recognition — optimized for **Indian license plates**.

---

## 📌 Project Overview

This system provides real-time license plate detection and OCR for smart gate management. It integrates a FastAPI backend, a live camera feed (DroidCam/USB), a web dashboard, and an alert notification system.

### Key Features

- 🔍 **YOLOv11-based plate detection** — fast and accurate bounding box localization
- 🔤 **Custom CNN OCR** — trained on Indian license plate fonts for high-accuracy character recognition
- 🎥 **Live camera streaming** — supports DroidCam (USB/Wi-Fi) and webcam
- 🖥️ **Admin dashboard** — real-time gate logs, visitor verification, and manual control
- 📧 **Email alerts** — automated notifications on gate events
- 🗄️ **SQLite database** — stores plate records, gate logs, and vehicle metadata

---

## 🧱 System Architecture

```
Camera Feed
    │
    ▼
YOLOv11 Plate Detector  ──►  Plate Crop
                                  │
                                  ▼
                        CNN Character Recognizer
                                  │
                                  ▼
                        Plate Validation & DB Lookup
                                  │
                        ┌─────────┴──────────┐
                        ▼                    ▼
                   Grant Access          Alert / Log
```

---

## 🗂️ Project Structure

```
phase5_application/
├── main.py                  # FastAPI backend (detection, OCR, gate logic)
├── dashboard.html           # Admin web dashboard (frontend)
├── START_ALPR.bat           # One-click startup script (Windows)
├── train_yolo11.py          # YOLOv11 training script
├── train_cnn_ocr.py         # CNN OCR training script
├── diagnose.py              # Diagnostics tool for detection pipeline
├── models/
│   ├── yolo11_plate.pt      # Trained YOLOv11 plate detector
│   └── plate_ocr_cnn.pth    # Trained CNN character recognizer
├── database/                # SQLite database files
├── dataset_indian/          # Indian plate dataset for training
├── chars_dataset/           # Character-level dataset for CNN
├── easyocr_models/          # EasyOCR model cache
├── uploads/                 # Uploaded image/video files
└── frontend/                # Additional frontend assets
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.9+
- pip
- (Optional) CUDA-enabled GPU for faster training

### Step 1 — Clone the repository

```bash
git clone https://github.com/lhragavendhar5-art/ALPR-Project.git
cd ALPR-Project
```

### Step 2 — Install dependencies

```bash
pip install fastapi uvicorn ultralytics torch torchvision easyocr opencv-python pillow
```

### Step 3 — Run the application

```bash
# Windows (recommended)
START_ALPR.bat

# Or manually
python main.py
```

### Step 4 — Open the dashboard

Navigate to `http://localhost:8000` in your browser.

---

## 🏋️ Training the Models

### Option A — CNN OCR Only (Fastest, ~15 min on CPU)
```bash
pip install torch torchvision
python train_cnn_ocr.py --generate --epochs 40
```
Output: `models/plate_ocr_cnn.pth` (~98% accuracy)

### Option B — YOLOv11 + CNN OCR (Best Accuracy, needs free Roboflow account)
```bash
pip install ultralytics roboflow torch torchvision
python train_yolo11.py --api-key YOUR_ROBOFLOW_KEY
python train_cnn_ocr.py --generate --epochs 40
```

### Option C — Kaggle Dataset (No Roboflow needed)
```bash
# Place kaggle.json in C:\Users\YOU\.kaggle\
pip install kaggle ultralytics torch torchvision
python train_yolo11.py --source kaggle
```

### Option D — Custom Dataset
```bash
python train_yolo11.py --source local --dataset your_dataset/
```

---

## 🧪 Testing & Diagnostics

```bash
# Test plate detection on a single image
python train_yolo11.py --test your_photo.jpg --weights models/yolo11_plate.pt

# Test CNN OCR on a plate crop
python train_cnn_ocr.py --test plate_crop.jpg

# Full pipeline diagnostic
python diagnose.py your_photo.jpg
```

---

## 🔧 Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce batch size: `--batch 8` or `--batch 4` |
| `No module named ultralytics` | `pip install ultralytics` |
| YOLOv11 not found | `pip install ultralytics --upgrade` (needs >= 8.3.0) |
| Training too slow | Use nano model (default), reduce epochs, or use Google Colab |
| Still misreading after training | Run `python diagnose.py your_photo.jpg` for debug images |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| Detection | YOLOv11 (Ultralytics) |
| OCR | Custom CNN + EasyOCR |
| Database | SQLite |
| Frontend | HTML/CSS/JavaScript |
| Camera | OpenCV + DroidCam |

---

## 📄 License

This project is for educational purposes as part of an academic submission.

---

## 👤 Author

**lhragavendhar5-art**  
[GitHub Profile](https://github.com/lhragavendhar5-art)
