
ALPR — YOLOv11 + CNN OCR Training Guide
=========================================

YOUR PROBLEM: Reading wrong plates (e.g. "DL 7CO 1939" misread)
SOLUTION: Train on actual Indian plate dataset + dedicated character CNN

FILES ADDED:
  train_yolo11.py    — Train YOLOv11 plate detector
  train_cnn_ocr.py   — Train CNN character recognizer
  patch_main.py      — Auto-patch main.py to use new models

═══════════════════════════════════════════════════════
 OPTION A — FASTEST (no GPU, no account needed)
 CNN OCR only — fixes most misreads in ~15 min on CPU
═══════════════════════════════════════════════════════

Step 1: Install deps
  pip install torch torchvision

Step 2: Generate synthetic data + train CNN
  python train_cnn_ocr.py --generate --epochs 40

  Output: models/plate_ocr_cnn.pth  (~98% accuracy)

Step 3: Patch main.py
  python patch_main.py

Step 4: Restart server
  START_ALPR.bat   (or python main.py)

═══════════════════════════════════════════════════════
 OPTION B — BEST ACCURACY (recommended)
 YOLOv11 + CNN OCR (needs free Roboflow account)
═══════════════════════════════════════════════════════

Step 1: Get FREE Roboflow API key
  → https://roboflow.com → Sign up → Settings → API Keys

Step 2: Install deps
  pip install ultralytics roboflow torch torchvision

Step 3: Train YOLOv11 on Indian plate dataset
  python train_yolo11.py --api-key YOUR_KEY_HERE

  This downloads ~3,400 Indian plate images and trains YOLOv11n.
  Takes ~20-40 min on CPU, ~5 min on GPU.
  Output: models/yolo11_plate.pt

Step 4: Train CNN OCR
  python train_cnn_ocr.py --generate --epochs 40
  Output: models/plate_ocr_cnn.pth

Step 5: Patch + restart
  python patch_main.py
  python main.py

═══════════════════════════════════════════════════════
 OPTION C — KAGGLE DATASET (no Roboflow needed)
═══════════════════════════════════════════════════════

Step 1: Get Kaggle API key
  → https://www.kaggle.com/settings → API → Create New API Token
  → Save kaggle.json to C:\Users\YOU\.kaggle\kaggle.json

Step 2:
  pip install kaggle ultralytics torch torchvision
  python train_yolo11.py --source kaggle

═══════════════════════════════════════════════════════
 OPTION D — YOUR OWN DATASET
═══════════════════════════════════════════════════════

If you have images + YOLO-format labels:

  dataset/
    images/train/*.jpg
    images/val/*.jpg
    labels/train/*.txt   (class cx cy w h)
    labels/val/*.txt

Run:
  python train_yolo11.py --source local --dataset dataset/

Annotate your own images free at:
  https://roboflow.com  (point at images, draw boxes, export YOLO)

═══════════════════════════════════════════════════════
 AFTER TRAINING — MODEL LOCATIONS
═══════════════════════════════════════════════════════

  models/yolo11_plate.pt      ← YOLOv11 detection
  models/plate_ocr_cnn.pth    ← CNN character recognition
  models/yolo11_plate/        ← Training results, plots, metrics

main.py will auto-load these on startup. No code changes needed
after running patch_main.py.

═══════════════════════════════════════════════════════
 TESTING
═══════════════════════════════════════════════════════

Test detector on an image:
  python train_yolo11.py --test your_photo.jpg --weights models/yolo11_plate.pt

Test CNN OCR on a plate crop:
  python train_cnn_ocr.py --test plate_crop.jpg

Run the ALPR diagnostic:
  python diagnose.py your_photo.jpg

═══════════════════════════════════════════════════════
 TROUBLESHOOTING
═══════════════════════════════════════════════════════

Q: "CUDA out of memory"
A: Reduce batch size: --batch 8  or  --batch 4

Q: "No module named ultralytics"
A: pip install ultralytics

Q: YOLOv11 not found / wrong version
A: pip install ultralytics --upgrade
   (ultralytics >= 8.3.0 includes YOLOv11)

Q: Training too slow
A: Use nano model (default), reduce epochs to 30,
   or use Google Colab with free GPU.

Q: Still reading wrong after training
A: Check debug images with: python diagnose.py your_photo.jpg
   Make sure plate is well-lit, not at extreme angle.
   The CNN OCR needs clean character segmentation.
