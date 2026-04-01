"""
test_cnn.py — Standalone CNN OCR tester
========================================
Tests the CNN model directly WITHOUT starting the full Flask server.

If you pass a FULL car image (like creta.jpg), it will first try YOLO
to auto-detect and crop the plate region, then run CNN OCR on the crop.

If you pass a CROPPED plate image, it skips YOLO and reads it directly.

Usage:
    python test_cnn.py              # uses creta.jpg (full car — YOLO crops plate first)
    python test_cnn.py myplate.jpg  # any image — auto-detects or reads directly
"""

import sys, os, cv2, numpy as np, torch, torch.nn as nn

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'plate_ocr_cnn.pth')
YOLO_PATH   = os.path.join(BASE_DIR, 'models', 'yolo11_plate.pt')

print("=" * 52)
print("  CNN OCR — STANDALONE TEST")
print("=" * 52)

# ── 1. Build & load CNN model ──────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"\n  ❌ CNN model not found: {MODEL_PATH}")
    print("  Run: python train_cnn_ocr.py --generate")
    sys.exit(1)

class _DW(nn.Module):
    def __init__(self, i, o, s=1):
        super().__init__()
        self.dw  = nn.Conv2d(i, i, 3, stride=s, padding=1, groups=i, bias=False)
        self.pw  = nn.Conv2d(i, o, 1, bias=False)
        self.bn  = nn.BatchNorm2d(o)
        self.act = nn.Hardswish()
    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class _PlateOCR(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.Hardswish(),
            _DW(32,  64,  1), _DW(64,  128, 2),
            _DW(128, 128, 1), _DW(128, 256, 2),
            _DW(256, 256, 1), _DW(256, 512, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.Hardswish(),
            nn.Dropout(0.2), nn.Linear(256, nc),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

ck    = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
chars = ck.get('chars', list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
model = _PlateOCR(len(chars))
model.load_state_dict(ck['model_state_dict'])
model.eval()

print(f"\n  ✅ CNN loaded  — val_acc={ck.get('val_acc',0)*100:.1f}%  "
      f"epoch={ck.get('epoch','?')}  classes={len(chars)}")

# ── 2. Load image ──────────────────────────────────────
img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, 'creta.jpg')
if not os.path.exists(img_path):
    print(f"\n  ❌ Image not found: {img_path}")
    sys.exit(1)

img = cv2.imread(img_path)
if img is None:
    print(f"\n  ❌ Cannot read image: {img_path}")
    sys.exit(1)

print(f"  Image   : {os.path.basename(img_path)}  ({img.shape[1]}x{img.shape[0]})")

# ── 3. Try YOLO to crop the plate (if full-car image) ──
crop = img   # default: treat whole image as plate crop
h, w = img.shape[:2]
is_full_image = (w > 400 and h > 300)   # likely a full car image

if is_full_image and os.path.exists(YOLO_PATH):
    print("\n  Full-size image detected → running YOLO to crop plate...")
    try:
        # Patch torch.load to allow weights_only=False
        _orig = torch.load
        def _patch(f, *a, **kw): kw['weights_only'] = False; return _orig(f, *a, **kw)
        torch.load = _patch
        from ultralytics import YOLO as _YOLO
        yolo = _YOLO(YOLO_PATH)
        torch.load = _orig

        preds = yolo(img, conf=0.15, iou=0.45, verbose=False, imgsz=640)[0]
        if preds.boxes and len(preds.boxes):
            x1, y1, x2, y2 = map(int, preds.boxes[0].xyxy[0].tolist())
            pad = 12
            x1=max(0,x1-pad); y1=max(0,y1-pad)
            x2=min(w,x2+pad); y2=min(h,y2+pad)
            crop = img[y1:y2, x1:x2]
            cf   = float(preds.boxes[0].conf[0])
            print(f"  ✅ YOLO detected plate  conf={cf*100:.1f}%  "
                  f"crop={crop.shape[1]}x{crop.shape[0]}")
            # Save crop so you can inspect it
            crop_out = os.path.join(BASE_DIR, '_cnn_test_crop.jpg')
            cv2.imwrite(crop_out, crop)
            print(f"  💾 Crop saved → {crop_out}")
        else:
            print("  ⚠  YOLO found no plate — running CNN on full image anyway")
    except Exception as e:
        print(f"  ⚠  YOLO failed ({e}) — running CNN on full image anyway")
        torch.load = _orig  # restore just in case
elif is_full_image:
    print("  ⚠  YOLO model not found — running CNN on full image (may be inaccurate)")

# ── 4. Segment characters from crop ────────────────────
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
h2, w2 = gray.shape
if w2 < 200:
    gray = cv2.resize(gray, (300, int(h2 * 300 / w2)), interpolation=cv2.INTER_LANCZOS4)
    h2, w2 = gray.shape

blur      = cv2.GaussianBlur(gray, (3, 3), 0)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
if np.mean(thresh) > 127:
    thresh = cv2.bitwise_not(thresh)
k      = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)

cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
regions = []
for cnt in cnts:
    x, y, cw, ch = cv2.boundingRect(cnt)
    ar = cw / ch if ch > 0 else 0
    if ch > h2 * 0.15 and cw > 3 and ch > 8 and ar < 2.0 and cw < w2 * 0.30:
        regions.append((x, y, cw, ch))
regions.sort(key=lambda r: r[0])

print(f"\n  Character regions found: {len(regions)}")

if len(regions) < 2:
    print("  ❌ Too few regions — plate crop unclear or image not cropped tightly.")
    print("     Check _cnn_test_crop.jpg to see what YOLO cropped.")
    sys.exit(0)

# ── 5. Classify each character ─────────────────────────
print(f"\n  {'#':<4} {'Char':<6} {'Conf':>6}")
print(f"  {'-'*18}")
preds = []
with torch.no_grad():
    for i, (rx, ry, rcw, rch) in enumerate(regions):
        char_crop = gray[ry:ry+rch, rx:rx+rcw]
        char_crop = cv2.resize(char_crop, (48, 64))
        t = torch.tensor(char_crop, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        probs     = torch.softmax(model(t), dim=1)
        conf, idx = probs.max(1)
        c  = chars[int(idx[0])]
        cf = float(conf[0])
        preds.append((c, cf))
        print(f"  {i+1:<4} {c:<6} {cf*100:>5.1f}%")

plate    = ''.join(c for c, _ in preds)
avg_conf = sum(cf for _, cf in preds) / len(preds)

import re
PATTERNS = [
    r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$',
    r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{3}$',
    r'^[A-Z]{2}\d{2}\d{4}$',
]
valid = any(re.match(p, plate.upper()) for p in PATTERNS)

print(f"\n  {'='*30}")
print(f"  Plate      : {plate}")
print(f"  Avg conf   : {avg_conf*100:.1f}%")
print(f"  Valid fmt  : {'✅ Yes' if valid else '⚠  No (partial read or non-standard)'}")
print(f"  {'='*30}\n")
