"""
ALPR ENGINE v6 — REAL WORLD PRODUCTION
========================================
Fixes:
  1. Long distance: super-resolution upscale before OCR (EDSR-style bicubic)
  2. Image mistakes: tighter character correction + confidence threshold
  3. Faster R-CNN: loads automatically when models/fasterrcnn_plate.pth exists
  4. Better plate detection: adaptive multi-scale search
  5. Night/dark plates: histogram equalisation + gamma correction variant
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from datetime import datetime
import json, os, re, base64, threading, time, smtplib
import cv2, numpy as np, easyocr
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import defaultdict, Counter

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR   = os.path.join(BASE_DIR, 'database')
DB_FILE  = os.path.join(DB_DIR, 'vehicles.json')
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'uploads'), exist_ok=True)

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'endpoint not found'}), 404
    p = os.path.join(BASE_DIR, 'dashboard.html')
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            return Response(f.read(), mimetype='text/html')
    return Response('<h2>ALPR: place dashboard.html here</h2>',
                    mimetype='text/html'), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': str(e)}), 500

SYSTEM_CONFIG = {
    # ✅ FIX: You MUST generate a 16-letter App Password from your Google Account
    # Regular passwords will result in 'error:auth_failed'
    'email_sender':   'ragavendharlh@gmail.com',
    'email_password': 'ubgllebxlzvhdfzb',
    'telegram_token': os.environ.get('ALPR_TG_TOKEN', ''),
    'twilio_sid':     os.environ.get('ALPR_TWILIO_SID', ''),
    'twilio_token':   os.environ.get('ALPR_TWILIO_TOKEN', ''),
    'twilio_from':    os.environ.get('ALPR_TWILIO_FROM',
                                     'whatsapp:+14155238886'),
    'gate_mode':      'simulate',
    'gate_webhook':   '',
    'gate_open_ms':   5000,
}

print("=" * 62)
print("  ALPR ENGINE v6 — REAL WORLD PRODUCTION")
print("  Loading EasyOCR (one-time ~30s)...")
print("=" * 62)
reader = easyocr.Reader(['en'], gpu=False, verbose=False)
print("  ✅ EasyOCR ready")

# ══════════════════════════════════════════════════════════════
#  CNN OCR MODULE (injected by patch_main.py)
#  Fast character-level CNN trained on Indian plate fonts.
#  Runs before EasyOCR; EasyOCR is fallback if CNN confidence low.
# ══════════════════════════════════════════════════════════════

_cnn_ocr_model  = None
_cnn_ocr_chars  = None
_cnn_ocr_device = None
_cnn_enabled    = False

def _load_cnn_ocr():
    global _cnn_ocr_model, _cnn_ocr_chars, _cnn_ocr_device, _cnn_enabled
    cnn_paths = [
        os.path.join(BASE_DIR, 'models', 'plate_ocr_cnn.pth'),
        os.path.join(BASE_DIR, 'plate_ocr_cnn.pth'),
    ]
    for p in cnn_paths:
        if not os.path.exists(p):
            continue
        try:
            import torch
            import torch.nn as nn

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

            ck    = torch.load(p, map_location='cpu', weights_only=False)
            chars = ck.get('chars', list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
            m     = _PlateOCR(len(chars))
            m.load_state_dict(ck['model_state_dict'])
            m.eval()
            _cnn_ocr_model  = m
            _cnn_ocr_chars  = chars
            _cnn_ocr_device = torch.device('cpu')
            _cnn_enabled    = True
            print(f"  ✅ CNN OCR: {os.path.basename(p)}  "
                  f"({len(chars)} chars, val_acc={ck.get('val_acc',0)*100:.1f}%)")
            return
        except Exception as e:
            print(f"  ⚠  CNN OCR load failed: {e}")
    if not _cnn_enabled:
        print("  ℹ  CNN OCR: not loaded (train with train_cnn_ocr.py --generate)")

_load_cnn_ocr()


def _cnn_read_plate(plate_bgr, conf_threshold=0.40):
    """
    Segment plate into chars, classify each with CNN.
    Returns (plate_text, confidence) or (None, 0).

    conf_threshold lowered to 0.40 — synthetically trained models
    typically score 0.40–0.65 on real-world crops; 0.55 was too strict.
    """
    if not _cnn_enabled or plate_bgr is None or plate_bgr.size == 0:
        return None, 0.0
    try:
        import torch  # noqa: PLC0415
        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        # Upscale small crops for better segmentation
        h, w = gray.shape
        if w < 200:
            gray = cv2.resize(gray, (300, int(h * 300 / w)),
                               interpolation=cv2.INTER_LANCZOS4)
        # Binarize
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if np.mean(thresh) > 127:
            thresh = cv2.bitwise_not(thresh)
        # Morphological cleanup
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)

        # Find character contours
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        h2, w2 = gray.shape
        regions = []
        for cnt in cnts:
            x, y, cw, ch = cv2.boundingRect(cnt)
            ar = cw / ch if ch > 0 else 0
            # FIX: lowered height floor (0.15) and raised width ceiling (0.30)
            # to be highly permissive for all Indian plate fonts/styles.
            if (ch > h2 * 0.15 and cw > 3 and ch > 8
                    and ar < 2.0 and cw < w2 * 0.30):
                regions.append((x, y, cw, ch))
        regions.sort(key=lambda r: r[0])

        if len(regions) < 2:  # Indian plates have 8-10 chars; allow ≥2
            print(f"  [CNN-OCR] only {len(regions)} region(s) found — skip")
            return None, 0.0

        preds = []
        _cnn_ocr_model.eval()
        with torch.no_grad():
            for (rx, ry, rcw, rch) in regions:
                crop = gray[ry:ry+rch, rx:rx+rcw]
                crop = cv2.resize(crop, (48, 64))
                t = (torch.tensor(crop, dtype=torch.float32)
                     .unsqueeze(0).unsqueeze(0) / 255.0)
                logits = _cnn_ocr_model(t)
                probs  = torch.softmax(logits, dim=1)
                conf_t, idx = probs.max(1)
                preds.append((_cnn_ocr_chars[int(idx[0])], float(conf_t[0])))

        text      = ''.join(ch for ch, _ in preds)
        avg_conf  = sum(cf for _, cf in preds) / len(preds)

        if avg_conf < conf_threshold:
            print(f"  [CNN-OCR] low conf {avg_conf*100:.1f}% — skip")
            return None, 0.0

        # Validate as Indian plate
        validated = validate_plate(text)
        if validated:
            print(f"  [CNN-OCR] '{text}'→'{validated}' ({avg_conf*100:.1f}%)")
            return validated, avg_conf
        print(f"  [CNN-OCR] '{text}' failed validation")
        return None, 0.0

    except Exception as e:  # noqa: BLE001
        print(f"  [CNN-OCR] error: {e}")
        return None, 0.0



# ── YOLOv11 ───────────────────────────────────────────────────
_YOLO_PATHS = [
    os.path.join(BASE_DIR, 'models', 'yolo11_plate.pt'),   # ✅ YOUR TRAINED MODEL (5315 KB)
    os.path.join(BASE_DIR, 'models', 'yolov8_plate.pt'),
    os.path.join(BASE_DIR, 'yolo11n.pt'),
    os.path.join(BASE_DIR, 'yolov8n.pt'),
    os.path.join(BASE_DIR, 'models', 'yolov8n.pt'),
]
yolo_model   = None
yolo_enabled = False
yolo_name    = ''
for _p in _YOLO_PATHS:
    if os.path.exists(_p):
        try:
            from ultralytics import YOLO as _YOLO
            import torch as _t
            _o = _t.load
            def _patch(f, *a, **kw):
                kw['weights_only'] = False
                return _o(f, *a, **kw)
            _t.load = _patch
            try:
                yolo_model = _YOLO(_p)
            finally:
                _t.load = _o
            yolo_enabled = True
            yolo_name    = os.path.basename(_p)
            print(f"  ✅ YOLOv8: {yolo_name}")
            if 'plate' in yolo_name.lower():
                print("     (plate-specific model — best accuracy)")
            else:
                print("     ⚠  General model — get yolov8_plate.pt for better accuracy")
            break
        except Exception as e:
            print(f"  ⚠  YOLO: {e}")
            break

# ── Faster R-CNN ───────────────────────────────────────────────
frcnn_model   = None
frcnn_enabled = False
for _fp in [
    os.path.join(BASE_DIR, 'models', 'fasterrcnn_plate.pth'),
    os.path.join(BASE_DIR, 'fasterrcnn_plate.pth'),
]:
    if os.path.exists(_fp):
        try:
            import torch
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            _m = fasterrcnn_resnet50_fpn(weights=None)
            _m.roi_heads.box_predictor = FastRCNNPredictor(
                _m.roi_heads.box_predictor.cls_score.in_features, 2)
            _m.load_state_dict(
                torch.load(_fp, map_location='cpu', weights_only=False))
            _m.eval()
            frcnn_model   = _m
            frcnn_enabled = True
            print(f"  ✅ Faster R-CNN: {os.path.basename(_fp)}")
            break
        except Exception as e:
            print(f"  ⚠  Faster R-CNN: {e}")
            break

if not frcnn_enabled:
    print("  ℹ  Faster R-CNN: not loaded")
    print("     → Train with: python train_fasterrcnn.py")
    print("     → Then place models/fasterrcnn_plate.pth here")

print("=" * 62)

# ══════════════════════════════════════════════════════════════
#  VEHICLE ATTRIBUTES (COLOR & TYPE)
# ══════════════════════════════════════════════════════════════
class VehicleAnalyzer:
    def __init__(self):
        self.enabled = False
        try:
            from torchvision.models import mobilenet_v2
            from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
            self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
            self.enabled = True
            self.v_classes = {
                436:'Beach Wagon', 468:'Taxi', 511:'Convertible', 
                581:'Hatchback', 609:'Jeep / SUV', 627:'Limousine', 
                656:'Minivan', 675:'Moving Van', 717:'Pickup Truck', 
                751:'Race Car', 817:'Sports Car', 864:'Tow Truck'
            }
            import torchvision.transforms as T
            self.transform = T.Compose([
                T.ToPILImage(), T.Resize(256), T.CenterCrop(224),
                T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            print("  ✅ Vehicle Analyzer loaded (MobileNetV2 ImageNet)")
        except Exception as e:
            print(f"  ⚠  Vehicle Analyzer error: {e}")

    def get_color(self, img_bgr):
        import numpy as np
        import cv2
        img = cv2.resize(img_bgr, (50, 50))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixels = np.float32(img.reshape(-1, 3))
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, pal = cv2.kmeans(pixels, 3, None, crit, 5, cv2.KMEANS_RANDOM_CENTERS)
        dominant = pal[np.argmax(np.bincount(labels.flatten()))]
        COLORS = {
            "Black": (30,30,30), "White": (220,220,220), "Silver": (170,170,170),
            "Gray": (100,100,100), "Red": (200,30,30), "Blue": (30,30,200), 
            "Yellow": (200,200,30), "Green": (30,150,30)
        }
        return min(COLORS.keys(), key=lambda k: np.linalg.norm(dominant - np.array(COLORS[k])))

    def analyze(self, img_bgr):
        color = self.get_color(img_bgr)
        vtype = "Car"
        if self.enabled:
            import torch
            with torch.no_grad():
                t = self.transform(img_bgr).unsqueeze(0)
                _, pred = self.model(t).max(1)
                idx = pred.item()
                if idx in self.v_classes:
                    vtype = self.v_classes[idx]
        return color, vtype

vehicle_analyzer = VehicleAnalyzer()


# ══════════════════════════════════════════════════════════════
#  INDIAN PLATE CHARACTER CORRECTION
# ══════════════════════════════════════════════════════════════
INDIA_STATES = {
    'AP','AR','AS','BR','CG','CH','DD','DL','DN','GA','GJ','HP','HR',
    'JH','JK','KA','KL','LA','LD','MH','ML','MN','MP','MZ','NL','OD',
    'PB','PY','RJ','SK','TN','TR','TS','UK','UP','WB'
}
# letter → digit (for positions that must be digits)
L2D = {
    'O': '0', 'Q': '0', 'D': '0', 'U': '0', 'C': '0',
    'I': '1', 'L': '1', 'T': '1', 'J': '1',
    'S': '5', 'Z': '2', 'B': '8', 'G': '6', 'E': '3',
}
# digit → letter (for positions that must be letters)
D2L = {
    '0': 'O', '1': 'I', '5': 'S', '8': 'B',
    '6': 'G', '2': 'Z', '3': 'E',
}
# first-letter fixes for state codes
SC_FIX = {'I': 'T', 'F': 'P', 'E': 'K', 'J': 'U', '1': 'T'}

def to_letter(c):
    return D2L.get(c, c) if c.isdigit() else c

def to_digit(c):
    return L2D.get(c, c) if c.isalpha() else c

def normalize_fuzzy(p):
    return (p.upper()
             .replace('Q','0').replace('O','0').replace('D','0')
             .replace('U','0').replace('C','0')
             .replace('I','1').replace('L','1').replace('T','1').replace('J','1')
             .replace('S','5').replace('Z','2').replace('B','8').replace('G','6'))

def fix_plate(raw):
    t = re.sub(r'[^A-Z0-9]', '', raw.upper().strip())
    if len(t) < 5:
        return t
    c = list(t)
    n = len(c)

    # pos 0,1: always letters (state code)
    c[0] = to_letter(c[0])
    if n > 1:
        c[1] = to_letter(c[1])

    # fix state code misreads — brute-force against INDIA_STATES
    _sc = c[0] + (c[1] if n > 1 else '')
    if _sc not in INDIA_STATES:
        _second = c[1] if n > 1 else ''
        # Expanded visual similarity map — covers all common OCR confusions
        # Key = what OCR read, Value = what it could actually be
        _VIS = {
            'I': 'TDTJ1L',  'O': 'DQ0CG',  '1': 'ITL',
            'F': 'PE',      'E': 'KF',      'J': 'UI',
            '0': 'DOQ',     'B': '8',       'S': '5',
            'Z': '27',      'L': 'I1',
            # ── NEW fixes ──────────────────────────────────────────────
            'D': 'TO',      # D looks like T (your TN→DN bug) and O
            'T': 'DI1',     # T looks like D, I, 1
            'C': 'G0O',     # C looks like G (CH→GH etc.)
            'G': 'C6',      # G looks like C or 6
            'M': 'N',       # M/N confusion
            'N': 'M',
            'U': 'VJ0',     # U looks like V, J, 0
            'V': 'U',
            'P': 'F',
            'R': 'P',
            '6': '0G',      # 6 looks like 0 (your 02→62 bug at state pos)
            '8': 'B',
        }
        _c0_cands = [a for a in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                     if a + _second in INDIA_STATES]
        if _c0_cands:
            _prefs = _VIS.get(c[0], '')
            c[0] = next((p for p in _prefs if p in _c0_cands),
                        _c0_cands[0])
        elif n > 1:
            _c1_cands = [b for b in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                         if c[0] + b in INDIA_STATES]
            if _c1_cands:
                _prefs = _VIS.get(c[1], '')
                c[1] = next((p for p in _prefs if p in _c1_cands),
                            _c1_cands[0])

    if n < 4:
        return ''.join(c)

    # pos 2,3: district digits (e.g. TN[02]AH1234)
    # Common OCR errors at digit positions: 6→0, Z→7, S→5
    _DIGIT_FIX = {'Z': '7', 'S': '5', 'G': '6', '6': '6'}
    # Special: '6' at district pos often means OCR misread '0' as '6'
    # BUT '6' is a valid district digit too — only fix if state+district
    # does not match a known format. Keep '6' as-is by default.
    _DISTRICT_Z_FIX = {'Z': '7'}  # only Z is unambiguous → always fix
    if c[2] in _DISTRICT_Z_FIX:
        c[2] = _DISTRICT_Z_FIX[c[2]]
    else:
        c[2] = to_digit(c[2])

    # pos 3: OLD format (DL7CQ1939) has letter; NEW (TN09AB1234) has digit
    UNAMBIG_LETTERS = set('ACDEFHJKLMNPRTUVWXY')
    is_old = t[3] in UNAMBIG_LETTERS

    if is_old:
        c[3] = to_letter(c[3])
        if n > 4:
            c[4] = to_letter(c[4])
        serial_start = 3 + max(1, min(n - 7, 2))
        for i in range(serial_start, n):
            c[i] = to_digit(c[i])
    else:
        c[3] = to_digit(c[3])
        serial_start = max(4, n - 4)
        for i in range(4, serial_start):
            c[i] = to_letter(c[i])
        for i in range(serial_start, n):
            c[i] = to_digit(c[i])

    return ''.join(c)

PLATE_PATTERNS = [
    r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$',
    r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{3}$',
    r'^[A-Z]{2}\d{2}\d{4}$',
    r'^[A-Z]{2}\d{1}[A-Z]{1,2}\d{4}$',
    r'^[A-Z]{2}\d{1}[A-Z]{1,2}\d{5}$',
    r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{5}$',
    r'^\d{2}BH\d{4}[A-Z]{1,2}$',
]

def validate_plate(raw):
    t = re.sub(r'[^A-Z0-9]', '', raw.upper().strip())
    if not t or len(t) < 4:
        return None
    fixed = fix_plate(t)

    def _try_match(s):
        for p in PLATE_PATTERNS:
            if re.match(p, s):
                return s
        return None

    # 1. Strict match
    if _try_match(fixed):
        return fixed

    # 2. Trim one extra char (OCR over-read)
    if len(fixed) > 6 and _try_match(fixed[:-1]):
        return fixed[:-1]

    # 3. Digit-swap retries — fix common 0↔6 and 1↔7 confusions
    #    at every digit position until a pattern matches.
    #    This fixes TN[6]2AH72 → TN[0]2AH7200 partial reads.
    SWAP = {'6': '0', '0': '6', '1': '7', '7': '1', '8': '3', '3': '8'}
    chars = list(fixed)
    for i, ch in enumerate(chars):
        if ch in SWAP:
            trial = chars.copy()
            trial[i] = SWAP[ch]
            s = ''.join(trial)
            if _try_match(s):
                print(f"  [VAL] digit-swap fix pos{i}: {fixed}→{s}")
                return s
            # Also try trimming after swap
            if len(s) > 6 and _try_match(s[:-1]):
                print(f"  [VAL] digit-swap+trim fix: {fixed}→{s[:-1]}")
                return s[:-1]

    # 4. Lenient fallback — ONLY if valid Indian state code at start
    state_part = fixed[:2]
    if (5 <= len(fixed) <= 11
            and state_part in INDIA_STATES
            and re.search(r'[A-Z]', fixed)
            and re.search(r'\d', fixed)):
        return fixed
    return None


# ══════════════════════════════════════════════════════════════
#  SUPER-RESOLUTION UPSCALE
#  For long-distance plates that are small in the frame.
#  Uses Lanczos4 + sharpening — much better than simple resize.
# ══════════════════════════════════════════════════════════════

def super_resolve(img, target_w=800):
    """
    Upscale small plate crops to target_w using Lanczos4
    followed by unsharp masking.

    This is what makes long-distance plates readable —
    a plate that's 60px wide becomes 800px wide with
    sharp character edges.
    """
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return img
    if w >= target_w:
        return img  # already large enough

    scale = target_w / w
    # Lanczos4 is the best interpolation for upscaling text
    upscaled = cv2.resize(img, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_LANCZOS4)

    # Unsharp mask to recover edge sharpness after upscaling
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY) \
           if len(upscaled.shape) == 3 else upscaled
    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    sharp_gray = np.clip(
        cv2.addWeighted(gray, 2.2, blur, -1.2, 0),
        0, 255).astype(np.uint8)

    if len(upscaled.shape) == 3:
        # Apply sharpening to all channels
        result = upscaled.copy()
        for i in range(3):
            ch = upscaled[:, :, i].astype(float)
            bl = cv2.GaussianBlur(upscaled[:, :, i], (0, 0), 2.0)
            result[:, :, i] = np.clip(
                cv2.addWeighted(upscaled[:, :, i], 2.2, bl, -1.2, 0),
                0, 255)
        return result
    return cv2.cvtColor(sharp_gray, cv2.COLOR_GRAY2BGR)


# ══════════════════════════════════════════════════════════════
#  PREPROCESSING — 5 variants
#  V1: CLAHE + unsharp      — normal daylight plates
#  V2: Gamma + HistEq       — night / underexposed plates
#  V3: Adaptive threshold   — shadow / uneven lighting
#  V4: Deblur (unsharp ×3)  — motion blur / long distance
#  V5: CLAHE on HSV V-chan  — night with colour info preserved
# ══════════════════════════════════════════════════════════════

def preprocess_variants(crop_bgr):
    """
    Returns list of preprocessed BGR images for OCR.
    V1-V3: original variants (fast)
    V4: aggressive deblur for motion-blurred / distant plates
    V5: CLAHE on HSV value channel — best for very dark night plates
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return []

    # Super-resolve first — key for long-distance plates
    crop = super_resolve(crop_bgr, target_w=800)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    variants = []

    # V1: CLAHE + unsharp — best for normal daylight plates
    v1 = clahe.apply(gray)
    blur = cv2.GaussianBlur(v1, (0, 0), 2.0)
    v1 = np.clip(cv2.addWeighted(v1, 2.5, blur, -1.5, 0),
                 0, 255).astype(np.uint8)
    variants.append(cv2.cvtColor(v1, cv2.COLOR_GRAY2BGR))

    # V2: Gamma correction + equalization — night / underexposed
    gamma = 1.8
    v2 = np.uint8(np.clip(
        np.power(gray / 255.0, 1.0 / gamma) * 255, 0, 255))
    v2 = cv2.equalizeHist(v2)
    blur2 = cv2.GaussianBlur(v2, (0, 0), 1.5)
    v2 = np.clip(cv2.addWeighted(v2, 2.0, blur2, -1.0, 0),
                 0, 255).astype(np.uint8)
    variants.append(cv2.cvtColor(v2, cv2.COLOR_GRAY2BGR))

    # V3: Adaptive threshold — handles uneven lighting / shadows
    v3 = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (3, 3), 0), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 6)
    v3 = cv2.morphologyEx(
        v3, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    if np.mean(v3) < 100:
        v3 = cv2.bitwise_not(v3)
    variants.append(cv2.cvtColor(v3, cv2.COLOR_GRAY2BGR))

    # V4: Aggressive deblur — motion blur / long-distance plates
    # Stronger unsharp mask (weight 3.5 vs 2.5) recovers blurred edges
    blur4 = cv2.GaussianBlur(gray, (0, 0), 3.0)
    v4 = np.clip(cv2.addWeighted(gray, 3.5, blur4, -2.5, 0),
                 0, 255).astype(np.uint8)
    # Second pass: bilateral filter to suppress deblur noise
    v4 = cv2.bilateralFilter(v4, 5, 50, 50)
    variants.append(cv2.cvtColor(v4, cv2.COLOR_GRAY2BGR))

    # V5: CLAHE on HSV V-channel — best for dark night plates
    # Works on brightness only, keeps colour intact
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    clahe_night = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    v_enhanced  = clahe_night.apply(v_ch)
    v5_bgr = cv2.cvtColor(cv2.merge([h_ch, s_ch, v_enhanced]),
                           cv2.COLOR_HSV2BGR)
    # Convert to gray + unsharp after enhancement
    v5_gray = cv2.cvtColor(v5_bgr, cv2.COLOR_BGR2GRAY)
    blur5   = cv2.GaussianBlur(v5_gray, (0, 0), 1.5)
    v5_gray = np.clip(cv2.addWeighted(v5_gray, 2.0, blur5, -1.0, 0),
                      0, 255).astype(np.uint8)
    variants.append(cv2.cvtColor(v5_gray, cv2.COLOR_GRAY2BGR))

    return variants


# ══════════════════════════════════════════════════════════════
#  OCR — single fast call
# ══════════════════════════════════════════════════════════════

def ocr_single(img_bgr):
    """Single EasyOCR call — tuned for speed on CPU."""
    results = []
    try:
        raw_results = reader.readtext(
            img_bgr,
            detail=1,
            paragraph=False,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            width_ths=0.8,
            link_threshold=0.3,
            decoder='greedy',
            canvas_size=1280,  # 2560 was ~6-8s per call on CPU — 1280 is 1-2s
            mag_ratio=1.5,
            min_size=10,
        )
        for _, raw, conf in raw_results:
            plate = validate_plate(raw)
            if plate:
                results.append((plate, float(conf)))
                print(f"    [OCR] '{raw}'→'{plate}' ({conf*100:.1f}%)")
    except Exception as e:  # noqa: BLE001
        print(f"  [OCR] error: {e}")
    return results


def ocr_best(crop_bgr, fast_mode=False):
    """
    OCR PIPELINE — speed-balanced:
    1. CNN OCR (fast ~50ms)  — exit immediately if conf ≥ 0.75
    2. EasyOCR:
       fast_mode=True  → 1 variant  (live camera  — ~1-2s total)
       fast_mode=False → 2 variants  (image upload — ~3-4s total)
    3. Vote
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0

    all_hits = []

    # ── Step 1: CNN OCR (fast path) ───────────────────────────────
    cnn_plate, cnn_conf = _cnn_read_plate(crop_bgr, conf_threshold=0.50)
    if cnn_plate:
        all_hits.append((cnn_plate, cnn_conf, 'CNN'))
        # Lowered early-exit threshold: 0.75 (was 0.82)
        # → exits sooner, saves 1-3s of EasyOCR time in most cases
        if cnn_conf >= 0.75:
            print(f"  [OCR] CNN fast-exit: {cnn_plate} ({cnn_conf*100:.1f}%)")
            return cnn_plate, cnn_conf

    # ── Step 2: EasyOCR with preprocessing ───────────────────────
    variants = preprocess_variants(crop_bgr)
    # fast_mode (live camera) → 1 variant only
    # normal (image upload)   → 2 variants max (V1 normal + V4 deblur)
    #   skip V3/V5 — they overlap and add 2-4s with minimal gain
    if fast_mode:
        variants_to_run = variants[:1]          # V1 only
    else:
        variants_to_run = [variants[0], variants[3]] if len(variants) > 3 \
                          else variants[:2]      # V1 + V4 (deblur)

    for i, variant in enumerate(variants_to_run):
        hits = ocr_single(variant)
        v_label = 'EasyOCR-v1' if i == 0 else 'EasyOCR-v4'
        for plate, conf in hits:
            all_hits.append((plate, conf, v_label))
        # Early EasyOCR exit if confident enough
        if hits and hits[0][1] >= 0.65:
            best = max(hits, key=lambda x: x[1])
            print(f"  [OCR] EasyOCR early exit {v_label}: {best[0]} ({best[1]*100:.1f}%)")
            if cnn_plate and cnn_conf >= best[1] * 0.90:
                return cnn_plate, cnn_conf
            return best[0], best[1]

    if not all_hits:
        return None, 0.0

    # ── Step 3: Vote across all sources ──────────────────────────
    counts = Counter(p for p, _, _ in all_hits)
    scores = defaultdict(float)
    src_bonus = {
        'CNN':          1.3,
        'EasyOCR-v1':   1.0,
        'EasyOCR-v4':   0.95,
    }
    for p, c, src in all_hits:
        scores[p] += c * src_bonus.get(src, 1.0)
    for p in scores:
        if counts[p] > 1:
            scores[p] *= counts[p]

    best_plate = max(scores, key=scores.__getitem__)
    best_conf  = min(1.0, scores[best_plate] /
                     max(counts[best_plate] ** 2, 1))
    print(f"  [VOTE] {best_plate} ({best_conf*100:.1f}%) "
          f"sources={counts[best_plate]}/{len(set(p for p,_,_ in all_hits))}")
    return best_plate, best_conf

# ══════════════════════════════════════════════════════════════
#  PLATE DETECTION
# ══════════════════════════════════════════════════════════════

def _pad(img, x, y, w, h, pad=12):
    ih, iw = img.shape[:2]
    return img[max(0, y-pad):min(ih, y+h+pad),
               max(0, x-pad):min(iw, x+w+pad)]

def detect_yolo(image):
    if not yolo_enabled:
        return []
    try:
        # Run at standard scale (max 800px width capped previously)
        results = []
        for imgsz in [640]:
            preds = yolo_model(
                image, conf=0.15, iou=0.45,
                verbose=False, imgsz=imgsz)[0]
            for box in preds.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cf = float(box.conf[0])
                p = 12
                x1 = max(0, x1-p); y1 = max(0, y1-p)
                x2 = min(image.shape[1], x2+p)
                y2 = min(image.shape[0], y2+p)
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    results.append((crop, (x1, y1, x2-x1, y2-y1), cf))
            if results:
                break  # found at smaller size — no need for larger
        results.sort(key=lambda r: r[2], reverse=True)
        # Deduplicate overlapping boxes
        deduped = []
        for crop, bbox, cf in results:
            x, y, w, h = bbox
            overlap = False
            for _, (bx, by, bw, bh), _ in deduped:
                if abs(x-bx) < 40 and abs(y-by) < 40:
                    overlap = True; break
            if not overlap:
                deduped.append((crop, bbox, cf))
        if deduped:
            print(f"  [YOLO] {len(deduped)} region(s), "
                  f"top={round(deduped[0][2]*100,1)}%")
        return deduped[:3]
    except Exception as e:
        print(f"  [YOLO] {e}")
        return []

def detect_frcnn(image):
    if not frcnn_enabled:
        return []
    try:
        import torch
        from torchvision import transforms
        t = transforms.ToTensor()(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            out = frcnn_model([t])[0]
        results = []
        ih, iw = image.shape[:2]
        for box, score in zip(out['boxes'].numpy(),
                               out['scores'].numpy()):
            if score < 0.40:
                continue
            x1, y1, x2, y2 = map(int, box)
            x1=max(0,x1-10); y1=max(0,y1-10)
            x2=min(iw,x2+10); y2=min(ih,y2+10)
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                results.append((crop, (x1,y1,x2-x1,y2-y1), score))
        results.sort(key=lambda r: r[2], reverse=True)
        if results:
            print(f"  [FRCNN] {len(results)} region(s)")
        return results[:2]
    except Exception as e:
        print(f"  [FRCNN] {e}")
        return []

def detect_contour(image):
    """Multi-scale contour detection — finds plates at different distances."""
    ih, iw = image.shape[:2]
    all_results = []

    # Try at original size AND 2x upscale (for long-distance)
    scales = [1.0]
    if iw < 1280:
        scales.append(2.0)

    for scale in scales:
        if scale != 1.0:
            img = cv2.resize(image, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_LANCZOS4)
        else:
            img = image

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur  = cv2.bilateralFilter(gray, 9, 17, 17)
        edged = cv2.Canny(blur, 20, 180)
        edged = cv2.dilate(
            edged, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2)))
        cnts, _ = cv2.findContours(edged, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:25]

        for c in cnts:
            if cv2.contourArea(c) < 150:
                continue
            x, y, w, h = cv2.boundingRect(c)
            ar = w / h if h > 0 else 0
            if (1.8 < ar < 8.0 and w > 30 and h > 8
                    and w < img.shape[1] * 0.95
                    and h < img.shape[0] * 0.6):
                # Convert back to original coords
                ox = int(x / scale); oy = int(y / scale)
                ow = int(w / scale); oh = int(h / scale)
                crop = _pad(image, ox, oy, ow, oh, 8)
                if crop.size > 0:
                    # Quality score
                    mb = np.mean(cv2.cvtColor(
                        crop, cv2.COLOR_BGR2GRAY))
                    score = (1.0 if 2.5 < ar < 5.5 else 0.5) + \
                            (0.5 if 120 < mb < 255 else 0.1)
                    all_results.append(
                        (crop, (ox, oy, ow, oh), score))

    # Deduplicate
    seen = set()
    deduped = []
    for crop, bbox, score in sorted(all_results,
                                     key=lambda r: r[2],
                                     reverse=True):
        x, y, w, h = bbox
        key = (x//15, y//15, w//15)
        if key not in seen:
            seen.add(key)
            deduped.append((crop, bbox, score))
    return deduped[:4]

def detect_whitebox(image):
    """White rectangle detection — reliable for Indian white plates."""
    ih, iw = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = []
    for thresh in [185, 200, 170]:
        _, wm = cv2.threshold(gray, thresh, 255,
                              cv2.THRESH_BINARY)
        wm = cv2.morphologyEx(
            wm, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (20, 7)))
        num, _, stats, _ = cv2.connectedComponentsWithStats(wm)
        for i in range(1, num):
            x, y, w, h = (stats[i, 0], stats[i, 1],
                          stats[i, 2], stats[i, 3])
            area = stats[i, 4]
            ar   = w / h if h > 0 else 0
            if (1.8 < ar < 8.5 and area > 400
                    and w > 50 and w < iw * 0.92):
                crop = _pad(image, x, y, w, h, 10)
                if crop.size > 0:
                    results.append((crop, (x, y, w, h), ar))
    return results[:4]


# ══════════════════════════════════════════════════════════════
#  MASTER PIPELINE
# ══════════════════════════════════════════════════════════════

def _calc_iou(boxA, boxB):
    """Calculate IoU between two (x,y,w,h) boxes."""
    ax1, ay1 = boxA[0], boxA[1]
    ax2, ay2 = boxA[0]+boxA[2], boxA[1]+boxA[3]
    bx1, by1 = boxB[0], boxB[1]
    bx2, by2 = boxB[0]+boxB[2], boxB[1]+boxB[3]
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0: return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return round(inter / union, 3) if union > 0 else 0.0


def _mem_mb():
    """Current process RSS memory in MB."""
    try:
        import psutil, os
        return round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 1)
    except Exception:
        return None


def process_image(image, fast_scan=False):
    """
    DUAL-MODEL PIPELINE v7
    ══════════════════════
    Runs YOLOv8 and Faster R-CNN INDEPENDENTLY with separate timers.
    Each model gets its own:
      • detection time (ms)
      • detection confidence (%)
      • bounding box
      • OCR result + OCR confidence
      • IoU vs the other model's box
      • estimated FPS
      • memory snapshot (MB)

    The WINNER (highest OCR confidence) is used for gate/notification.
    Full comparison metrics returned in meta['model_comparison'].
    """
    t0 = time.time()
    annotated = image.copy()
    ih, iw    = image.shape[:2]

    mem_before = _mem_mb()

    # ══════════════════════════════════════════════════════════
    #  MODEL A — YOLOv8
    # ══════════════════════════════════════════════════════════
    yolo_result = {
        'model':       'YOLO11',
        'detected':    False,
        'det_conf':    0.0,
        'det_time_ms': 0,
        'fps':         0.0,
        'bbox':        None,
        'plate':       None,
        'ocr_conf':    0.0,
        'iou':         0.0,
        'memory_mb':   None,
        'winner':      False,
    }

    if yolo_enabled:
        t_yolo = time.time()
        yolo_regions = detect_yolo(image)
        yolo_result['det_time_ms'] = int((time.time() - t_yolo) * 1000)
        yolo_result['fps']         = round(1000 / max(yolo_result['det_time_ms'], 1), 1)
        yolo_result['memory_mb']   = _mem_mb()

        if yolo_regions:
            crop, bbox, cf = yolo_regions[0]
            yolo_result['detected']  = True
            yolo_result['det_conf']  = round(cf * 100, 1)
            yolo_result['bbox']      = bbox
            plate_y, conf_y          = ocr_best(crop, fast_mode=fast_scan)
            yolo_result['plate']     = plate_y
            yolo_result['ocr_conf']  = round(conf_y * 100, 1)
            print(f"  [YOLO] {plate_y} det={yolo_result['det_conf']}% "
                  f"ocr={yolo_result['ocr_conf']}% {yolo_result['det_time_ms']}ms")
    else:
        print("  [YOLO] not loaded")

    # ══════════════════════════════════════════════════════════
    #  MODEL B — Faster R-CNN
    # ══════════════════════════════════════════════════════════
    frcnn_result = {
        'model':       'Faster R-CNN (ResNet-50 FPN)',
        'detected':    False,
        'det_conf':    0.0,
        'det_time_ms': 0,
        'fps':         0.0,
        'bbox':        None,
        'plate':       None,
        'ocr_conf':    0.0,
        'iou':         0.0,
        'memory_mb':   None,
        'winner':      False,
    }

    if frcnn_enabled:
        t_frcnn = time.time()
        frcnn_regions = detect_frcnn(image)
        frcnn_result['det_time_ms'] = int((time.time() - t_frcnn) * 1000)
        frcnn_result['fps']         = round(1000 / max(frcnn_result['det_time_ms'], 1), 1)
        frcnn_result['memory_mb']   = _mem_mb()

        if frcnn_regions:
            crop_f, bbox_f, cf_f = frcnn_regions[0]
            frcnn_result['detected']  = True
            frcnn_result['det_conf']  = round(float(cf_f) * 100, 1) if isinstance(cf_f, float) else 0.0
            frcnn_result['bbox']      = bbox_f
            plate_f, conf_f_ocr       = ocr_best(crop_f, fast_mode=fast_scan)
            frcnn_result['plate']     = plate_f
            frcnn_result['ocr_conf']  = round(conf_f_ocr * 100, 1)
            print(f"  [FRCNN] {plate_f} det={frcnn_result['det_conf']}% "
                  f"ocr={frcnn_result['ocr_conf']}% {frcnn_result['det_time_ms']}ms")
    else:
        print("  [FRCNN] not loaded — train with train_fasterrcnn.py")

    # ── Compute IoU between the two bounding boxes ────────────
    if yolo_result['bbox'] and frcnn_result['bbox']:
        iou_val = _calc_iou(yolo_result['bbox'], frcnn_result['bbox'])
        yolo_result['iou']  = iou_val
        frcnn_result['iou'] = iou_val

    # ── Pick WINNER by OCR confidence ─────────────────────────
    plate_text = None
    confidence = 0.0
    found_bbox = None
    winner_src = 'none'

    both = [(yolo_result, 'YOLO'), (frcnn_result, 'FRCNN')]
    for res, src in sorted(both, key=lambda x: x[0]['ocr_conf'], reverse=True):
        if res['plate']:
            plate_text = res['plate']
            confidence = res['ocr_conf'] / 100.0
            found_bbox = res['bbox']
            winner_src = src
            res['winner'] = True
            print(f"  [WINNER] {src} → {plate_text} ({res['ocr_conf']}%)")
            break

    # ── CV fallback if both neural models failed ──────────────
    if not plate_text:
        print("  [PIPE] Both neural models failed → CV detectors")
        cv_regions = detect_contour(image) + detect_whitebox(image)
        cv_regions.sort(key=lambda r: r[2], reverse=True)
        for crop, bbox, _ in cv_regions[:3]:
            plate_text, confidence = ocr_best(crop, fast_mode=fast_scan)
            if plate_text:
                found_bbox = bbox
                winner_src = 'CV'
                break

    # ── ROI last resort ───────────────────────────────────────
    if not plate_text:
        print("  [PIPE] CV failed → bottom-half ROI")
        roi = image[int(ih * 0.40):, :]
        plate_text, confidence = ocr_best(roi, fast_mode=fast_scan)
        if plate_text:
            confidence *= 0.75
            winner_src = 'ROI'

    # ── Analyze Vehicle Attributes (Color & Type) ─────────────
    v_color, v_type = vehicle_analyzer.analyze(image)
    if plate_text:
        print(f"  [VEHICLE] {v_color} {v_type}")
    else:
        print(f"  [VEHICLE] No plate, but identified as: {v_color} {v_type}")

    total_ms   = int((time.time() - t0) * 1000)
    mem_after  = _mem_mb()
    mem_used   = round(mem_after - mem_before, 1) if mem_before and mem_after else None

    print(f"  [TIME] Total {total_ms}ms")

    # ── Annotate image ────────────────────────────────────────
    COL_YOLO  = (0, 255, 100)    # green
    COL_FRCNN = (255, 100, 0)    # blue-orange

    if yolo_result['bbox']:
        x, y, w, h = yolo_result['bbox']
        cv2.rectangle(annotated, (x,y), (x+w,y+h), COL_YOLO, 2)
        cv2.putText(annotated, f"YOLO {yolo_result['det_conf']}%",
                    (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_YOLO, 1)

    if frcnn_result['bbox'] and frcnn_result['bbox'] != yolo_result['bbox']:
        x, y, w, h = frcnn_result['bbox']
        cv2.rectangle(annotated, (x,y), (x+w,y+h), COL_FRCNN, 2)
        cv2.putText(annotated, f"FRCNN {frcnn_result['det_conf']}%",
                    (x, y+h+14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_FRCNN, 1)

    if plate_text and found_bbox:
        x, y, w, h = found_bbox
        COL = COL_YOLO if winner_src == 'YOLO' else COL_FRCNN
        lbl = f" {plate_text} "
        fs  = min(1.0, max(0.5, w / 220))
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, fs, 2)
        cv2.rectangle(annotated, (x-1, y-th-18), (x+tw+2, y-1), COL, -1)
        cv2.putText(annotated, lbl, (x, y-5),
                    cv2.FONT_HERSHEY_DUPLEX, fs, (0,0,0), 2)
        bw = int(w * min(1.0, confidence))
        cv2.rectangle(annotated, (x, y+h+4), (x+w, y+h+9), (40,40,40), -1)
        cv2.rectangle(annotated, (x, y+h+4), (x+bw, y+h+9), COL, -1)
    elif plate_text:
        lbl = f"  {plate_text}  "
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
        cv2.rectangle(annotated, (0,0), (tw+10, th+20), COL_YOLO, -1)
        cv2.putText(annotated, lbl, (5, th+6),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,0), 2)

    if not plate_text:
        print("  [RESULT] ✗ No plate detected")

    # ── Build full comparison meta ────────────────────────────
    meta = {
        'detectors': {
            'd0': len(yolo_result['bbox'] and [1] or []),
            'd1': 0, 'd2': 0, 'd3': 0, 'd4': 0, 'd5': 0,
        },
        'vehicle_color':  v_color,
        'vehicle_type':   v_type,
        'regions_scored': int(yolo_result['detected']) + int(frcnn_result['detected']),
        'candidates':     {plate_text: round(confidence, 3)} if plate_text else {},
        'time_ms':        total_ms,
        'winner_sources': [winner_src],

        # Legacy dashboard fields
        'model_a': {
            'plate': yolo_result['plate'],
            'conf':  yolo_result['ocr_conf'],
        } if yolo_result['plate'] else None,
        'model_b': {
            'plate': frcnn_result['plate'],
            'conf':  frcnn_result['ocr_conf'],
        } if frcnn_result['plate'] else None,
        'regions_a': int(yolo_result['detected']),
        'regions_b': int(frcnn_result['detected']),

        # ── FULL COMPARISON TABLE ──────────────────────────────
        'model_comparison': {
            'yolo': {
                'model':          yolo_result['model'],
                'loaded':         yolo_enabled,
                'detected':       yolo_result['detected'],
                'det_conf_pct':   yolo_result['det_conf'],
                'det_time_ms':    yolo_result['det_time_ms'],
                'fps':            yolo_result['fps'],
                'plate':          yolo_result['plate'],
                'ocr_conf_pct':   yolo_result['ocr_conf'],
                'iou':            yolo_result['iou'],
                'memory_mb':      yolo_result['memory_mb'],
                'winner':         yolo_result['winner'],
            },
            'frcnn': {
                'model':          frcnn_result['model'],
                'loaded':         frcnn_enabled,
                'detected':       frcnn_result['detected'],
                'det_conf_pct':   frcnn_result['det_conf'],
                'det_time_ms':    frcnn_result['det_time_ms'],
                'fps':            frcnn_result['fps'],
                'plate':          frcnn_result['plate'],
                'ocr_conf_pct':   frcnn_result['ocr_conf'],
                'iou':            frcnn_result['iou'],
                'memory_mb':      frcnn_result['memory_mb'],
                'winner':         frcnn_result['winner'],
            },
            'iou_between_models': yolo_result['iou'],
            'winner':             winner_src,
            'total_time_ms':      total_ms,
            'memory_delta_mb':    mem_used,
        },
    }

    if plate_text:
        print(f"  [RESULT] ✅ {plate_text} — {round(confidence*100,1)}%  "
              f"winner={winner_src}  {total_ms}ms")

    return plate_text, confidence, annotated, meta


# ══════════════════════════════════════════════════════════════
#  GATE
# ══════════════════════════════════════════════════════════════
class GateController:
    def __init__(self):
        self.state = 'CLOSED'
        self.last_action = None
        self.last_plate = None

    def trigger_open(self, plate, reason=''):
        self.last_plate  = plate
        self.last_action = datetime.now().isoformat()
        mode = SYSTEM_CONFIG['gate_mode']
        try:
            if mode == 'simulate':
                self.state = 'OPEN'
                print(f"  [GATE] OPEN — {plate}")
                threading.Timer(
                    SYSTEM_CONFIG['gate_open_ms'] / 1000,
                    self._close).start()
                return True, 'Gate opened (simulated)'
            elif mode == 'webhook':
                import requests as rq
                r = rq.post(
                    SYSTEM_CONFIG['gate_webhook'],
                    json={'plate': plate, 'reason': reason},
                    timeout=3)
                if r.ok:
                    self.state = 'OPEN'
                    threading.Timer(
                        SYSTEM_CONFIG['gate_open_ms'] / 1000,
                        self._close).start()
                    return True, 'Gate opened via webhook'
                return False, f'Webhook error {r.status_code}'
            elif mode == 'serial':
                import serial
                with serial.Serial('COM3', 9600, timeout=1) as s:
                    s.write(b'OPEN\n')
                self.state = 'OPEN'
                threading.Timer(
                    SYSTEM_CONFIG['gate_open_ms'] / 1000,
                    self._close).start()
                return True, 'Gate opened via serial'
        except Exception as e:
            self.state = 'ERROR'
            return False, f'Gate error: {e}'
        return False, 'Unknown mode'

    def _close(self):
        self.state = 'CLOSED'
        print("  [GATE] CLOSED")

    def status(self):
        return {
            'state':       self.state,
            'last_plate':  self.last_plate,
            'last_action': self.last_action,
            'mode':        SYSTEM_CONFIG['gate_mode'],
        }

gate = GateController()


# ══════════════════════════════════════════════════════════════
#  NOTIFIER
# ══════════════════════════════════════════════════════════════
class Notifier:
    def send(self, vehicle, event_type, plate, confidence):
        channels = vehicle.get('notify_channels', [])
        if not channels:
            return {}
        msg = (
            f"{'🟢' if event_type=='ENTRY' else '🔴'} "
            f"ALPR Alert — {event_type}\n"
            f"Plate : {plate}\n"
            f"Owner : {vehicle['owner_name']}\n"
            f"Flat  : {vehicle.get('flat_number', '-')}\n"
            f"Conf  : {round(confidence*100, 1)}%\n"
            f"Time  : {datetime.now().strftime('%d %b %Y, %I:%M %p')}"
        )
        results = {}
        for ch in channels:
            try:
                if ch == 'telegram':
                    results['telegram'] = self._tg(vehicle, msg)
                elif ch == 'whatsapp':
                    results['whatsapp'] = self._wa(vehicle, msg)
                elif ch == 'email':
                    results['email'] = self._email(
                        vehicle, event_type, plate)
            except Exception as e:
                results[ch] = f'error:{e}'
        return results

    def _tg(self, v, msg):
        tok  = SYSTEM_CONFIG['telegram_token']
        chat = v.get('telegram_chat_id', '')
        if not tok or not chat:
            return 'skipped'
        import requests as rq
        r = rq.post(
            f'https://api.telegram.org/bot{tok}/sendMessage',
            json={'chat_id': chat, 'text': msg}, timeout=5)
        return 'sent' if r.ok else f'error {r.status_code}'

    def _wa(self, v, msg):
        sid   = SYSTEM_CONFIG['twilio_sid']
        tok   = SYSTEM_CONFIG['twilio_token']
        phone = v.get('whatsapp_number') or v.get('owner_phone', '')
        if not sid or not tok or not phone:
            return 'skipped'
        from twilio.rest import Client
        to = (f"whatsapp:{phone}"
              if not phone.startswith('whatsapp:') else phone)
        Client(sid, tok).messages.create(
            body=msg, from_=SYSTEM_CONFIG['twilio_from'], to=to)
        return 'sent'

    def _email(self, v, et, plate):
        # Send to owner's email if set, otherwise send to default
        to     = v.get('owner_email', '').strip() or 'ragavendharlh@gmail.com'
        sender = SYSTEM_CONFIG['email_sender']
        pw     = SYSTEM_CONFIG['email_password']

        if not sender or not pw:
            print("  [EMAIL] ❌ No sender/password configured")
            return 'skipped:no_credentials'

        icon      = '🟢' if et == 'ENTRY' else '🔴'
        color     = '#16a34a' if et == 'ENTRY' else '#dc2626'
        timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
        flat      = v.get('flat_number', '-')
        vtype     = v.get('vehicle_type', 'Car')

        html = f"""
        <div style="font-family:Arial,sans-serif;max-width:480px;
                    margin:auto;border:1px solid #e5e7eb;
                    border-radius:10px;overflow:hidden;">
          <div style="background:#070b10;padding:20px;text-align:center;">
            <h2 style="color:#00d4ff;margin:0;letter-spacing:2px;">
              🚗 ALPR GATE ALERT
            </h2>
          </div>
          <div style="background:{color};padding:10px;text-align:center;">
            <span style="color:#fff;font-size:18px;font-weight:bold;">
              {icon} VEHICLE {et}
            </span>
          </div>
          <div style="padding:24px;background:#fff;">
            <table style="width:100%;border-collapse:collapse;">
              <tr style="background:#f0f9ff;">
                <td style="padding:10px 14px;font-weight:bold;
                           color:#374151;width:40%;">🔢 Plate Number</td>
                <td style="padding:10px 14px;font-size:17px;
                           font-weight:bold;color:#111;">{plate}</td>
              </tr>
              <tr>
                <td style="padding:10px 14px;font-weight:bold;
                           color:#374151;">👤 Owner</td>
                <td style="padding:10px 14px;">{v["owner_name"]}</td>
              </tr>
              <tr style="background:#f0f9ff;">
                <td style="padding:10px 14px;font-weight:bold;
                           color:#374151;">🏠 Flat</td>
                <td style="padding:10px 14px;">{flat}</td>
              </tr>
              <tr>
                <td style="padding:10px 14px;font-weight:bold;
                           color:#374151;">🚙 Vehicle</td>
                <td style="padding:10px 14px;">{vtype}</td>
              </tr>
              <tr style="background:#f0f9ff;">
                <td style="padding:10px 14px;font-weight:bold;
                           color:#374151;">🕐 Time</td>
                <td style="padding:10px 14px;">{timestamp}</td>
              </tr>
            </table>
          </div>
          <div style="background:#f9fafb;padding:12px;
                      text-align:center;border-top:1px solid #e5e7eb;">
            <span style="color:#9ca3af;font-size:12px;">
              Sent by ALPR Gate Intelligence System
            </span>
          </div>
        </div>
        """

        try:
            m = MIMEMultipart('alternative')
            m['Subject'] = f"ALPR Gate Alert: {plate} — {et} at {timestamp}"
            m['From']    = sender
            m['To']      = to
            m.attach(MIMEText(html, 'html'))
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as srv:
                srv.login(sender, pw)
                srv.sendmail(sender, to, m.as_string())
            print(f"  [EMAIL] ✅ Sent to {to} for plate {plate}")
            return 'sent'
        except smtplib.SMTPAuthenticationError:
            print("  [EMAIL] ❌ Auth failed — check email and app password")
            return 'error:auth_failed'
        except Exception as e:
            print(f"  [EMAIL] ❌ Error: {e}")
            return f'error:{e}'

notifier = Notifier()


# ══════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════
class VehicleDatabase:
    def __init__(self):
        self.vehicles      = {}
        self.events        = []
        self.notifications = []
        self._lock         = threading.Lock()
        self.load()

    def register(self, data):
        with self._lock:
            plate = re.sub(r'[^A-Z0-9]', '', data['plate'].upper())
            if not plate:
                raise ValueError("Invalid plate")
            self.vehicles[plate] = {
                'plate':           plate,
                'owner_name':      data.get('owner_name', '').strip(),
                'owner_phone':     data.get('owner_phone', '').strip(),
                'owner_email':     data.get('owner_email', '').strip(),
                'vehicle_type':    data.get('vehicle_type', 'Car'),
                'color':           data.get('color', '').strip(),
                'flat_number':     data.get('flat_number', '').strip(),
                'notify_channels': data.get('notify_channels', []),
                'telegram_chat_id':data.get('telegram_chat_id','').strip(),
                'whatsapp_number': data.get('whatsapp_number','').strip(),
                'registered_at':   datetime.now().isoformat(),
            }
            self.save()
            print(f"  [DB] Registered: {plate}")
            return self.vehicles[plate]

    def get(self, plate):
        clean = re.sub(r'[^A-Z0-9]', '', plate.upper())
        if clean in self.vehicles:
            return self.vehicles[clean]
        norm = normalize_fuzzy(clean)
        for p, v in self.vehicles.items():
            if normalize_fuzzy(p) == norm:
                return v
        return None

    def update(self, plate, data):
        with self._lock:
            clean = re.sub(r'[^A-Z0-9]', '', plate.upper())
            if clean not in self.vehicles:
                raise ValueError(f"{clean} not found")
            v = self.vehicles[clean]
            for f in ['owner_name','owner_phone','owner_email',
                      'vehicle_type','color','flat_number',
                      'notify_channels','telegram_chat_id',
                      'whatsapp_number']:
                if f in data:
                    v[f] = data[f]
            v['updated_at'] = datetime.now().isoformat()
            self.save()
            return v

    def delete(self, plate):
        with self._lock:
            plate = re.sub(r'[^A-Z0-9]', '', plate.upper())
            if plate in self.vehicles:
                del self.vehicles[plate]
                self.save()
                return True
            return False

    def log_event(self, plate, event_type, confidence=0.0,
                  image_b64=None, gate_result=None,
                  notif_results=None):
        with self._lock:
            plate   = re.sub(r'[^A-Z0-9]', '', plate.upper())
            vehicle = self.get(plate)
            event   = {
                'id':           len(self.events) + 1,
                'plate':        plate,
                'event_type':   event_type,
                'timestamp':    datetime.now().isoformat(),
                'confidence':   round(confidence * 100, 1),
                'is_registered':vehicle is not None,
                'owner_name':   (vehicle['owner_name']
                                 if vehicle else 'Unknown'),
                'flat_number':  (vehicle.get('flat_number', '')
                                 if vehicle else ''),
                'gate_action':  gate_result,
                'notif_results':notif_results or {},
                'image':        image_b64,
            }
            self.events.append(event)
            self.notifications.insert(0, {
                'id':          len(self.notifications) + 1,
                'type':        event_type,
                'plate':       plate,
                'owner':       (vehicle['owner_name']
                                if vehicle else 'UNKNOWN'),
                'flat':        (vehicle.get('flat_number', '')
                                if vehicle else ''),
                'registered':  vehicle is not None,
                'gate':        gate_result,
                'notif_status':notif_results or {},
                'timestamp':   datetime.now().isoformat(),
                'read':        False,
            })
            self.notifications = self.notifications[:100]
            self.save()
            return event

    def save(self):
        try:
            tmp = DB_FILE + '.tmp'
            with open(tmp, 'w') as f:
                json.dump({
                    'vehicles':      self.vehicles,
                    'events':        self.events,
                    'notifications': self.notifications,
                }, f, indent=2)
            os.replace(tmp, DB_FILE)
        except Exception as e:
            print(f"  [DB ERR] {e}")

    def load(self):
        try:
            if os.path.exists(DB_FILE):
                with open(DB_FILE) as f:
                    d = json.load(f)
                self.vehicles      = d.get('vehicles', {})
                self.events        = d.get('events', [])
                self.notifications = d.get('notifications', [])
                print(f"  [DB] {len(self.vehicles)} vehicles, "
                      f"{len(self.events)} events")
            else:
                print("  [DB] New database")
        except Exception as e:
            print(f"  [DB ERR] {e}")

    def stats(self):
        today = datetime.now().date().isoformat()
        te    = [e for e in self.events
                 if e['timestamp'].startswith(today)]
        return {
            'total_registered': len(self.vehicles),
            'total_events':     len(self.events),
            'today_entries':    sum(1 for e in te
                                    if e['event_type'] == 'ENTRY'),
            'today_exits':      sum(1 for e in te
                                    if e['event_type'] == 'EXIT'),
            'today_unknown':    sum(1 for e in te
                                    if not e['is_registered']),
        }

db = VehicleDatabase()


def handle_detection(plate, confidence, event_type,
                     image_b64=None):
    vehicle       = db.get(plate)
    gate_result   = None
    notif_results = {}
    if vehicle:
        ok, msg     = gate.trigger_open(
            plate, f"{event_type} — {vehicle['owner_name']}")
        gate_result   = msg
        notif_results = notifier.send(
            vehicle, event_type, plate, confidence)
    else:
        gate_result = 'blocked (unregistered)'
        print(f"  [ALERT] Unknown: {plate}")
    event = db.log_event(plate, event_type, confidence,
                         image_b64, gate_result, notif_results)
    return event, vehicle, gate_result, notif_results


# ══════════════════════════════════════════════════════════════
#  CAMERA
# ══════════════════════════════════════════════════════════════
class CameraStream:
    def __init__(self):
        self.cap     = None
        self.frame   = None
        self.running = False
        self.lock    = threading.Lock()

    def start(self, cam_id=0):
        if self.running:
            self.stop()
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(cam_id, backend)
            if not cap.isOpened():
                cap.release()
                continue
            for tw, th in [(1920,1080),(1280,720),(640,480)]:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  tw)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, th)
                gw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                if gw >= tw * 0.9:
                    break
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for _ in range(3):
                cap.read()
            gh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.cap     = cap
            self.running = True
            print(f"  [CAM] ID={cam_id} {gw}x{gh}")
            threading.Thread(target=self._loop,
                             daemon=True).start()
            return True
        return False

    def stop(self):
        self.running = False
        time.sleep(0.15)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.frame = None

    def _loop(self):
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
            time.sleep(0.02)

    def get_frame(self):
        with self.lock:
            return (self.frame.copy()
                    if self.frame is not None else None)

    def get_jpeg(self):
        f = self.get_frame()
        if f is None:
            return None
        _, j = cv2.imencode(
            '.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return j.tobytes()

    def scan_frame(self):
        f = self.get_frame()
        if f is None:
            return None, 0, None, {}
        h, w = f.shape[:2]

        # For live camera: upscale small frames so YOLO can detect plate regions
        # Cap large frames to avoid CPU timeout (>22s) on slow machines
        TARGET_W = 640
        MAX_W    = 960
        if w < TARGET_W:
            sc = TARGET_W / w
            f  = cv2.resize(f, None, fx=sc, fy=sc, interpolation=cv2.INTER_LANCZOS4)
        elif w > MAX_W:
            sc = MAX_W / w
            f  = cv2.resize(f, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)

        plate, conf, ann, meta = process_image(f, fast_scan=True)

        if ann.shape[1] > 1280:
            ann = cv2.resize(
                ann,
                (1280, int(ann.shape[0] * 1280 / ann.shape[1])),
                interpolation=cv2.INTER_AREA)

        _, j = cv2.imencode('.jpg', ann, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return plate, conf, base64.b64encode(j).decode(), meta

camera = CameraStream()


# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    p = os.path.join(BASE_DIR, 'dashboard.html')
    with open(p, 'r', encoding='utf-8') as f:
        return Response(f.read(), mimetype='text/html')

@app.route('/api/health')
def health():
    return jsonify({
        'status':         'running',
        'timestamp':      datetime.now().isoformat(),
        'total_vehicles': len(db.vehicles),
        'total_events':   len(db.events),
        'camera_active':  camera.running,
        'gate':           gate.status(),
    })

@app.route('/api/statistics')
def statistics():
    try:
        s = db.stats()
        s['gate']      = gate.status()
        s['timestamp'] = datetime.now().isoformat()
        return jsonify(s)
    except Exception as e:
        return jsonify({
            'error': str(e), 'total_registered': 0,
            'total_events': 0, 'today_entries': 0,
            'today_exits': 0, 'today_unknown': 0,
        })

@app.route('/api/register-vehicle', methods=['POST'])
def register_vehicle():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({'success':False,'error':'No data'}),400
        if not data.get('plate_number'):
            return jsonify({'success':False,
                            'error':'plate_number required'}),400
        if not data.get('owner_name'):
            return jsonify({'success':False,
                            'error':'owner_name required'}),400
        v = db.register({
            'plate':           data['plate_number'],
            'owner_name':      data['owner_name'],
            'owner_phone':     data.get('owner_phone',''),
            'owner_email':     data.get('owner_email',''),
            'vehicle_type':    data.get('vehicle_type','Car'),
            'color':           data.get('color',''),
            'flat_number':     data.get('flat_number',''),
            'notify_channels': data.get('notify_channels',[]),
            'telegram_chat_id':data.get('telegram_chat_id',''),
            'whatsapp_number': data.get('whatsapp_number',''),
        })
        return jsonify({'success':True,
                        'message':f"Vehicle {v['plate']} registered",
                        'vehicle':v})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success':False,'error':str(e)}),500

@app.route('/api/all-vehicles')
def all_vehicles():
    return jsonify({'total':len(db.vehicles),
                    'vehicles':list(db.vehicles.values())})

@app.route('/api/vehicle/<plate>', methods=['GET'])
def get_vehicle(plate):
    v = db.get(plate)
    return jsonify(v) if v else (jsonify({'error':'Not found'}),404)

@app.route('/api/vehicle/<plate>', methods=['DELETE'])
def delete_vehicle(plate):
    return jsonify({'success':db.delete(plate)})

@app.route('/api/vehicle/<plate>', methods=['PUT'])
def update_vehicle(plate):
    try:
        data = request.get_json(force=True, silent=True) or {}
        v    = db.update(plate, data)
        return jsonify({'success':True,'vehicle':v})
    except ValueError as e:
        return jsonify({'success':False,'error':str(e)}),404
    except Exception as e:
        return jsonify({'success':False,'error':str(e)}),500

@app.route('/api/scan-image', methods=['POST'])
def scan_image():
    try:
        image      = None
        event_type = 'ENTRY'
        if (request.content_type
                and 'multipart' in request.content_type):
            file = request.files.get('image')
            if not file:
                return jsonify({'success':False,'error':'No file'}),400
            npimg  = np.frombuffer(file.read(), np.uint8)
            image  = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            event_type = request.form.get('event_type','ENTRY')
        else:
            data     = request.get_json(force=True,silent=True) or {}
            event_type = data.get('event_type','ENTRY')
            img_data   = data.get('image_data','')
            if not img_data:
                return jsonify({'success':False,
                                'error':'No image_data'}),400
            if ',' in img_data:
                img_data = img_data.split(',')[1]
            npimg = np.frombuffer(
                base64.b64decode(img_data), np.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'success':False,
                            'error':'Cannot decode image'}),400

        plate, conf, ann, meta = process_image(image)
        _, j = cv2.imencode('.jpg', ann,
                             [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(j).decode()

        if plate:
            event, vehicle, gate_result, notif_results = \
                handle_detection(plate, conf, event_type, img_b64)
            # Store comparison history
            if 'model_comparison' in meta:
                _comparison_history.append({
                    **meta['model_comparison'],
                    'timestamp': datetime.now().isoformat(),
                    'plate':     plate,
                })
            return jsonify({
                'success':       True,
                'plate':         plate,
                'confidence':    round(conf*100, 1),
                'vehicle':       vehicle,
                'event':         event,
                'gate':          gate_result,
                'notifications': notif_results,
                'image':         img_b64,
                'model_meta':    meta,
            })
        return jsonify({
            'success':    False,
            'error':      'No plate detected',
            'image':      img_b64,
            'tip':        'Move closer · Better lighting · Avoid angle',
            'model_meta': meta,
        })
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc()  # noqa: E702
        return jsonify({'success':False,'error':str(e)}),500

@app.route('/api/camera/list')
def list_cameras():
    found = []
    for i in range(6):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                continue
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                continue
            nat_h, nat_w = frame.shape[:2]
            cap.release()
            label = ('Default Webcam' if i == 0
                     else f'Camera {i} (DroidCam?)')
            found.append({
                'id': i, 'resolution': f'{nat_w}x{nat_h}',
                'max_resolution': f'{nat_w}x{nat_h}',
                'label': f'{i} — {label} {nat_w}x{nat_h}',
            })
        except Exception:  # noqa: BLE001
            pass
    return jsonify({'cameras': found, 'count': len(found)})

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    data   = request.get_json(force=True, silent=True) or {}
    cam_id = int(data.get('camera_id', 0))
    ok     = camera.start(cam_id)
    return jsonify({
        'success': ok,
        'message': (f'Camera {cam_id} started'
                    if ok else 'Failed to open camera'),
    })

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    camera.stop()
    return jsonify({'success': True})

@app.route('/api/camera/status')
def camera_status():
    return jsonify({'running':    camera.running,
                    'has_frame':  camera.frame is not None})

@app.route('/api/camera/feed')
def camera_feed():
    def generate():
        while True:
            j = camera.get_jpeg()
            if j:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n'
                       + j + b'\r\n')
            time.sleep(0.033)
    return Response(generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/scan', methods=['POST'])
def camera_scan():
    if not camera.running:
        return jsonify({
            'success': False,
            'error':   'Camera not started — click START first',
        }), 400

    data       = request.get_json(force=True, silent=True) or {}
    event_type = data.get('event_type', 'ENTRY')

    import concurrent.futures
    result_box = {}

    def _do():
        p, c, img, m = camera.scan_frame()
        result_box.update({'plate': p, 'conf': c, 'img': img, 'meta': m})

    ex  = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(_do)
    try:
        fut.result(timeout=30)
    except concurrent.futures.TimeoutError:
        ex.shutdown(wait=False)
        return jsonify({
            'success':    False,
            'error':      'Scan timed out — try moving closer or improving lighting',
            'model_meta': {},
        }), 200
    ex.shutdown(wait=False)

    plate   = result_box.get('plate')
    conf    = result_box.get('conf', 0)
    img_b64 = result_box.get('img')
    meta    = result_box.get('meta', {})

    if plate:
        event, vehicle, gate_result, notif_results = \
            handle_detection(plate, conf, event_type, img_b64)
        return jsonify({
            'success':       True,
            'plate':         plate,
            'confidence':    round(conf * 100, 1),
            'vehicle':       vehicle,
            'event':         event,
            'gate':          gate_result,
            'notifications': notif_results,
            'image':         img_b64,
            'model_meta':    meta,
        })
    return jsonify({
        'success':    False,
        'error':      'No plate detected',
        'image':      img_b64,
        'model_meta': meta,
        'tip':        'Move closer · Good lighting · Hold still',
    })


@app.route('/api/events')
def get_events():
    limit = int(request.args.get('limit', 50))
    return jsonify({
        'total':  len(db.events),
        'events': list(reversed(db.events))[:limit],
    })

@app.route('/api/log-event', methods=['POST'])
def log_event():
    try:
        data  = request.get_json(force=True, silent=True) or {}
        plate = data.get('plate', '').strip().upper()
        if not plate:
            return jsonify({'success':False,'error':'plate required'}),400
        event, vehicle, gate_result, notif_results = \
            handle_detection(
                plate,
                float(data.get('confidence', 1.0)),
                data.get('event_type', 'ENTRY'))
        return jsonify({'success':True,'event':event,
                        'gate':gate_result,
                        'notifications':notif_results})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)}),400

@app.route('/api/notifications')
def get_notifications():
    unread = sum(1 for n in db.notifications if not n['read'])
    return jsonify({'notifications': db.notifications[:30],
                    'unread_count':  unread})

@app.route('/api/notifications/mark-read', methods=['POST'])
def mark_read():
    for n in db.notifications:
        n['read'] = True
    db.save()
    return jsonify({'success': True})

@app.route('/api/gate/status')
def gate_status():
    return jsonify(gate.status())

@app.route('/api/gate/open', methods=['POST'])
def gate_open_manual():
    data    = request.get_json(force=True, silent=True) or {}
    ok, msg = gate.trigger_open(
        data.get('plate', 'MANUAL'), 'Manual override')
    return jsonify({'success':ok,'message':msg,'gate':gate.status()})

@app.route('/api/gate/config', methods=['GET', 'POST'])
def gate_config():
    keys = ('gate_mode', 'gate_webhook', 'gate_open_ms')
    if request.method == 'POST':
        for k, v in (request.get_json(
                force=True, silent=True) or {}).items():
            if k in SYSTEM_CONFIG:
                SYSTEM_CONFIG[k] = v
        return jsonify({'success': True})
    return jsonify({k: SYSTEM_CONFIG[k] for k in keys})


@app.route('/api/cnn-status')
def cnn_status():
    return jsonify({
        'cnn_enabled':  _cnn_enabled,
        'cnn_chars':    len(_cnn_ocr_chars) if _cnn_ocr_chars else 0,
        'description':  'Compact CNN trained on Indian plate character fonts',
        'pipeline':     'CNN-OCR (fast) → EasyOCR (fallback)',
    })

@app.route('/api/engine-status')
def engine_status():
    yn = (next((os.path.basename(p) for p in _YOLO_PATHS
                if os.path.exists(p)), '')
          if yolo_enabled else None)
    return jsonify({
        'yolo_enabled':    yolo_enabled,
        'yolo_model':      yn,
        'frcnn_enabled':   frcnn_enabled,
        'cnn_enabled':     _cnn_enabled,
        'easyocr':         True,
        'ocr_decoder':     'greedy',
        'ocr_variants':    3,
        'ocr_canvas':      2560,   # matches actual ocr_single() setting
        'ocr_mag_ratio':   2.0,    # matches actual ocr_single() setting
        'pipeline':        'CNN (fast) → DUAL-MODEL: YOLO + Faster R-CNN → OCR vote',
        'long_distance':   'super-resolution upscale enabled',
        'detectors': {
            'D0_CNN':     _cnn_enabled,
            'D0_YOLO':    yolo_enabled,
            'D0b_FRCNN':  frcnn_enabled,
            'D1_Contour': True,
            'D4_White':   True,
        },
    })


# ── last 10 model comparison results (stored per scan) ────────
_comparison_history = []

@app.route('/api/model-comparison')
def model_comparison():
    """
    Returns the last 10 dual-model comparison results.
    Each entry has full metrics for both YOLOv8 and Faster R-CNN:
      det_conf_pct, det_time_ms, fps, plate, ocr_conf_pct, iou, memory_mb, winner
    """
    return jsonify({
        'yolo_loaded':  yolo_enabled,
        'frcnn_loaded': frcnn_enabled,
        'history':      _comparison_history[-10:],
        'summary': {
            'yolo': {
                'avg_det_time_ms': round(
                    sum(h['yolo']['det_time_ms'] for h in _comparison_history) /
                    max(len(_comparison_history), 1), 1),
                'avg_ocr_conf': round(
                    sum(h['yolo']['ocr_conf_pct'] for h in _comparison_history) /
                    max(len(_comparison_history), 1), 1),
                'win_count': sum(1 for h in _comparison_history if h.get('winner') == 'YOLO'),
            },
            'frcnn': {
                'avg_det_time_ms': round(
                    sum(h['frcnn']['det_time_ms'] for h in _comparison_history) /
                    max(len(_comparison_history), 1), 1),
                'avg_ocr_conf': round(
                    sum(h['frcnn']['ocr_conf_pct'] for h in _comparison_history) /
                    max(len(_comparison_history), 1), 1),
                'win_count': sum(1 for h in _comparison_history if h.get('winner') == 'FRCNN'),
            },
        }
    })


if __name__ == '__main__':
    print()
    print("=" * 62)
    print("   ALPR ENGINE v6  —  REAL WORLD PRODUCTION")
    print("=" * 62)
    print(f"   Vehicles     : {len(db.vehicles)}")
    print(f"   Events       : {len(db.events)}")
    print(f"   Gate         : {SYSTEM_CONFIG['gate_mode'].upper()}")
    print(f"   YOLO         : "
          f"{'✅ ACTIVE — ' + yolo_name if yolo_enabled else 'not loaded'}")
    if yolo_enabled and 'plate' not in yolo_name.lower():
        print(f"   ⚠  Using general YOLO — for best accuracy train:")
        print(f"      python train_yolo11.py")
        print(f"      Save as models/yolo11_plate.pt")
    print(f"   Faster R-CNN : "
          f"{'✅ ACTIVE' if frcnn_enabled else 'not loaded — train with train_fasterrcnn.py'}")
    print(f"   CNN OCR      : "
          f"{'✅ ACTIVE — ' + str(len(_cnn_ocr_chars)) + ' chars' if _cnn_enabled else 'not loaded — run train_cnn_ocr.py --generate'}")
    print(f"   Long distance: super-resolution upscale enabled")
    print(f"   Target time  : 3-5s on CPU")
    print()
    print("   Open: http://localhost:5000")
    print("=" * 62)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)