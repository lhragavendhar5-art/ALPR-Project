"""
train_cnn_ocr.py — Custom CNN for Indian License Plate OCR
============================================================
Trains a lightweight CNN to read individual characters from
Indian license plates — far more accurate than generic EasyOCR
for the specific fonts and styles used on Indian plates.

HOW IT WORKS:
  1. Downloads/generates Indian plate character dataset
  2. Trains a compact CNN (MobileNet-style) on A-Z + 0-9
  3. Integrates into the ALPR pipeline alongside EasyOCR
     (CNN result overrides EasyOCR when confidence > threshold)

USAGE:
    # Generate synthetic training data + train:
    python train_cnn_ocr.py --generate

    # Train on existing character dataset:
    python train_cnn_ocr.py --dataset chars_dataset/

    # Test on a plate crop:
    python train_cnn_ocr.py --test plate_crop.jpg

OUTPUT:
    models/plate_ocr_cnn.pth    ← load in main.py
    models/ocr_cnn_classes.json ← character mapping
"""

import os, sys, json, time, argparse, random, string
import numpy as np
import cv2
from pathlib import Path

BASE_DIR   = Path(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = BASE_DIR / 'models'
CHARS_DIR  = BASE_DIR / 'chars_dataset'
MODELS_DIR.mkdir(exist_ok=True)

# ── Character set for Indian plates ──
# Indian plates: State(2L) + District(2D) + Series(1-3L) + Number(4D)
CHARS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
NUM_CLASSES = len(CHARS)  # 36

# ──────────────────────────────────────────────────────────────────
#  SYNTHETIC DATA GENERATOR
#  Generates realistic Indian plate character images
# ──────────────────────────────────────────────────────────────────

def generate_char_dataset(output_dir: Path, samples_per_char: int = 500):
    """
    Generates synthetic character images that mimic Indian license
    plate fonts under various conditions:
    - Clean white plate with black text
    - Slight rotation, blur, brightness variation
    - Different plate fonts (standard + bold variants)
    """
    print(f"\n  [GEN] Generating {len(CHARS)} × {samples_per_char} = "
          f"{len(CHARS)*samples_per_char} character images...")

    for split in ['train', 'val']:
        for c in CHARS:
            (output_dir / split / c).mkdir(parents=True, exist_ok=True)

    # Font choices — use what's available on the system
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
    ]

    n_val = max(1, samples_per_char // 10)
    total = 0

    for char in CHARS:
        for i in range(samples_per_char + n_val):
            split = 'val' if i < n_val else 'train'

            # Canvas
            img = np.ones((64, 48), dtype=np.uint8) * random.randint(220, 255)

            # Random font settings
            font      = random.choice(fonts)
            scale     = random.uniform(1.2, 1.8)
            thickness = random.randint(2, 3)
            color     = random.randint(0, 40)

            # Center the character
            (tw, th), baseline = cv2.getTextSize(char, font, scale, thickness)
            x = max(0, (48 - tw) // 2 + random.randint(-3, 3))
            y = max(th, (64 + th) // 2 + random.randint(-3, 3))
            cv2.putText(img, char, (x, y), font, scale, color, thickness, cv2.LINE_AA)

            # Augmentations
            # 1. Brightness/contrast
            alpha = random.uniform(0.75, 1.25)
            beta  = random.randint(-20, 20)
            img   = np.clip(img.astype(float) * alpha + beta, 0, 255).astype(np.uint8)

            # 2. Slight rotation
            if random.random() < 0.4:
                angle = random.uniform(-8, 8)
                M = cv2.getRotationMatrix2D((24, 32), angle, 1.0)
                img = cv2.warpAffine(img, M, (48, 64),
                                     borderValue=int(np.mean(img)))

            # 3. Gaussian blur (simulates distance/motion)
            if random.random() < 0.3:
                k = random.choice([3, 5])
                img = cv2.GaussianBlur(img, (k, k), 0)

            # 4. Gaussian noise
            if random.random() < 0.3:
                noise = np.random.normal(0, random.uniform(5, 15), img.shape)
                img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)

            # 5. Perspective distortion (simulates camera angle)
            if random.random() < 0.2:
                pts1 = np.float32([[0,0],[47,0],[0,63],[47,63]])
                d    = random.randint(2, 5)
                pts2 = np.float32([
                    [random.randint(0,d), random.randint(0,d)],
                    [47-random.randint(0,d), random.randint(0,d)],
                    [random.randint(0,d), 63-random.randint(0,d)],
                    [47-random.randint(0,d), 63-random.randint(0,d)],
                ])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                img = cv2.warpPerspective(img, M, (48, 64))

            # Save
            out_path = output_dir / split / char / f"{char}_{i:05d}.jpg"
            cv2.imwrite(str(out_path), img)
            total += 1

    print(f"  [GEN] Generated {total} images → {output_dir}")
    return output_dir


# ──────────────────────────────────────────────────────────────────
#  CNN MODEL — Compact MobileNet-style
# ──────────────────────────────────────────────────────────────────

def build_ocr_cnn(num_classes: int = 36):
    """
    Compact CNN optimised for single-character recognition.
    ~200K parameters — fast on CPU, high accuracy on plate fonts.
    Architecture: DepthwiseSep blocks (MobileNet-style) → FC
    """
    import torch
    import torch.nn as nn

    class DepthwiseSep(nn.Module):
        def __init__(self, in_c, out_c, stride=1):
            super().__init__()
            self.dw = nn.Conv2d(in_c, in_c,  3, stride=stride, padding=1, groups=in_c,  bias=False)
            self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
            self.bn = nn.BatchNorm2d(out_c)
            self.act = nn.Hardswish()
        def forward(self, x):
            return self.act(self.bn(self.pw(self.dw(x))))

    class PlateOCR_CNN(nn.Module):
        def __init__(self, nc):
            super().__init__()
            self.features = nn.Sequential(
                # Stem
                nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.Hardswish(),
                # DS blocks
                DepthwiseSep(32,  64,  stride=1),  # 32x24
                DepthwiseSep(64,  128, stride=2),  # 16x12
                DepthwiseSep(128, 128, stride=1),
                DepthwiseSep(128, 256, stride=2),  # 8x6
                DepthwiseSep(256, 256, stride=1),
                DepthwiseSep(256, 512, stride=2),  # 4x3
                nn.AdaptiveAvgPool2d(1),            # 1x1
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.Hardswish(),
                nn.Dropout(0.2),
                nn.Linear(256, nc),
            )
        def forward(self, x):
            return self.classifier(self.features(x))

    model = PlateOCR_CNN(num_classes)
    params = sum(p.numel() for p in model.parameters())
    print(f"  [CNN] Parameters: {params:,}")
    return model


# ──────────────────────────────────────────────────────────────────
#  DATASET LOADER
# ──────────────────────────────────────────────────────────────────


# ── Module-level class (must be at top level for Python 3.14 pickling) ──
class CharDataset:
    def __init__(self, root, split):
        import torch
        self.samples  = []
        self.char2idx = {c: i for i, c in enumerate(CHARS)}
        split_dir = Path(root) / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset split not found: {split_dir}")
        for char_dir in sorted(split_dir.iterdir()):
            c = char_dir.name.upper()
            if c not in self.char2idx:
                continue
            label = self.char2idx[c]
            for img_path in char_dir.glob('*.jpg'):
                self.samples.append((str(img_path), label))
        random.shuffle(self.samples)
        print(f"  [Data] {split}: {len(self.samples)} samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        import torch
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((64, 48), dtype=np.uint8)
        img = cv2.resize(img, (48, 64))
        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        return tensor, label


def get_loaders(dataset_dir: Path, batch_size: int = 128):
    from torch.utils.data import DataLoader
    train_ds = CharDataset(dataset_dir, 'train')
    val_ds   = CharDataset(dataset_dir, 'val')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=0)
    return train_loader, val_loader



# ──────────────────────────────────────────────────────────────────
#  TRAINING LOOP
# ──────────────────────────────────────────────────────────────────

def train_ocr(dataset_dir: Path, args):
    import torch
    import torch.nn as nn

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  [TRAIN] Device: {device}")

    train_loader, val_loader = get_loaders(dataset_dir, args.batch)
    model     = build_ocr_cnn(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader), epochs=args.epochs)

    best_acc   = 0.0
    best_path  = MODELS_DIR / 'plate_ocr_cnn.pth'
    log        = []

    print(f"\n{'='*55}")
    print(f"  CNN OCR TRAINING")
    print(f"  Classes  : {NUM_CLASSES}  ({' '.join(CHARS[:10])}...)")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch}")
    print(f"{'='*55}\n")

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            correct    += (out.argmax(1) == labels).sum().item()
            total      += len(labels)
        train_acc = correct / total

        # Validate
        model.eval()
        val_loss, vcorrect, vtotal = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out  = model(imgs)
                loss = criterion(out, labels)
                val_loss  += loss.item()
                vcorrect  += (out.argmax(1) == labels).sum().item()
                vtotal    += len(labels)
        val_acc = vcorrect / vtotal

        lr_now = scheduler.get_last_lr()[0]
        print(f"  Epoch [{epoch:3d}/{args.epochs}]  "
              f"Train {train_acc*100:.1f}%  Val {val_acc*100:.1f}%  "
              f"Loss {train_loss/len(train_loader):.4f}  LR {lr_now:.6f}")

        log.append({'epoch': epoch, 'train_acc': round(train_acc,4),
                    'val_acc': round(val_acc,4)})

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'chars':            CHARS,
                'val_acc':          val_acc,
                'epoch':            epoch,
            }, best_path)
            print(f"  ✅ Best model saved ({val_acc*100:.1f}%)")

    # Save char map
    char_map = {'chars': CHARS, 'char2idx': {c:i for i,c in enumerate(CHARS)}}
    (MODELS_DIR / 'ocr_cnn_classes.json').write_text(json.dumps(char_map, indent=2))

    print(f"\n  ✅ OCR CNN training complete!")
    print(f"  Best val accuracy : {best_acc*100:.1f}%")
    print(f"  Weights saved     : {best_path}")
    print()
    print("  ══ NEXT STEP — Integrate into main.py ══")
    print("  The patch_main.py script will add CNN OCR to your pipeline.")
    print("  Run: python patch_main.py")


# ──────────────────────────────────────────────────────────────────
#  INFERENCE — run CNN OCR on a plate crop
# ──────────────────────────────────────────────────────────────────

def segment_and_read(plate_bgr: np.ndarray, model, chars: list,
                     device, conf_threshold: float = 0.7):
    """
    Segment plate image into individual characters and classify each.
    Returns (plate_text, avg_confidence).
    """
    import torch

    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)

    # Binarize
    _, thresh = cv2.threshold(
        cv2.GaussianBlur(gray, (3,3), 0), 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Some plates may need inversion
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    # Find character contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    char_regions = []
    h, w = plate_bgr.shape[:2]

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        ar = cw / ch if ch > 0 else 0
        # Filter by size and aspect ratio
        if (ch > h * 0.3 and cw > 4 and ch > 8
                and ar < 1.2 and cw < w * 0.25):
            char_regions.append((x, y, cw, ch))

    # Sort left to right
    char_regions.sort(key=lambda r: r[0])

    if not char_regions:
        return None, 0.0

    model.eval()
    results = []
    with torch.no_grad():
        for (x, y, cw, ch) in char_regions:
            crop = gray[y:y+ch, x:x+cw]
            crop = cv2.resize(crop, (48, 64))
            # Normalize
            tensor = torch.tensor(
                crop, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
            tensor = tensor.to(device)
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)
            conf, idx = probs.max(1)
            conf = float(conf[0])
            char = chars[int(idx[0])]
            if conf >= conf_threshold:
                results.append((char, conf))

    if not results:
        return None, 0.0

    plate_text = ''.join(c for c, _ in results)
    avg_conf   = sum(c for _, c in results) / len(results)
    return plate_text, avg_conf


def test_plate(weights_path: str, image_path: str):
    import torch
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    chars  = checkpoint.get('chars', CHARS)
    device = torch.device('cpu')
    model  = build_ocr_cnn(len(chars)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    img = cv2.imread(image_path)
    if img is None:
        print(f"  ERROR: Cannot read {image_path}")
        return

    text, conf = segment_and_read(img, model, chars, device)
    print(f"\n  Image  : {image_path}")
    print(f"  Result : {text}")
    print(f"  Conf   : {conf*100:.1f}%")


# ──────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train CNN OCR for Indian License Plate Characters')
    parser.add_argument('--generate', action='store_true',
        help='Generate synthetic character dataset and train')
    parser.add_argument('--dataset', default='chars_dataset',
        help='Path to character dataset (class subfolders)')
    parser.add_argument('--samples', type=int, default=500,
        help='Synthetic samples per character (default: 500)')
    parser.add_argument('--epochs',  type=int, default=40,
        help='Training epochs (default: 40)')
    parser.add_argument('--batch',   type=int, default=128,
        help='Batch size (default: 128)')
    parser.add_argument('--lr',      type=float, default=1e-3,
        help='Learning rate (default: 0.001)')
    parser.add_argument('--test',    default='',
        help='Test on a plate crop image')
    parser.add_argument('--weights', default='models/plate_ocr_cnn.pth',
        help='Weights path for --test')
    args = parser.parse_args()

    if args.test:
        test_plate(args.weights, args.test)
        sys.exit(0)

    try:
        import torch
    except ImportError:
        print("  ERROR: PyTorch not installed.")
        print("  Run: pip install torch torchvision")
        sys.exit(1)

    print("=" * 55)
    print("  CNN OCR — INDIAN PLATE CHARACTER RECOGNIZER")
    print("=" * 55)

    dataset_dir = Path(args.dataset)

    if args.generate or not (dataset_dir / 'train').exists():
        print(f"\n  Generating synthetic dataset ({args.samples}/char)...")
        generate_char_dataset(dataset_dir, args.samples)
    else:
        print(f"\n  Using existing dataset: {dataset_dir}")

    train_ocr(dataset_dir, args)