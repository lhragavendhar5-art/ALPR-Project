"""
train_yolo11.py — YOLOv11 License Plate Detector for Indian Plates
====================================================================
Downloads the Indian License Plate dataset and trains YOLOv11n.

STEP 1 — Install dependencies:
    pip install ultralytics roboflow opencv-python tqdm

STEP 2 — Get a FREE Roboflow API key:
    Go to: https://roboflow.com  →  Sign up (free)
    Create project or use existing → Settings → API Keys → Copy key

STEP 3 — Run training:
    python train_yolo11.py --api-key YOUR_ROBOFLOW_KEY

    OR use Kaggle dataset (no key needed):
    python train_yolo11.py --source kaggle

Output:
    models/yolo11_plate/weights/best.pt   ← use this in main.py
    models/yolo11_plate/weights/last.pt
"""

import os, sys, shutil, argparse, zipfile, json
from pathlib import Path

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = BASE_DIR / 'models'
DATASET_DIR = BASE_DIR / 'dataset_yolo11'
MODELS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────
#  DATASET SOURCES
# ──────────────────────────────────────────────────────────────────

ROBOFLOW_DATASETS = [
    # Best Indian plate datasets on Roboflow (public, free)
    {
        'workspace': 'license-plate-recognition-rxdml',
        'project':   'indian-license-plate-zqrif',
        'version':   3,
        'name':      'Indian License Plate (3.4k images)',
    },
    {
        'workspace': 'augmented-startups',
        'project':   'vehicle-registration-plates-trudk',
        'version':   1,
        'name':      'Vehicle Registration Plates (India+)',
    },
    {
        'workspace': 'indian-number-plate',
        'project':   'number-plate-detection-xbsrq',
        'version':   3,
        'name':      'Indian Number Plate Detection',
    },
]

KAGGLE_DATASET = {
    'owner':   'andrewmvd',
    'dataset': 'car-plate-detection',
    'url':     'https://www.kaggle.com/datasets/andrewmvd/car-plate-detection',
}


# ──────────────────────────────────────────────────────────────────
#  DOWNLOAD — ROBOFLOW
# ──────────────────────────────────────────────────────────────────

def download_roboflow(api_key: str, choice: int = 0) -> Path:
    try:
        from roboflow import Roboflow
    except ImportError:
        print("  [ERROR] roboflow not installed. Run:")
        print("          pip install roboflow")
        sys.exit(1)

    ds_info = ROBOFLOW_DATASETS[choice]
    print(f"\n  [DATASET] Downloading: {ds_info['name']}")
    print(f"  Source : roboflow.com/{ds_info['workspace']}/{ds_info['project']}")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ds_info['workspace']).project(ds_info['project'])
    dataset = project.version(ds_info['version']).download(
        'yolov8',  # YOLOv8 format = compatible with YOLOv11
        location=str(DATASET_DIR),
        overwrite=True
    )
    print(f"  [DATASET] Downloaded to: {DATASET_DIR}")
    return DATASET_DIR


# ──────────────────────────────────────────────────────────────────
#  DOWNLOAD — KAGGLE
# ──────────────────────────────────────────────────────────────────

def download_kaggle() -> Path:
    """
    Downloads andrewmvd/car-plate-detection from Kaggle.
    Requires: pip install kaggle  AND  ~/.kaggle/kaggle.json
    Get key: kaggle.com → Account → API → Create New API Token
    """
    try:
        import kaggle
    except ImportError:
        print("  [ERROR] kaggle not installed. Run: pip install kaggle")
        print("  Then get API key from: https://www.kaggle.com/settings")
        sys.exit(1)

    out = DATASET_DIR / 'kaggle_raw'
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n  [DATASET] Downloading from Kaggle: {KAGGLE_DATASET['dataset']}")
    os.system(f'kaggle datasets download -d {KAGGLE_DATASET["owner"]}/{KAGGLE_DATASET["dataset"]} -p {out} --unzip')

    # Convert XML annotations → YOLO format
    converted = _convert_kaggle_to_yolo(out, DATASET_DIR)
    return DATASET_DIR


def _convert_kaggle_to_yolo(src: Path, dst: Path) -> Path:
    """Convert Pascal VOC XML annotations from kaggle dataset to YOLO format."""
    import xml.etree.ElementTree as ET
    import cv2

    print("  [CONVERT] Pascal VOC → YOLO format...")
    for split in ['train', 'val']:
        (dst / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dst / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Find all images and annotations
    img_files = list(src.rglob('*.png')) + list(src.rglob('*.jpg'))
    ann_files = list(src.rglob('*.xml'))
    ann_map   = {a.stem: a for a in ann_files}

    # 90/10 split
    from random import shuffle
    shuffle(img_files)
    n_val = max(1, int(len(img_files) * 0.1))
    splits = {'val': img_files[:n_val], 'train': img_files[n_val:]}

    converted = 0
    for split, files in splits.items():
        for img_path in files:
            ann_path = ann_map.get(img_path.stem)
            if not ann_path:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            try:
                tree = ET.parse(ann_path)
                root = tree.getroot()
            except Exception:
                continue

            lines = []
            for obj in root.findall('object'):
                bnd = obj.find('bndbox')
                if bnd is None:
                    continue
                x1 = float(bnd.find('xmin').text)
                y1 = float(bnd.find('ymin').text)
                x2 = float(bnd.find('xmax').text)
                y2 = float(bnd.find('ymax').text)
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if not lines:
                continue

            shutil.copy(img_path, dst / 'images' / split / img_path.name)
            lbl_path = dst / 'labels' / split / (img_path.stem + '.txt')
            lbl_path.write_text('\n'.join(lines))
            converted += 1

    print(f"  [CONVERT] {converted} images converted")

    # Write data.yaml
    yaml_content = f"""path: {dst}
train: images/train
val: images/val
nc: 1
names: ['license_plate']
"""
    (dst / 'data.yaml').write_text(yaml_content)
    return dst


# ──────────────────────────────────────────────────────────────────
#  MANUAL DATASET PREP (if you already have images)
# ──────────────────────────────────────────────────────────────────

def prepare_existing_dataset(src_dir: str) -> Path:
    """
    If you already have a YOLO-format dataset, just point to it.
    This validates and creates the data.yaml if missing.
    """
    src = Path(src_dir)
    required = [
        src / 'images' / 'train',
        src / 'images' / 'val',
        src / 'labels' / 'train',
        src / 'labels' / 'val',
    ]
    for r in required:
        if not r.exists():
            print(f"  [ERROR] Missing: {r}")
            print("  Dataset must have: images/train, images/val, labels/train, labels/val")
            sys.exit(1)

    yaml_path = src / 'data.yaml'
    if not yaml_path.exists():
        yaml_content = f"""path: {src.absolute()}
train: images/train
val: images/val
nc: 1
names: ['license_plate']
"""
        yaml_path.write_text(yaml_content)
        print(f"  [PREP] Created data.yaml at {yaml_path}")

    # Count images
    n_train = len(list((src/'images'/'train').glob('*')))
    n_val   = len(list((src/'images'/'val').glob('*')))
    print(f"  [PREP] Train: {n_train} images | Val: {n_val} images")
    return src


# ──────────────────────────────────────────────────────────────────
#  TRAIN YOLOv11
# ──────────────────────────────────────────────────────────────────

def train_yolo11(dataset_path: Path, args):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  [ERROR] ultralytics not installed. Run:")
        print("          pip install ultralytics")
        sys.exit(1)

    yaml_path = dataset_path / 'data.yaml'
    if not yaml_path.exists():
        # Try to find data.yaml anywhere in dataset
        yamls = list(dataset_path.rglob('data.yaml'))
        if yamls:
            yaml_path = yamls[0]
        else:
            print(f"  [ERROR] data.yaml not found in {dataset_path}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("  YOLOv11 LICENSE PLATE TRAINING")
    print(f"{'='*60}")
    print(f"  Model   : yolo11{args.model_size}.pt")
    print(f"  Dataset : {yaml_path}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  ImgSz   : {args.imgsz}")
    print(f"{'='*60}\n")

    # Load YOLOv11 (auto-downloads weights on first run)
    model = YOLO(f'yolo11{args.model_size}.pt')

    # Train
    results = model.train(
        data      = str(yaml_path),
        epochs    = args.epochs,
        batch     = args.batch,
        imgsz     = args.imgsz,
        device    = args.device,
        project   = str(MODELS_DIR),
        name      = 'yolo11_plate',
        patience  = 20,         # early stop after 20 epochs no improvement
        save      = True,
        plots     = True,
        exist_ok  = True,

        # Augmentation — tuned for license plates
        hsv_h     = 0.01,       # tiny hue shift (plates have fixed colors)
        hsv_s     = 0.3,
        hsv_v     = 0.4,
        degrees   = 3.0,        # slight rotation
        translate = 0.1,
        scale     = 0.3,
        shear     = 2.0,
        perspective = 0.0003,
        flipud    = 0.0,        # no vertical flip (plates are never upside down)
        fliplr    = 0.3,        # horizontal flip OK
        mosaic    = 0.8,
        mixup     = 0.05,

        # Optimizer
        optimizer = 'AdamW',
        lr0       = 0.001,
        lrf       = 0.01,
        warmup_epochs = 3,

        # Confidence & NMS
        conf      = 0.25,
        iou       = 0.45,
    )

    best_pt = MODELS_DIR / 'yolo11_plate' / 'weights' / 'best.pt'
    if best_pt.exists():
        # Copy to top-level for easy use
        dest = MODELS_DIR / 'yolo11_plate.pt'
        shutil.copy(best_pt, dest)
        print(f"\n  ✅ Training complete!")
        print(f"  Best weights : {best_pt}")
        print(f"  Copied to   : {dest}")
        print()
        print("  ══ NEXT STEP — Update main.py ══")
        print(f"  In main.py, _YOLO_PATHS, add this as FIRST entry:")
        print(f"  '{dest}'")
        print()
        print("  OR rename/replace your existing yolov8n.pt:")
        print(f"  copy models\\yolo11_plate.pt yolov8n.pt")
        _print_map(results)
    else:
        print("  [WARN] best.pt not found — check training output above")
        print(f"  Training outputs: {MODELS_DIR / 'yolo11_plate'}")


def _print_map(results):
    try:
        # Print final mAP from results
        metrics = results.results_dict
        print(f"\n  ── Final Metrics ──────────────────")
        print(f"  mAP50    : {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP50-95 : {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall   : {metrics.get('metrics/recall(B)', 0):.4f}")
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────
#  VALIDATE after training
# ──────────────────────────────────────────────────────────────────

def validate(weights_path: str, yaml_path: str = None):
    from ultralytics import YOLO
    model = YOLO(weights_path)
    if yaml_path:
        results = model.val(data=yaml_path)
    else:
        results = model.val()
    print(f"\n  Validation mAP50: {results.box.map50:.4f}")
    print(f"  Validation mAP50-95: {results.box.map:.4f}")


# ──────────────────────────────────────────────────────────────────
#  QUICK INFERENCE TEST
# ──────────────────────────────────────────────────────────────────

def test_image(weights_path: str, image_path: str):
    from ultralytics import YOLO
    import cv2

    model = YOLO(weights_path)
    results = model(image_path, conf=0.25, verbose=False)
    img = cv2.imread(image_path)
    print(f"\n  Testing: {image_path}")
    for r in results:
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            print(f"  Detection {i+1}: conf={conf:.3f}  box=({x1},{y1},{x2},{y2})")
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, f"{conf:.2f}", (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    out_path = str(Path(image_path).with_suffix('')) + '_yolo11_test.jpg'
    cv2.imwrite(out_path, img)
    print(f"  Saved annotated: {out_path}")


# ──────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train YOLOv11 for Indian License Plate Detection',
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Dataset source
    parser.add_argument('--source', default='roboflow',
        choices=['roboflow', 'kaggle', 'local'],
        help='Dataset source:\n'
             '  roboflow  — Download Indian plate dataset (needs --api-key)\n'
             '  kaggle    — Download from Kaggle (needs kaggle.json)\n'
             '  local     — Use existing YOLO dataset (needs --dataset)')
    parser.add_argument('--api-key', default='',
        help='Roboflow API key (get free at roboflow.com)')
    parser.add_argument('--rf-dataset', type=int, default=0,
        choices=[0, 1, 2],
        help='Which Roboflow dataset to use:\n'
             '  0 = Indian License Plate (3.4k images) [default]\n'
             '  1 = Vehicle Registration Plates\n'
             '  2 = Number Plate Detection')
    parser.add_argument('--dataset', default='dataset',
        help='Path to existing YOLO dataset (used with --source local)')

    # Training hyperparams
    parser.add_argument('--model-size', default='n', choices=['n','s','m','l','x'],
        help='YOLOv11 model size: n=nano, s=small, m=medium (default: n)')
    parser.add_argument('--epochs',  type=int,   default=50,
        help='Training epochs (default: 50)')
    parser.add_argument('--batch',   type=int,   default=16,
        help='Batch size (default: 16, reduce to 8 if OOM)')
    parser.add_argument('--imgsz',   type=int,   default=640,
        help='Image size (default: 640)')
    parser.add_argument('--device',  default='',
        help='Device: "" = auto, "0" = GPU0, "cpu" = CPU')

    # Other
    parser.add_argument('--validate', default='',
        help='Validate weights (path to .pt file)')
    parser.add_argument('--test',     default='',
        help='Test on single image (path to image)')
    parser.add_argument('--weights',  default='models/yolo11_plate.pt',
        help='Weights for --test or --validate')
    parser.add_argument('--skip-download', action='store_true',
        help='Skip download, use existing dataset_yolo11/ folder')

    args = parser.parse_args()

    # ── Modes ──
    if args.validate:
        validate(args.validate)
        sys.exit(0)

    if args.test:
        test_image(args.weights, args.test)
        sys.exit(0)

    print("=" * 60)
    print("  YOLOv11 INDIAN LICENSE PLATE TRAINER")
    print("=" * 60)

    # ── Download / prep dataset ──
    if args.skip_download and DATASET_DIR.exists():
        print(f"  [SKIP] Using existing dataset: {DATASET_DIR}")
        dataset_path = DATASET_DIR
        # Make sure data.yaml exists
        if not (dataset_path / 'data.yaml').exists():
            dataset_path = prepare_existing_dataset(str(dataset_path))
    elif args.source == 'roboflow':
        if not args.api_key:
            print("\n  ❌  Roboflow API key required!")
            print("  1. Go to: https://roboflow.com (free signup)")
            print("  2. Settings → API Keys → Copy")
            print("  3. Run: python train_yolo11.py --api-key YOUR_KEY\n")
            print("  OR use Kaggle (no key needed):")
            print("  python train_yolo11.py --source kaggle\n")
            print("  OR use your own dataset:")
            print("  python train_yolo11.py --source local --dataset /path/to/dataset\n")
            print("  Available Roboflow datasets:")
            for i, ds in enumerate(ROBOFLOW_DATASETS):
                print(f"    [{i}] {ds['name']}")
            sys.exit(0)
        dataset_path = download_roboflow(args.api_key, args.rf_dataset)

    elif args.source == 'kaggle':
        dataset_path = download_kaggle()

    elif args.source == 'local':
        dataset_path = prepare_existing_dataset(args.dataset)

    # ── Train ──
    train_yolo11(dataset_path, args)
