"""
ALPR Diagnostic Tool
====================
Run this directly on your plate image to see EXACTLY what
each step of the pipeline is doing.

Usage:
    python diagnose.py

It will:
1. Try to read the plate directly with EasyOCR
2. Show what preprocessing does
3. Save debug images so you can see what OCR is working with
4. Print exactly what text is being read at each step
"""

import cv2
import numpy as np
import easyocr
import os
import sys

print("=" * 55)
print("  ALPR DIAGNOSTIC TOOL")
print("=" * 55)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Find test image ────────────────────────────────────────
# Put your plate photo in the same folder as this script
# OR pass the path as argument: python diagnose.py myplate.jpg
test_image_path = None

if len(sys.argv) > 1:
    test_image_path = sys.argv[1]
else:
    # Auto-find any image in current folder
    for ext in ['*.jpg','*.jpeg','*.png','*.bmp']:
        import glob
        files = glob.glob(os.path.join(BASE_DIR, ext))
        if files:
            # Prefer files with "plate" in name
            plate_files = [f for f in files if 'plate' in f.lower() or 'dl' in f.lower().replace('\\','/').split('/')[-1]]
            test_image_path = plate_files[0] if plate_files else files[0]
            break

if not test_image_path or not os.path.exists(test_image_path):
    print("\n  ERROR: No image found!")
    print("  Usage: python diagnose.py your_plate_photo.jpg")
    print("  OR: place a .jpg file in this folder and run: python diagnose.py")
    sys.exit(1)

print(f"\n  Testing image: {test_image_path}")

# ── Load image ─────────────────────────────────────────────
img = cv2.imread(test_image_path)
if img is None:
    print(f"  ERROR: Cannot read {test_image_path}")
    sys.exit(1)

h, w = img.shape[:2]
print(f"  Image size: {w}x{h} pixels")

# ── Save debug folder ──────────────────────────────────────
debug_dir = os.path.join(BASE_DIR, 'debug_output')
os.makedirs(debug_dir, exist_ok=True)
print(f"  Debug images saved to: {debug_dir}/")

# ── Load EasyOCR ───────────────────────────────────────────
print("\n  Loading EasyOCR (may take 30s first time)...")
reader = easyocr.Reader(['en'], gpu=False, verbose=False)
print("  ✅ EasyOCR ready")

def run_ocr(image, label, canvas=2560, mag=2.0):
    """Run OCR and print ALL text found (not just plates)."""
    print(f"\n  ── {label} ──")
    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
        decoder='greedy',
        canvas_size=canvas,
        mag_ratio=mag,
        width_ths=0.85,
        link_threshold=0.3,
    )
    if not results:
        print("    ❌ No text found at all")
        return []
    for bbox, text, conf in results:
        print(f"    ✅ '{text}'  conf={conf*100:.1f}%")
    return results

# ── TEST 1: Full image ─────────────────────────────────────
print("\n" + "─"*55)
print("  TEST 1: Full image OCR (no preprocessing)")
run_ocr(img, "Full image")

# ── TEST 2: Bottom half (where plates usually are) ─────────
print("\n" + "─"*55)
print("  TEST 2: Bottom 60% of image")
bottom = img[int(h*0.40):, :]
cv2.imwrite(os.path.join(debug_dir, '1_bottom_half.jpg'), bottom)
run_ocr(bottom, "Bottom 60%")

# ── TEST 3: Upscale full image 2x ─────────────────────────
print("\n" + "─"*55)
print("  TEST 3: Upscaled 2x")
upscaled = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
cv2.imwrite(os.path.join(debug_dir, '2_upscaled_2x.jpg'), upscaled)
run_ocr(upscaled, "Upscaled 2x", canvas=4096, mag=1.5)

# ── TEST 4: Manual plate crop ─────────────────────────────
# The plate in your image is approximately in the bottom-center
# We'll try multiple crops of the lower portion
print("\n" + "─"*55)
print("  TEST 4: Cropped plate region")

# Try center-bottom crop (where plates typically are)
plate_crops = [
    ("center_bottom", img[int(h*0.60):int(h*0.90), int(w*0.25):int(w*0.75)]),
    ("lower_third",   img[int(h*0.65):, int(w*0.20):int(w*0.80)]),
    ("wide_bottom",   img[int(h*0.55):, :]),
]

for crop_name, crop in plate_crops:
    if crop.size == 0: continue
    crop_h, crop_w = crop.shape[:2]
    print(f"\n  Crop '{crop_name}': {crop_w}x{crop_h}")
    cv2.imwrite(os.path.join(debug_dir, f'3_crop_{crop_name}.jpg'), crop)

    # Upscale crop
    if crop_w < 600:
        scale = 600 / crop_w
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        print(f"    Upscaled to {crop.shape[1]}x{crop.shape[0]}")

    run_ocr(crop, crop_name, canvas=2560, mag=2.0)

# ── TEST 5: Enhanced preprocessing on bottom crop ─────────
print("\n" + "─"*55)
print("  TEST 5: Preprocessed versions of bottom region")

plate_region = img[int(h*0.60):int(h*0.92), int(w*0.20):int(w*0.80)]
if plate_region.size > 0:
    # Upscale first
    scale = 800 / plate_region.shape[1] if plate_region.shape[1] < 800 else 1.0
    if scale > 1:
        plate_region = cv2.resize(plate_region, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_LANCZOS4)

    gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    v1 = clahe.apply(gray)
    blr = cv2.GaussianBlur(v1,(0,0),2.0)
    v1 = np.clip(cv2.addWeighted(v1,2.5,blr,-1.5,0),0,255).astype(np.uint8)
    cv2.imwrite(os.path.join(debug_dir,'4_clahe.jpg'), v1)
    run_ocr(cv2.cvtColor(v1,cv2.COLOR_GRAY2BGR), "CLAHE")

    # Adaptive threshold
    v2 = cv2.adaptiveThreshold(cv2.GaussianBlur(gray,(3,3),0),255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,6)
    if np.mean(v2)<100: v2=cv2.bitwise_not(v2)
    cv2.imwrite(os.path.join(debug_dir,'5_adaptive.jpg'), v2)
    run_ocr(cv2.cvtColor(v2,cv2.COLOR_GRAY2BGR), "Adaptive threshold")

    # Otsu
    _,v3=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if np.mean(v3)<127: v3=cv2.bitwise_not(v3)
    cv2.imwrite(os.path.join(debug_dir,'6_otsu.jpg'), v3)
    run_ocr(cv2.cvtColor(v3,cv2.COLOR_GRAY2BGR), "Otsu threshold")

# ── SUMMARY ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("  DIAGNOSTIC COMPLETE")
print(f"  Check debug images in: {debug_dir}/")
print()
print("  What to look for in debug images:")
print("  - 3_crop_center_bottom.jpg  → should show the plate")
print("  - 4_clahe.jpg               → enhanced plate region")
print("  - 5_adaptive.jpg            → binary plate region")
print()
print("  If the plate is NOT visible in debug images,")
print("  the crop coordinates need adjustment for your image.")
print("=" * 55)
input("\n  Press Enter to exit...")
