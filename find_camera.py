"""
CAMERA FINDER — Run this to find your DroidCam camera ID
Usage: python find_camera.py
"""
import cv2

print()
print("=" * 50)
print("  CAMERA FINDER")
print("=" * 50)
print("  Scanning camera IDs 0 to 5...")
print()

found = []
for i in range(6):
    # Try DirectShow first (Windows)
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"  ✅  Camera ID {i} — FOUND  ({w}x{h})")
            found.append(i)
        else:
            print(f"  ⚠️   Camera ID {i} — opens but no frame")
        cap.release()
    else:
        cap2 = cv2.VideoCapture(i)
        if cap2.isOpened():
            ret, frame = cap2.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"  ✅  Camera ID {i} — FOUND  ({w}x{h})")
                found.append(i)
            cap2.release()
        else:
            print(f"  ✗   Camera ID {i} — not available")

print()
print("=" * 50)
if found:
    print(f"  Found {len(found)} camera(s): {found}")
    if len(found) == 1:
        print(f"  → Use Camera ID: {found[0]}")
    else:
        print(f"  → Default webcam is probably: {found[0]}")
        print(f"  → DroidCam is probably: {found[1]}")
        print(f"  → Try each ID in the dashboard Webcam tab")
else:
    print("  No cameras found!")
    print("  Make sure DroidCam PC client is running")
    print("  and DroidCam app is open on your phone")
print("=" * 50)
print()
input("  Press Enter to exit...")
