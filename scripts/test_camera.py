# Script untuk test koneksi webcam
import cv2
import sys
import os

# Tambahkan parent directory ke path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[FAIL] Camera not accessible.")
else:
    print("[SUCCESS] Webcam connected.")
    cap.release()

