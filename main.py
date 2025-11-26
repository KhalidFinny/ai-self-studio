import os
import sys

# Tambahkan src ke path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.efficientnet_detector import ResNet50GestureDetector
from src.utils.camera import Camera
from src.utils.imaging_edge import ImagingEdgeController
from src.gui.web_app import WebApp

def main():
    # Setup camera
    camera = Camera(camera_id=0, width=1280, height=720)
    if not camera.open():
        print("[FAIL] Camera not accessible.")
        return
    
    # Setup detector dengan ResNet50
    model_path = "models/resnet50/best_model.h5"
    try:
        detector = ResNet50GestureDetector(model_path=model_path, confidence=0.85)
        print("[SUCCESS] ResNet50 model loaded.")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # Setup Imaging Edge controller
    imaging_edge = ImagingEdgeController()
    
    # Setup dan jalankan aplikasi Web
    app = WebApp(
        camera=camera,
        detector=detector,
        imaging_edge=imaging_edge,
        countdown_seconds=3
    )
    app.run()


if __name__ == "__main__":
    main()

