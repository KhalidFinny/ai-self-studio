# Utility untuk deteksi gesture menggunakan YOLO
import cv2
from ultralytics import YOLO
import os


class GestureDetector:
    # Kelas untuk mendeteksi gesture (fist) menggunakan YOLO
    
    def __init__(self, model_path="runs/detect/train/weights/best.pt", confidence=0.85):
        # Inisialisasi detector
        # Args:
        #   model_path: Path ke model YOLO
        #   confidence: Threshold confidence untuk deteksi (default: 0.85)
        # Cek apakah model path ada, jika tidak coba path alternatif
        if not os.path.exists(model_path):
            # Coba path relatif dari root project
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            alt_path = os.path.join(root_dir, model_path)
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Model not found at {model_path} or {alt_path}. "
                    "Please ensure the model file exists."
                )
        
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.fist_detected_frames = 0
        self.trigger_active = False
        
    def detect(self, frame, min_frames=5):
        # Deteksi gesture pada frame
        # Args:
        #   frame: Frame gambar dari camera
        #   min_frames: Jumlah frame minimum untuk trigger (default: 5)
        # Returns: tuple: (frame yang sudah di-annotate, apakah fist terdeteksi)
        results = self.model.predict(frame, imgsz=640, conf=self.confidence, verbose=False)
        
        detected_fist = False
        
        for r in results:
            for box in r.boxes:
                # Extract box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0])
                
                if cls >= len(self.model.names):  # safety guard
                    continue
                    
                label = self.model.names[cls]
                conf = float(box.conf[0])
                
                # Pilih warna box berdasarkan jumlah frame yang terdeteksi
                color = (0, 255, 0) if self.fist_detected_frames >= min_frames else (0, 255, 255)
                
                # Gambar rectangle + label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Cek apakah fist terdeteksi
                if label.lower() == "fist":
                    detected_fist = True
        
        # Hitung frame berturut-turut dengan fist
        if detected_fist:
            self.fist_detected_frames += 1
        else:
            self.fist_detected_frames = 0
        
        # Trigger hanya jika fist stabil untuk beberapa frame
        should_trigger = self.fist_detected_frames >= min_frames and not self.trigger_active
        
        return frame, detected_fist, should_trigger
    
    def reset_trigger(self):
        # Reset status trigger
        self.trigger_active = False
        self.fist_detected_frames = 0
    
    def set_trigger_active(self, value):
        # Set status trigger
        self.trigger_active = value

