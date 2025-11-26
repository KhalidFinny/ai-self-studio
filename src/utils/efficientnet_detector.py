# Utility untuk deteksi gesture menggunakan ResNet50
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import mediapipe as mp


class ResNet50GestureDetector:
    # Kelas untuk mendeteksi gesture (palm/fist) menggunakan ResNet50
    
    def __init__(self, model_path="models/resnet50/best_model.h5", confidence=0.85):
        # Inisialisasi detector
        # Args:
        #   model_path: Path ke model ResNet50 (.h5)
        #   confidence: Threshold confidence untuk deteksi (default: 0.85)
        
        # Resolve model path (relatif dari root project)
        if not os.path.isabs(model_path):
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            model_path = os.path.join(root_dir, model_path)
        
        # Cek apakah model path ada
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please ensure the model file exists."
            )
        
        # Load model
        self.model = keras.models.load_model(model_path)
        self.confidence = confidence
        self.fist_detected_frames = 0
        self.trigger_active = False
        
        # Setup MediaPipe untuk deteksi tangan
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Class names (sesuai training: 0=palm, 1=fist)
        self.class_names = ['palm', 'fist']
    
    def preprocess_frame(self, frame, bbox=None):
        # Preprocess frame untuk ResNet50
        # Args:
        #   frame: Frame gambar BGR
        #   bbox: Bounding box (x, y, w, h) untuk crop, jika None gunakan seluruh frame
        # Returns: Preprocessed image (224x224, RGB, normalized)
        
        if bbox:
            x, y, w, h = bbox
            # Crop hand region dengan padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            cropped = frame[y:y+h, x:x+w]
        else:
            cropped = frame
        
        # Resize ke 224x224
        resized = cv2.resize(cropped, (224, 224))
        
        # Convert BGR ke RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize ke [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Expand dimensions untuk batch
        expanded = np.expand_dims(normalized, axis=0)
        
        return expanded
    
    def detect_hand(self, frame):
        # Deteksi tangan menggunakan MediaPipe
        # Returns: (bbox, landmarks) atau (None, None) jika tidak terdeteksi
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get bounding box dari landmarks
            h, w = frame.shape[:2]
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            
            x_min = int(min(x_coords))
            x_max = int(max(x_coords))
            y_min = int(min(y_coords))
            y_max = int(max(y_coords))
            
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            
            return bbox, hand_landmarks
        
        return None, None
    
    def classify_gesture(self, preprocessed_img):
        # Klasifikasi gesture menggunakan ResNet50
        # Args:
        #   preprocessed_img: Image yang sudah di-preprocess
        # Returns: (predicted_class, confidence, probabilities)
        
        predictions = self.model.predict(preprocessed_img, verbose=0)
        
        # Binary classification: predictions[0][0] adalah probabilitas fist
        fist_prob = float(predictions[0][0])
        palm_prob = 1.0 - fist_prob
        
        # Tentukan class berdasarkan threshold
        if fist_prob >= 0.5:
            predicted_class = 1  # fist
            confidence = fist_prob
        else:
            predicted_class = 0  # palm
            confidence = palm_prob
        
        return predicted_class, confidence, (palm_prob, fist_prob)
    
    def detect(self, frame, min_frames=5):
        # Deteksi gesture pada frame
        # Args:
        #   frame: Frame gambar dari camera
        #   min_frames: Jumlah frame minimum untuk trigger (default: 5)
        # Returns: tuple: (frame yang sudah di-annotate, apakah fist terdeteksi, should_trigger)
        
        # Deteksi tangan
        bbox, landmarks = self.detect_hand(frame)
        
        detected_fist = False
        confidence = 0.0
        label = "no hand"
        
        if bbox:
            # Preprocess dan classify
            preprocessed = self.preprocess_frame(frame, bbox)
            predicted_class, conf, probs = self.classify_gesture(preprocessed)
            
            confidence = conf
            label = self.class_names[predicted_class]
            
            # Cek apakah fist terdeteksi
            if predicted_class == 1 and confidence >= self.confidence:
                detected_fist = True
            
            # Draw bounding box dan label
            x, y, w, h = bbox
            color = (0, 255, 0) if detected_fist and self.fist_detected_frames >= min_frames else (0, 255, 255)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw landmarks
            if landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
            
            # Draw label
            text = f"{label} {confidence:.2f}"
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            # Tidak ada tangan terdeteksi
            cv2.putText(frame, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
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
    
    def __del__(self):
        # Cleanup MediaPipe
        if hasattr(self, 'hands'):
            self.hands.close()

