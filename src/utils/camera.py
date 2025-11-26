# Utility untuk menangani webcam
import cv2


class Camera:
    # Kelas untuk menangani akses webcam
    
    def __init__(self, camera_id=0, width=1280, height=720):
        # Inisialisasi camera
        # Args:
        #   camera_id: ID camera (default: 0)
        #   width: Lebar frame (default: 1280)
        #   height: Tinggi frame (default: 720)
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        
    def open(self):
        # Buka koneksi ke webcam
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            return True
        return False
    
    def read(self):
        # Baca frame dari webcam
        if not self.cap or not self.cap.isOpened():
            return False, None
        return self.cap.read()
    
    def release(self):
        # Tutup koneksi webcam
        if self.cap:
            self.cap.release()
    
    def is_opened(self):
        # Cek apakah camera terbuka
        return self.cap is not None and self.cap.isOpened()

