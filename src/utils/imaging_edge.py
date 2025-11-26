# Utility untuk mengontrol Sony Imaging Edge Remote
import time
import pyautogui
import pygetwindow as gw


class ImagingEdgeController:
    # Kelas untuk mengontrol Sony Imaging Edge Remote
    
    def __init__(self):
        # Inisialisasi controller
        self.camera_window = None
        
    def find_window(self):
        # Cari window Sony Imaging Edge Remote
        windows = gw.getAllWindows()
        for window in windows:
            if "imaging edge" in window.title.lower() or "remote" in window.title.lower():
                self.camera_window = window
                print(f"[INFO] Imaging Edge found: {window.title}")
                return True
        print("[WARNING] Imaging Edge not found")
        self.camera_window = None
        return False
    
    def focus_window(self):
        # Fokus ke window Imaging Edge
        if self.camera_window and not self.camera_window.isMinimized:
            try:
                self.camera_window.activate()
                print("[INFO] Imaging Edge focused")
                return True
            except Exception as e:
                print(f"[ERROR] Could not focus Imaging Edge: {e}")
                return False
        return False
    
    def trigger_shutter(self, x_offset=250, y_offset=250):
        # Trigger shutter via mouse click pada tombol shutter Imaging Edge
        # Args:
        #   x_offset: Offset dari kanan (default: 250)
        #   y_offset: Offset dari atas (default: 250)
        # Returns: bool: True jika berhasil, False jika gagal
        if not self.camera_window:
            print("[ERROR] Imaging Edge not available")
            return False

        try:
            rect = (self.camera_window.left, self.camera_window.top,
                    self.camera_window.width, self.camera_window.height)

            # Hitung posisi tombol shutter
            x = rect[0] + rect[2] - x_offset  # dari kanan
            y = rect[1] + y_offset  # dari atas

            self.focus_window()
            time.sleep(0.3)
            pyautogui.click(x, y)
            print(f"[ACTION] Mouse click at ({x},{y}) for shutter")
            return True
        except Exception as e:
            print(f"[ERROR] Shutter click failed: {e}")
            return False

