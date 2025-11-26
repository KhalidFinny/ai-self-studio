# GUI untuk aplikasi Fist Trigger Photo Booth
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import threading
import time


class PhotoBoothApp:
    # Kelas untuk aplikasi Photo Booth dengan GUI
    
    def __init__(self, camera, detector, imaging_edge, countdown_seconds=3):
        # Inisialisasi aplikasi
        # Args:
        #   camera: Instance Camera
        #   detector: Instance GestureDetector
        #   imaging_edge: Instance ImagingEdgeController
        #   countdown_seconds: Durasi countdown dalam detik (default: 3)
        self.camera = camera
        self.detector = detector
        self.imaging_edge = imaging_edge
        self.countdown_seconds = countdown_seconds
        
        # Setup GUI
        self.root = tk.Tk()
        self.root.title("Fist Trigger Photo Booth")
        self.root.configure(bg="black")
        self.root.attributes('-fullscreen', True)
        
        # Video feed display
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.grid(row=0, column=0, padx=20, pady=20)
        
        # Countdown label
        self.countdown_label = tk.Label(
            self.root,
            text="",
            font=("Helvetica", 96, "bold"),
            fg="red",
            bg="black"
        )
        self.countdown_label.grid(row=1, column=0, pady=30)
        
        # Instructions label
        self.instructions_label = tk.Label(
            self.root,
            text="✊ Show your fist to trigger a photo (hold steady for 5 frames)",
            font=("Helvetica", 16),
            fg="yellow",
            bg="black"
        )
        self.instructions_label.grid(row=2, column=0, pady=20)
        
        # Exit fullscreen with Esc
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        
    def start_countdown(self):
        # Mulai countdown dan trigger shutter
        self.detector.set_trigger_active(True)
        
        # Countdown
        for i in range(self.countdown_seconds, 0, -1):
            self.countdown_label.config(text=str(i))
            time.sleep(1)
        
        self.countdown_label.config(text="Shoot!")
        
        # Cari dan trigger Imaging Edge
        if not self.imaging_edge.find_window():
            self.countdown_label.config(text="Imaging Edge Not Found", fg="red")
            time.sleep(2)
            self.countdown_label.config(text="")
            self.detector.set_trigger_active(False)
            return
        
        if self.imaging_edge.trigger_shutter():
            print("[SUCCESS] Shutter executed")
        else:
            self.countdown_label.config(text="Shutter Failed!", fg="red")
            print("[FATAL] Could not trigger shutter")
        
        time.sleep(1)
        self.countdown_label.config(text="")
        self.detector.set_trigger_active(False)
    
    def update_frame(self):
        # Update frame dari camera
        if not self.camera.is_opened():
            self.countdown_label.config(text="Webcam Error", fg="red")
            return
        
        ret, frame = self.camera.read()
        if not ret:
            self.root.after(500, self.update_frame)
            return
        
        # Flip frame secara horizontal (mirror effect)
        frame = cv2.flip(frame, 1)
        
        # Deteksi gesture
        frame, detected_fist, should_trigger = self.detector.detect(frame)
        
        # Trigger countdown jika perlu
        if should_trigger:
            print("[INFO] Fist detected consistently, triggering countdown")
            threading.Thread(target=self.start_countdown, daemon=True).start()
        
        # Convert frame ke Tkinter image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_frame)
    
    def run(self):
        # Jalankan aplikasi
        if not self.camera.is_opened():
            print("[ERROR] Webcam not found. Exiting.")
            self.root.destroy()
            return
        
        print("[SUCCESS] Webcam connected.")
        self.update_frame()
        self.root.mainloop()
        
        # Cleanup
        self.camera.release()
        cv2.destroyAllWindows()

