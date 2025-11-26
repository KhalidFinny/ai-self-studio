import cv2
import time
import threading
import webbrowser
from flask import Flask, render_template, Response, jsonify

class WebApp:
    def __init__(self, camera, detector, imaging_edge, countdown_seconds=3):
        self.camera = camera
        self.detector = detector
        self.imaging_edge = imaging_edge
        self.countdown_seconds = countdown_seconds
        
        self.app = Flask(__name__)
        self.frame = None
        self.lock = threading.Lock()
        self.is_running = True
        
        # Shared state for UI
        self.state = {
            "countdown": None,
            "message": "",
            "flash": False
        }
        
        # Setup routes
        self.setup_routes()
        
        # Start camera loop in background
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()

    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/status')
        def status():
            return jsonify(self.state)

    def camera_loop(self):
        while self.is_running:
            if not self.camera.is_opened():
                time.sleep(1)
                continue
                
            ret, frame = self.camera.read()
            if not ret:
                continue
                
            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect gesture
            # Note: We modify the frame in-place with annotations if needed, 
            # or just use the detection result to trigger logic.
            # For the web UI, we might want a clean feed or annotated feed.
            # Let's use the annotated feed from the detector if it supports it,
            # otherwise we just use the raw frame and handle logic.
            
            # The detector.detect method returns (frame, detected_fist, should_trigger)
            # It likely draws on the frame too.
            frame, detected_fist, should_trigger = self.detector.detect(frame)
            
            if should_trigger and self.state["countdown"] is None:
                print("[INFO] Triggering countdown from WebApp")
                threading.Thread(target=self.start_countdown, daemon=True).start()
            
            with self.lock:
                self.frame = frame
            
            time.sleep(0.01)

    def generate_frames(self):
        while self.is_running:
            with self.lock:
                if self.frame is None:
                    continue
                
                # Encode frame to jpg
                ret, buffer = cv2.imencode('.jpg', self.frame)
                frame = buffer.tobytes()
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03) # Limit to ~30fps streaming

    def start_countdown(self):
        self.detector.set_trigger_active(True)
        self.state["message"] = "Get Ready..."
        
        for i in range(self.countdown_seconds, 0, -1):
            self.state["countdown"] = i
            self.state["message"] = str(i)
            time.sleep(1)
        
        self.state["countdown"] = 0
        self.state["message"] = "SMILE!"
        
        # Trigger shutter logic
        if not self.imaging_edge.find_window():
            self.state["message"] = "Camera Not Found!"
        else:
            if self.imaging_edge.trigger_shutter():
                self.state["flash"] = True
                print("[SUCCESS] Shutter executed")
                # Reset flash after a short delay
                threading.Timer(0.5, lambda: self.state.update({"flash": False})).start()
            else:
                self.state["message"] = "Shutter Failed!"
        
        time.sleep(2)
        self.state["countdown"] = None
        self.state["message"] = ""
        self.detector.set_trigger_active(False)

    def run(self, host='0.0.0.0', port=5000):
        # Open browser automatically
        webbrowser.open(f'http://127.0.0.1:{port}')
        self.app.run(host=host, port=port, debug=False, use_reloader=False)
