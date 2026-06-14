"""
Modul untuk capture image dari camera realtime
"""
import cv2
import numpy as np
import threading
import time


class CameraCapture:
    def __init__(self, camera_id=0, threaded=True):
        """
        Initialize camera capture
        
        Args:
            camera_id: ID camera (default 0 untuk webcam)
            threaded: Gunakan background thread untuk membaca frame (meningkatkan FPS & latency)
        """
        self.camera_id = camera_id
        self.cap = None
        self.threaded = threaded
        self.running = False
        self.thread = None
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        
    def start(self):
        """Mulai camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise Exception(f"Tidak bisa membuka camera {self.camera_id}")
        
        # Set resolusi camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.ret, self.frame = self.cap.read()
        self.running = True
        
        if self.threaded:
            self.thread = threading.Thread(target=self._update_loop, name="CameraCaptureThread")
            self.thread.daemon = True
            self.thread.start()
            print(f"Threaded Camera {self.camera_id} berhasil diaktifkan")
        else:
            print(f"Camera {self.camera_id} berhasil diaktifkan (Synchronous)")
        
    def _update_loop(self):
        """Loop internal thread untuk membaca frame secara berkelanjutan"""
        while self.running:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.ret = ret
                        self.frame = frame
            time.sleep(0.005)  # Sleep singkat untuk mengurangi beban CPU
            
    def read_frame(self):
        """
        Baca frame dari camera
        
        Returns:
            tuple: (success, frame)
        """
        if self.cap is None:
            raise Exception("Camera belum diaktifkan. Panggil start() terlebih dahulu")
        
        if self.threaded:
            with self.lock:
                return self.ret, (self.frame.copy() if self.frame is not None else None)
        else:
            ret, frame = self.cap.read()
            return ret, frame
    
    def stop(self):
        """Stop camera capture"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
            
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Camera dimatikan")
    
    def __del__(self):
        """Destructor untuk memastikan camera dimatikan"""
        self.stop()

