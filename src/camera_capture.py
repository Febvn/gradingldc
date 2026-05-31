"""
Modul untuk capture image dari camera realtime
"""
import cv2
import numpy as np


class CameraCapture:
    def __init__(self, camera_id=0):
        """
        Initialize camera capture
        
        Args:
            camera_id: ID camera (default 0 untuk webcam)
        """
        self.camera_id = camera_id
        self.cap = None
        
    def start(self):
        """Mulai camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise Exception(f"Tidak bisa membuka camera {self.camera_id}")
        
        # Set resolusi camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"Camera {self.camera_id} berhasil diaktifkan")
        
    def read_frame(self):
        """
        Baca frame dari camera
        
        Returns:
            tuple: (success, frame)
        """
        if self.cap is None:
            raise Exception("Camera belum diaktifkan. Panggil start() terlebih dahulu")
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def stop(self):
        """Stop camera capture"""
        if self.cap is not None:
            self.cap.release()
            print("Camera dimatikan")
    
    def __del__(self):
        """Destructor untuk memastikan camera dimatikan"""
        self.stop()
