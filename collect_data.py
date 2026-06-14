"""
Script untuk mengumpulkan data training dengan mudah
Capture gambar langsung dari camera dan simpan ke folder defect class
"""
import sys
import os
import cv2
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from camera_capture import CameraCapture
try:
    from config import Config
except ImportError:
    Config = None


class DataCollector:
    def __init__(self, camera_id=0):
        """Initialize data collector"""
        # Load config
        camera_id_to_use = Config.CAMERA_ID if Config is not None else camera_id
        self.camera = CameraCapture(camera_id_to_use)
        
        self.grade_labels = Config.GRADE_LABELS if Config is not None else ['Grade A', 'Grade B', 'Grade C']
        self.folder_names = [label.lower().replace(' ', '_') for label in self.grade_labels]
        
        self.current_folder = self.folder_names[0]
        self.current_label = self.grade_labels[0]
        
        self.count = {folder: 0 for folder in self.folder_names}
        
        # Buat folder jika belum ada dan hitung image yang sudah ada
        supported_formats = Config.SUPPORTED_FORMATS if Config is not None else ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        for folder in self.folder_names:
            os.makedirs(f'data/{folder}', exist_ok=True)
            if os.path.exists(f'data/{folder}'):
                existing_files = [f for f in os.listdir(f'data/{folder}') if f.lower().endswith(supported_formats)]
                self.count[folder] = len(existing_files)
    
    def save_image(self, frame):
        """Save image ke folder grade yang sesuai"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"data/{self.current_folder}/coffee_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        self.count[self.current_folder] += 1
        print(f"✓ Saved: {filename} (Total {self.current_label}: {self.count[self.current_folder]})")
    
    def draw_ui(self, frame):
        """Draw UI overlay pada frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay untuk panel atas
        overlay = frame.copy()
        box_height = 50 + len(self.grade_labels) * 30
        cv2.rectangle(overlay, (10, 10), (450, box_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Title
        cv2.putText(frame, "DATA COLLECTION MODE", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Current grade dengan highlight
        y_offset = 80
        for i, (label, folder) in enumerate(zip(self.grade_labels, self.folder_names)):
            is_active = (folder == self.current_folder)
            color = (0, 255, 0) if is_active else (150, 150, 150)
            marker = ">>>" if is_active else "   "
            text = f"{marker} [{i+1}] {label.upper()}: {self.count[folder]} images"
            cv2.putText(frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30
        
        # Instructions panel bawah
        cv2.rectangle(overlay, (10, h - 120), (550, h - 10), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        switch_keys = f"1-{len(self.grade_labels)}" if len(self.grade_labels) > 1 else "1"
        instructions = [
            "SPACE: Capture image",
            f"{switch_keys}: Switch defect class",
            "Q: Quit"
        ]
        
        y_offset = h - 95
        for instruction in instructions:
            cv2.putText(frame, instruction, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 30
        
        return frame
    
    def run(self):
        """Run data collection"""
        print("=" * 60)
        print("DATA COLLECTION - SISTEM GRADING KOPI")
        print("=" * 60)
        print("\nPanduan:")
        print("1. Letakkan biji kopi di depan camera")
        print("2. Tekan SPACE untuk capture")
        print(f"3. Tekan tombol angka untuk ganti kelas defect (1 s/d {len(self.grade_labels)})")
        print("4. Tekan Q untuk selesai")
        print("\nTips:")
        print("- Gunakan pencahayaan yang baik")
        print("- Background putih/netral")
        print("- Foto dari berbagai sudut")
        print("- Target: minimal 100 images per kelas")
        print("\n" + "=" * 60)
        
        self.camera.start()
        
        try:
            while True:
                ret, frame = self.camera.read_frame()
                
                if not ret:
                    print("Gagal membaca frame")
                    break
                
                # Draw UI
                display_frame = self.draw_ui(frame)
                
                # Show frame
                cv2.imshow('Data Collection', display_frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord(' '):  # Space bar
                    self.save_image(frame)
                elif ord('1') <= key <= ord('9'):
                    idx = key - ord('1')
                    if idx < len(self.grade_labels):
                        self.current_folder = self.folder_names[idx]
                        self.current_label = self.grade_labels[idx]
                        print(f"→ Switched to {self.current_label.upper()}")
        
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()
            
            # Summary
            print("\n" + "=" * 60)
            print("DATA COLLECTION SELESAI")
            print("=" * 60)
            total = sum(self.count.values())
            print(f"\nTotal images collected: {total}")
            for label, folder in zip(self.grade_labels, self.folder_names):
                print(f"  {label.upper()}: {self.count[folder]} images")
            print("\nData tersimpan di folder 'data/'")
            print("Selanjutnya jalankan: python train_model.py")


def main():
    collector = DataCollector(camera_id=0)
    collector.run()


if __name__ == "__main__":
    main()

