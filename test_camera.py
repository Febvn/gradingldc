"""
Script untuk test camera dan deteksi biji kopi
"""
import sys
import os
import cv2

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from camera_capture import CameraCapture
from preprocessing import ImagePreprocessor


def main():
    """Test camera dan preprocessing"""
    print("=" * 50)
    print("TEST CAMERA DAN DETEKSI BIJI KOPI")
    print("=" * 50)
    print()
    print("Tekan 'q' untuk quit")
    print("Tekan 's' untuk save image")
    print()
    
    # Initialize
    camera = CameraCapture(camera_id=0)
    preprocessor = ImagePreprocessor()
    
    try:
        camera.start()
        
        while True:
            # Read frame
            ret, frame = camera.read_frame()
            
            if not ret:
                print("Gagal membaca frame")
                break
            
            # Detect coffee beans
            detected_beans = preprocessor.detect_coffee_beans(frame)
            
            # Draw detections
            display_frame = frame.copy()
            
            for bean in detected_beans:
                x, y, w, h = bean['bbox']
                area = bean['area']
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw label
                label = f"Area: {int(area)}"
                cv2.putText(display_frame, label, (x, y - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw count
            count_text = f"Detected: {len(detected_beans)}"
            cv2.putText(display_frame, count_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Camera Test - Coffee Bean Detection', display_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = "test_capture.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"Image saved: {filename}")
    
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Camera test selesai")


if __name__ == "__main__":
    main()
