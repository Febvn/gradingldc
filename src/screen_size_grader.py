"""
Modul untuk menghitung ukuran fisik (Screen Size) biji kopi.
Menggunakan minAreaRect untuk mendapatkan dimensi piksel (panjang x lebar).
Lalu dikonversi ke mm untuk dipetakan ke ukuran ayakan ekspor standar (Screen 18, 16, dll).
"""

import cv2

class ScreenSizeGrader:
    def __init__(self, pixels_per_mm=10.0):
        """
        Inisialisasi kalkulator screen size.
        Args:
            pixels_per_mm (float): Rasio kalibrasi jarak kamera (berapa piksel untuk 1 milimeter).
        """
        self.pixels_per_mm = pixels_per_mm
        
    def get_physical_dimensions(self, contour):
        """
        Mengembalikan panjang dan lebar biji kopi dalam milimeter.
        """
        if len(contour) < 5:
            return 0.0, 0.0
            
        # minAreaRect returns: (center(x, y), (width, height), angle of rotation)
        rect = cv2.minAreaRect(contour)
        (w_px, h_px) = rect[1]
        
        # Sumbu mayor (panjang) dan minor (lebar). Lebar selalu yang terkecil.
        width_px = min(w_px, h_px)
        length_px = max(w_px, h_px)
        
        width_mm = width_px / self.pixels_per_mm
        length_mm = length_px / self.pixels_per_mm
        
        return length_mm, width_mm

    def determine_screen_size(self, width_mm):
        """
        Menentukan Screen Size (1/64 inci) berdasarkan ukuran lebar biji.
        Standar perkiraan milimeter:
        Screen 18 = ~7.14 mm
        Screen 17 = ~6.75 mm
        Screen 16 = ~6.35 mm
        Screen 15 = ~5.95 mm
        Screen 14 = ~5.55 mm
        """
        if width_mm >= 7.14:
            return "Scr 18"
        elif width_mm >= 6.75:
            return "Scr 17"
        elif width_mm >= 6.35:
            return "Scr 16"
        elif width_mm >= 5.95:
            return "Scr 15"
        elif width_mm >= 5.55:
            return "Scr 14"
        else:
            return "< Scr 14"
