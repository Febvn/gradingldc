"""
Modul untuk preprocessing image biji kopi
- CLAHE untuk normalisasi pencahayaan
- Adaptive thresholding untuk segmentasi robust
- Morphological operations untuk noise removal
- Multi-scale detection support
"""
import cv2
import numpy as np

try:
    from config import Config
except ImportError:
    Config = None


class ImagePreprocessor:
    def __init__(self, target_size=(224, 224)):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target size untuk resize (width, height)
        """
        if Config is not None:
            self.target_size = Config.TARGET_SIZE
            self.clahe = cv2.createCLAHE(
                clipLimit=Config.CLAHE_CLIP_LIMIT, 
                tileGridSize=Config.CLAHE_TILE_SIZE
            )
            self.min_area = Config.BEAN_MIN_AREA
            self.max_area = Config.BEAN_MAX_AREA
            self.min_aspect_ratio = Config.BEAN_MIN_ASPECT_RATIO
            self.max_aspect_ratio = Config.BEAN_MAX_ASPECT_RATIO
            self.min_solidity = Config.BEAN_MIN_SOLIDITY
            self.min_circularity = Config.BEAN_MIN_CIRCULARITY
        else:
            self.target_size = target_size
            self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            self.min_area = 500
            self.max_area = 50000
            self.min_aspect_ratio = 0.4
            self.max_aspect_ratio = 2.5
            self.min_solidity = 0.6
            self.min_circularity = 0.3
        
        # Morphological kernels
        self.kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    def enhance_image(self, image):
        """
        Enhance image menggunakan CLAHE di LAB color space
        untuk normalisasi pencahayaan yang tidak merata
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            numpy.ndarray: Enhanced image
        """
        # Convert ke LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE pada L channel (lightness)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        
        # Convert balik ke BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def preprocess(self, image):
        """
        Preprocessing image untuk deteksi biji kopi
        Menggunakan CLAHE + adaptive thresholding + morphological ops
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            dict: Dictionary berisi processed images
        """
        # Step 1: Enhance pencahayaan dengan CLAHE
        enhanced = self.enhance_image(image)
        
        # Step 2: Convert ke grayscale
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Step 3: Bilateral filter (edge-preserving noise removal)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Step 4: Adaptive thresholding (lebih robust daripada Otsu)
        thresh_adaptive = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15, 5
        )
        
        # Step 5: Otsu threshold sebagai fallback/kombinasi
        _, thresh_otsu = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Step 6: Combine kedua threshold (AND operation)
        thresh_combined = cv2.bitwise_or(thresh_adaptive, thresh_otsu)
        
        # Step 7: Morphological operations untuk clean up
        # Close: menutup lubang kecil di dalam objek
        thresh_clean = cv2.morphologyEx(
            thresh_combined, cv2.MORPH_CLOSE,
            self.kernel_ellipse, iterations=2
        )
        # Open: menghilangkan noise kecil di luar objek
        thresh_clean = cv2.morphologyEx(
            thresh_clean, cv2.MORPH_OPEN,
            self.kernel_small, iterations=1
        )
        
        return {
            'original': image,
            'enhanced': enhanced,
            'gray': gray,
            'blurred': blurred,
            'thresh_adaptive': thresh_adaptive,
            'thresh_otsu': thresh_otsu,
            'thresh': thresh_clean
        }
    
    def detect_coffee_beans(self, image):
        """
        Deteksi biji kopi dalam image dengan filtering yang lebih robust
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            list: List of detected coffee bean contours dan bounding boxes
        """
        processed = self.preprocess(image)
        thresh = processed['thresh']
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        detected_beans = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter berdasarkan area
            if area < self.min_area or area > self.max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter berdasarkan aspect ratio
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            
            # Filter berdasarkan solidity (area / convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Biji kopi harus cukup solid (bukan bentuk irregular)
            if solidity < self.min_solidity:
                continue
            
            # Circularity check
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Filter: terlalu tidak bundar mungkin bukan biji kopi
            if circularity < self.min_circularity:
                continue
            
            detected_beans.append({
                'contour': contour,
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'circularity': circularity
            })
        
        # Sort by area (terbesar dulu) untuk prioritas
        detected_beans.sort(key=lambda b: b['area'], reverse=True)
        
        return detected_beans
    
    def extract_bean_roi(self, image, bbox, padding=10):
        """
        Extract region of interest (ROI) untuk satu biji kopi
        dengan masking untuk menghilangkan background
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, w, h)
            padding: Padding around bbox
            
        Returns:
            numpy.ndarray: Cropped image of coffee bean
        """
        x, y, w, h = bbox
        h_img, w_img = image.shape[:2]
        
        # Add padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)
        
        roi = image[y1:y2, x1:x2]
        
        # Resize ke target size
        if roi.size > 0:
            roi_resized = cv2.resize(roi, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            return roi_resized
        
        return None
    
    def normalize_image(self, image):
        """
        Normalize image untuk input ke model ML
        Menggunakan preprocessing yang sesuai dengan EfficientNet
        (range [-1, 1])
        
        Args:
            image: Input image (uint8, 0-255)
            
        Returns:
            numpy.ndarray: Normalized image
        """
        # EfficientNet / MobileNetV2 preprocessing: scale ke [-1, 1]
        normalized = image.astype(np.float32)
        normalized = (normalized / 127.5) - 1.0
        return normalized
    
    def normalize_image_simple(self, image):
        """
        Normalize image ke range [0, 1] (untuk feature extraction)
        
        Args:
            image: Input image (uint8, 0-255)
            
        Returns:
            numpy.ndarray: Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def apply_nms(self, detections, iou_threshold=0.3):
        """
        Non-Maximum Suppression untuk menghapus deteksi duplikat
        
        Args:
            detections: List of detected beans dengan bbox
            iou_threshold: IoU threshold untuk suppression
            
        Returns:
            list: Filtered detections
        """
        if len(detections) == 0:
            return []
        
        boxes = np.array([d['bbox'] for d in detections])
        areas = np.array([d['area'] for d in detections])
        
        # Convert (x, y, w, h) ke (x1, y1, x2, y2)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        # Sort by area (descending)
        indices = np.argsort(areas)[::-1]
        
        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Compute IoU with remaining boxes
            rest = indices[1:]
            
            xx1 = np.maximum(x1[current], x1[rest])
            yy1 = np.maximum(y1[current], y1[rest])
            xx2 = np.minimum(x2[current], x2[rest])
            yy2 = np.minimum(y2[current], y2[rest])
            
            intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            union = areas[current] + areas[rest] - intersection
            iou = intersection / (union + 1e-6)
            
        # Keep boxes with IoU below threshold
            remaining = np.where(iou < iou_threshold)[0]
            indices = rest[remaining]
        
        return [detections[i] for i in keep]

    def apply_auto_white_balance(self, image):
        """
        Apply Auto White Balance (Gray World algorithm) to normalize colors.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            numpy.ndarray: Balanced image
        """
        # Fallback Gray World manual implementation
        b, g, r = cv2.split(image)
        r_avg = np.mean(r)
        g_avg = np.mean(g)
        b_avg = np.mean(b)
        
        if r_avg == 0 or g_avg == 0 or b_avg == 0:
            return image
            
        gray_val = (r_avg + g_avg + b_avg) / 3.0
        
        kr = gray_val / r_avg
        kg = gray_val / g_avg
        kb = gray_val / b_avg
        
        r_balanced = np.clip(r * kr, 0, 255).astype(np.uint8)
        g_balanced = np.clip(g * kg, 0, 255).astype(np.uint8)
        b_balanced = np.clip(b * kb, 0, 255).astype(np.uint8)
        
        return cv2.merge([b_balanced, g_balanced, r_balanced])

    def check_image_quality(self, image):
        """
        Check quality of the image: brightness, sharpness (Laplacian variance), and contrast.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            tuple: (is_good_quality, reason_dict)
        """
        # Convert to grayscale for calculations
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Brightness (mean pixel value)
        mean_brightness = np.mean(gray)
        
        # 2. Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 3. Contrast (standard deviation)
        std_contrast = np.std(gray)
        
        reasons = {
            'mean_brightness': float(mean_brightness),
            'sharpness': float(laplacian_var),
            'contrast': float(std_contrast),
            'issues': []
        }
        
        is_good = True
        
        # Read parameters from Config or defaults
        if Config is not None:
            min_brightness = Config.QUALITY_MIN_BRIGHTNESS
            max_brightness = Config.QUALITY_MAX_BRIGHTNESS
            min_sharpness = Config.QUALITY_MIN_SHARPNESS
            min_contrast = Config.QUALITY_MIN_CONTRAST
        else:
            min_brightness = 30
            max_brightness = 240
            min_sharpness = 50.0
            min_contrast = 20.0
            
        if mean_brightness < min_brightness:
            is_good = False
            reasons['issues'].append("Terlalu gelap")
        elif mean_brightness > max_brightness:
            is_good = False
            reasons['issues'].append("Terlalu terang / overexposed")
            
        if laplacian_var < min_sharpness:
            is_good = False
            reasons['issues'].append("Buram / out of focus")
            
        if std_contrast < min_contrast:
            is_good = False
            reasons['issues'].append("Kontras terlalu rendah")
            
        return is_good, reasons

    def segment_kmeans(self, image, k=3):
        """
        Segmentasi warna menggunakan K-Means Clustering (Unsupervised Learning).
        Mengelompokkan warna pixel ke dalam k cluster untuk segmentasi detail.
        
        Args:
            image: Input BGR image
            k: Jumlah cluster warna
            
        Returns:
            numpy.ndarray: Segmented BGR image
        """
        # Convert ke LAB color space (lebih baik untuk representasi warna seragam)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Reshape data ke 2D (pixels, channels)
        pixel_vals = lab.reshape((-1, 3)).astype(np.float32)
        
        # Define criteria (maximum 100 iterations atau epsilon 0.2)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        
        # Run K-Means
        _, labels, centers = cv2.kmeans(
            pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Convert centers back to uint8
        centers = np.uint8(centers)
        
        # Map labels ke centers
        segmented_data = centers[labels.flatten()]
        
        # Reshape kembali ke original shape
        segmented_lab = segmented_data.reshape(image.shape)
        
        # Convert back ke BGR
        segmented_bgr = cv2.cvtColor(segmented_lab, cv2.COLOR_LAB2BGR)
        
        return segmented_bgr

