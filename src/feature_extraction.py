"""
Modul untuk ekstraksi fitur dari biji kopi
- Color features: HSV + LAB color space analysis
- Shape features: area, circularity, eccentricity, Hu moments
- Texture features: GLCM, LBP, Laplacian, Sobel gradients
"""
import cv2
import numpy as np

# Try import skimage untuk GLCM, fallback ke manual implementation
try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("WARNING: scikit-image tidak terinstall. GLCM/LBP features akan menggunakan fallback.")


class FeatureExtractor:
    def __init__(self):
        """Initialize feature extractor"""
        # LBP parameters
        self.lbp_radius = 2
        self.lbp_n_points = 8 * self.lbp_radius
        
        # GLCM parameters
        self.glcm_distances = [1, 2]
        self.glcm_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.glcm_levels = 16  # Quantization levels
    
    def extract_color_features(self, image):
        """
        Ekstraksi fitur warna dari biji kopi di HSV color space
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            dict: Color features
        """
        # Convert ke HSV untuk analisis warna
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Hitung mean dan std untuk setiap channel
        h_mean, h_std = cv2.meanStdDev(hsv[:, :, 0])
        s_mean, s_std = cv2.meanStdDev(hsv[:, :, 1])
        v_mean, v_std = cv2.meanStdDev(hsv[:, :, 2])
        
        # Hitung histogram untuk setiap channel (H, S, V)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_h = hist_h.flatten() / (hist_h.sum() + 1e-8)
        
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_s = hist_s.flatten() / (hist_s.sum() + 1e-8)
        
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        hist_v = hist_v.flatten() / (hist_v.sum() + 1e-8)
        
        return {
            'h_mean': float(h_mean[0][0]),
            'h_std': float(h_std[0][0]),
            's_mean': float(s_mean[0][0]),
            's_std': float(s_std[0][0]),
            'v_mean': float(v_mean[0][0]),
            'v_std': float(v_std[0][0]),
            'hist_h': hist_h,
            'hist_s': hist_s,
            'hist_v': hist_v
        }
    
    def extract_lab_color_features(self, image):
        """
        Ekstraksi fitur warna di LAB color space
        LAB lebih perceptually uniform dan lebih baik untuk membedakan
        warna-warna coklat/roast biji kopi
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            dict: LAB color features
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        l_mean, l_std = cv2.meanStdDev(lab[:, :, 0])
        a_mean, a_std = cv2.meanStdDev(lab[:, :, 1])
        b_mean, b_std = cv2.meanStdDev(lab[:, :, 2])
        
        # Skewness dan kurtosis untuk distribusi warna
        l_channel = lab[:, :, 0].flatten().astype(np.float64)
        a_channel = lab[:, :, 1].flatten().astype(np.float64)
        b_channel = lab[:, :, 2].flatten().astype(np.float64)
        
        def calc_skewness(data):
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return float(np.mean(((data - mean) / std) ** 3))
        
        def calc_kurtosis(data):
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return float(np.mean(((data - mean) / std) ** 4) - 3.0)
        
        return {
            'lab_l_mean': float(l_mean[0][0]),
            'lab_l_std': float(l_std[0][0]),
            'lab_a_mean': float(a_mean[0][0]),
            'lab_a_std': float(a_std[0][0]),
            'lab_b_mean': float(b_mean[0][0]),
            'lab_b_std': float(b_std[0][0]),
            'lab_l_skew': calc_skewness(l_channel),
            'lab_a_skew': calc_skewness(a_channel),
            'lab_b_skew': calc_skewness(b_channel),
        }
    
    def extract_shape_features(self, contour):
        """
        Ekstraksi fitur bentuk dari contour biji kopi
        Ditambah Hu Moments untuk rotation-invariant shape descriptor
        
        Args:
            contour: Contour dari biji kopi
            
        Returns:
            dict: Shape features
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Circularity (seberapa bulat)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Fit ellipse jika contour cukup besar
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (x, y), (MA, ma), angle = ellipse
            # Fix: MA adalah major axis, ma adalah minor axis
            major = max(MA, ma)
            minor = min(MA, ma)
            eccentricity = np.sqrt(1 - (minor / major) ** 2) if major > 0 else 0
            aspect_ratio_ellipse = major / minor if minor > 0 else 0
        else:
            eccentricity = 0
            angle = 0
            aspect_ratio_ellipse = 1.0
        
        # Convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Convexity defects count
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        try:
            defects = cv2.convexityDefects(contour, hull_indices)
            n_defects = len(defects) if defects is not None else 0
        except cv2.error:
            n_defects = 0
        
        # Hu Moments (rotation, scale, translation invariant)
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        # Log transform untuk Hu moments (range sangat besar)
        hu_moments_log = np.array([
            -np.sign(h) * np.log10(abs(h) + 1e-10) for h in hu_moments
        ])
        
        # Bounding rect aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        rect_aspect_ratio = float(w) / h if h > 0 else 0
        
        # Extent (area / bounding rect area)
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0
        
        return {
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'eccentricity': eccentricity,
            'solidity': solidity,
            'angle': angle,
            'aspect_ratio_ellipse': aspect_ratio_ellipse,
            'rect_aspect_ratio': rect_aspect_ratio,
            'extent': extent,
            'n_defects': n_defects,
            'hu_moments': hu_moments_log
        }
    
    def extract_texture_features(self, image):
        """
        Ekstraksi fitur tekstur menggunakan Laplacian, Sobel, dan statistik
        
        Args:
            image: Input image (BGR atau grayscale)
            
        Returns:
            dict: Texture features
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Laplacian variance (ukuran sharpness/blur)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = float(laplacian.var())
        laplacian_mean = float(np.abs(laplacian).mean())
        
        # Gradient magnitude menggunakan Sobel
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        gradient_mean = float(gradient_mag.mean())
        gradient_std = float(gradient_mag.std())
        
        # Gradient direction histogram
        gradient_dir = np.arctan2(sobely, sobelx + 1e-8)
        dir_hist, _ = np.histogram(gradient_dir, bins=8, range=(-np.pi, np.pi))
        dir_hist = dir_hist.astype(np.float32)
        dir_hist = dir_hist / (dir_hist.sum() + 1e-8)
        
        # Gray-level statistics
        gray_float = gray.astype(np.float64)
        gray_mean = float(gray_float.mean())
        gray_std = float(gray_float.std())
        gray_entropy = float(-np.sum(
            (np.histogram(gray, bins=256, range=(0, 256))[0].astype(np.float64) / gray.size + 1e-10) *
            np.log2(np.histogram(gray, bins=256, range=(0, 256))[0].astype(np.float64) / gray.size + 1e-10)
        ))
        
        result = {
            'laplacian_var': laplacian_var,
            'laplacian_mean': laplacian_mean,
            'gradient_mean': gradient_mean,
            'gradient_std': gradient_std,
            'gradient_dir_hist': dir_hist,
            'gray_mean': gray_mean,
            'gray_std': gray_std,
            'gray_entropy': gray_entropy,
        }
        
        return result
    
    def extract_glcm_features(self, image):
        """
        Ekstraksi fitur tekstur GLCM (Gray-Level Co-occurrence Matrix)
        Metode standar industri untuk analisis tekstur
        
        Args:
            image: Input image (BGR atau grayscale)
            
        Returns:
            dict: GLCM texture features
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if HAS_SKIMAGE:
            # Quantize ke fewer levels untuk GLCM yang stabil
            gray_quantized = (gray // (256 // self.glcm_levels)).astype(np.uint8)
            
            # Compute GLCM
            glcm = graycomatrix(
                gray_quantized,
                distances=self.glcm_distances,
                angles=self.glcm_angles,
                levels=self.glcm_levels,
                symmetric=True,
                normed=True
            )
            
            # Extract GLCM properties (rata-rata dari semua distances & angles)
            contrast = float(graycoprops(glcm, 'contrast').mean())
            dissimilarity = float(graycoprops(glcm, 'dissimilarity').mean())
            homogeneity = float(graycoprops(glcm, 'homogeneity').mean())
            energy = float(graycoprops(glcm, 'energy').mean())
            correlation = float(graycoprops(glcm, 'correlation').mean())
            asm = float(graycoprops(glcm, 'ASM').mean())
        else:
            # Fallback: implementasi sederhana
            contrast = float(gray.astype(np.float64).var())
            dissimilarity = float(np.abs(np.diff(gray.astype(np.float64), axis=1)).mean())
            homogeneity = 1.0 / (1.0 + contrast)
            energy = float(np.sum(gray.astype(np.float64)**2) / (gray.size * 255.0**2))
            correlation = 0.0
            asm = energy ** 2
        
        return {
            'glcm_contrast': contrast,
            'glcm_dissimilarity': dissimilarity,
            'glcm_homogeneity': homogeneity,
            'glcm_energy': energy,
            'glcm_correlation': correlation,
            'glcm_asm': asm,
        }
    
    def extract_lbp_features(self, image, n_bins=16):
        """
        Ekstraksi fitur Local Binary Pattern (LBP)
        Untuk texture recognition yang robust
        
        Args:
            image: Input image (BGR atau grayscale)
            n_bins: Number of histogram bins
            
        Returns:
            dict: LBP features
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if HAS_SKIMAGE:
            lbp = local_binary_pattern(
                gray, self.lbp_n_points, self.lbp_radius, method='uniform'
            )
            
            # Histogram of LBP
            n_bins_actual = self.lbp_n_points + 2  # uniform LBP
            lbp_hist, _ = np.histogram(
                lbp.ravel(), bins=n_bins_actual,
                range=(0, n_bins_actual), density=True
            )
        else:
            # Fallback: simplified LBP
            lbp_hist = self._compute_simple_lbp(gray, n_bins)
        
        return {
            'lbp_hist': lbp_hist.astype(np.float32),
            'lbp_mean': float(lbp_hist.mean()),
            'lbp_std': float(lbp_hist.std()),
            'lbp_entropy': float(-np.sum(
                (lbp_hist + 1e-10) * np.log2(lbp_hist + 1e-10)
            ))
        }
    
    def _compute_simple_lbp(self, gray, n_bins=16):
        """Fallback LBP implementation tanpa skimage"""
        h, w = gray.shape
        lbp_img = np.zeros((h - 2, w - 2), dtype=np.uint8)
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] >= center) << 7
                code |= (gray[i-1, j] >= center) << 6
                code |= (gray[i-1, j+1] >= center) << 5
                code |= (gray[i, j+1] >= center) << 4
                code |= (gray[i+1, j+1] >= center) << 3
                code |= (gray[i+1, j] >= center) << 2
                code |= (gray[i+1, j-1] >= center) << 1
                code |= (gray[i, j-1] >= center) << 0
                lbp_img[i-1, j-1] = code
        
        hist, _ = np.histogram(lbp_img.ravel(), bins=n_bins, range=(0, 256), density=True)
        return hist.astype(np.float32)
    
    def extract_all_features(self, image, contour):
        """
        Ekstraksi semua fitur dari biji kopi (comprehensive)
        
        Args:
            image: Input image
            contour: Contour dari biji kopi
            
        Returns:
            dict: All features combined
        """
        color_features = self.extract_color_features(image)
        lab_features = self.extract_lab_color_features(image)
        shape_features = self.extract_shape_features(contour)
        texture_features = self.extract_texture_features(image)
        glcm_features = self.extract_glcm_features(image)
        lbp_features = self.extract_lbp_features(image)
        
        # Combine all features
        all_features = {
            **color_features,
            **lab_features,
            **shape_features,
            **texture_features,
            **glcm_features,
            **lbp_features,
        }
        
        return all_features
    
    def features_to_vector(self, features):
        """
        Convert features dict ke vector untuk ML model
        
        Args:
            features: Dictionary of features
            
        Returns:
            numpy.ndarray: Feature vector
        """
        # Scalar features
        scalar_keys = [
            # HSV color
            'h_mean', 'h_std', 's_mean', 's_std', 'v_mean', 'v_std',
            # LAB color
            'lab_l_mean', 'lab_l_std', 'lab_a_mean', 'lab_a_std',
            'lab_b_mean', 'lab_b_std',
            'lab_l_skew', 'lab_a_skew', 'lab_b_skew',
            # Shape
            'area', 'perimeter', 'circularity', 'eccentricity',
            'solidity', 'rect_aspect_ratio', 'extent', 'n_defects',
            # Texture
            'laplacian_var', 'laplacian_mean', 'gradient_mean', 'gradient_std',
            'gray_mean', 'gray_std', 'gray_entropy',
            # GLCM
            'glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity',
            'glcm_energy', 'glcm_correlation', 'glcm_asm',
            # LBP
            'lbp_mean', 'lbp_std', 'lbp_entropy',
        ]
        
        feature_vector = [features.get(k, 0.0) for k in scalar_keys]
        
        # Hu moments (7 values)
        if 'hu_moments' in features:
            feature_vector.extend(features['hu_moments'].tolist())
        
        return np.array(feature_vector, dtype=np.float32)
