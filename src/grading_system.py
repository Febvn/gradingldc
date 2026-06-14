"""
Sistem grading kopi realtime - Main system
- Temporal smoothing untuk prediksi stabil
- Confidence threshold untuk filter prediksi lemah
- Non-Maximum Suppression (NMS) untuk hapus duplikat
- FPS counter untuk monitoring performa
- Improved visualization
"""
import cv2
import numpy as np
import time
import tensorflow as tf
from collections import deque
from camera_capture import CameraCapture
from preprocessing import ImagePreprocessor
from feature_extraction import FeatureExtractor
from model import CoffeeGradingModel

try:
    from config import Config
except ImportError:
    Config = None


class PredictionSmoother:
    """
    Temporal smoothing untuk prediksi yang stabil antar frame.
    Menggunakan exponential moving average dari probabilitas
    untuk menghilangkan flicker pada prediksi.
    """
    
    def __init__(self, window_size=7, alpha=0.6):
        """
        Args:
            window_size: Jumlah frame history yang disimpan
            alpha: Weight untuk exponential moving average (0-1)
                   Lebih tinggi = lebih responsif, lebih rendah = lebih smooth
        """
        self.window_size = window_size
        self.alpha = alpha
        self.histories = {}  # Per-bean tracking berdasarkan posisi
    
    def _get_key(self, bbox, tolerance=50):
        """
        Generate key berdasarkan posisi bbox untuk tracking per-biji.
        Biji yang posisinya berdekatan dianggap sama.
        """
        cx = bbox[0] + bbox[2] // 2
        cy = bbox[1] + bbox[3] // 2
        # Quantize posisi
        key = (cx // tolerance, cy // tolerance)
        return key
    
    def smooth(self, bbox, probabilities):
        """
        Smooth prediksi menggunakan exponential moving average
        
        Args:
            bbox: Bounding box (x, y, w, h)
            probabilities: Array probabilitas per-class
            
        Returns:
            numpy.ndarray: Smoothed probabilities
        """
        key = self._get_key(bbox)
        
        if key not in self.histories:
            self.histories[key] = deque(maxlen=self.window_size)
        
        self.histories[key].append(probabilities.copy())
        
        # Exponential moving average
        history = list(self.histories[key])
        n = len(history)
        
        if n == 1:
            return probabilities
        
        smoothed = np.zeros_like(probabilities)
        weight_sum = 0.0
        
        for i, p in enumerate(history):
            weight = self.alpha ** (n - 1 - i)
            smoothed += weight * p
            weight_sum += weight
        
        smoothed /= weight_sum
        return smoothed
    
    def cleanup(self, active_keys):
        """Hapus history untuk biji yang sudah tidak terdeteksi"""
        keys_to_remove = [k for k in self.histories if k not in active_keys]
        for k in keys_to_remove:
            del self.histories[k]


class FPSCounter:
    """Counter FPS untuk monitoring performa realtime"""
    
    def __init__(self, avg_window=30):
        self.timestamps = deque(maxlen=avg_window)
    
    def tick(self):
        """Record timestamp saat ini"""
        self.timestamps.append(time.time())
    
    def get_fps(self):
        """Get average FPS"""
        if len(self.timestamps) < 2:
            return 0.0
        
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed == 0:
            return 0.0
        
        return (len(self.timestamps) - 1) / elapsed


class CoffeeGradingSystem:
    def __init__(self, model_path=None, camera_id=None, confidence_threshold=None):
        """
        Initialize sistem grading kopi
        
        Args:
            model_path: Path ke trained model (optional)
            camera_id: ID camera
            confidence_threshold: Minimum confidence untuk valid prediction
        """
        # Load from config if available
        if Config is not None:
            c_id = camera_id if camera_id is not None else Config.CAMERA_ID
            self.confidence_threshold = confidence_threshold if confidence_threshold is not None else Config.CONFIDENCE_THRESHOLD
            m_path = model_path if model_path is not None else Config.MODEL_SAVE_PATH
            
            # Smoother
            self.smoother = PredictionSmoother(
                window_size=Config.SMOOTHER_WINDOW_SIZE, 
                alpha=Config.SMOOTHER_ALPHA
            )
            # FPS counter
            self.fps_counter = FPSCounter(avg_window=Config.FPS_AVG_WINDOW)
            
            # Setup features activation flags
            self.awb_enabled = Config.AWB_ENABLED
            self.quality_check_enabled = Config.QUALITY_CHECK_ENABLED
            self.gradcam_enabled = Config.GRADCAM_ENABLED
        else:
            c_id = camera_id if camera_id is not None else 0
            self.confidence_threshold = confidence_threshold if confidence_threshold is not None else 0.6
            m_path = model_path
            
            self.smoother = PredictionSmoother(window_size=7, alpha=0.6)
            self.fps_counter = FPSCounter(avg_window=30)
            
            self.awb_enabled = True
            self.quality_check_enabled = True
            self.gradcam_enabled = True
            
        self.camera = CameraCapture(c_id)
        self.preprocessor = ImagePreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.model = CoffeeGradingModel(m_path)
        
        # Build dynamic label lists
        if Config is not None:
            grade_labels = Config.GRADE_LABELS
        else:
            grade_labels = ['Grade A', 'Grade B', 'Grade C']
        
        # UI controls
        self.show_gradcam = False
        
        # Latency statistics
        self.last_latency = {'preproc': 0.0, 'inference': 0.0, 'total': 0.0}
        
        # Statistics — dynamically built from labels
        self.stats = {label: 0 for label in grade_labels}
        self.stats['Uncertain'] = 0
        self.stats['total'] = 0
        
        # Colors untuk visualization (BGR format) — distinct for up to 6+ classes
        _color_palette = [
            (0, 200, 0),      # Green   → Normal / Premium
            (0, 0, 220),      # Red     → Biji Hitam
            (0, 120, 220),    # Dark Orange → Biji Cokelat
            (220, 180, 0),    # Cyan-blue → Berlubang
            (180, 0, 180),    # Purple  → Pecah
            (0, 180, 220),    # Yellow  → Berjamur
            (255, 100, 100),  # Light blue (extra)
            (100, 255, 100),  # Light green (extra)
        ]
        self.grade_colors = {}
        for i, label in enumerate(grade_labels):
            self.grade_colors[label] = _color_palette[i % len(_color_palette)]
        self.grade_colors['Uncertain'] = (128, 128, 128)  # Gray
        
        # Grade descriptions
        _default_descriptions = {
            'Normal': 'Biji Normal (Tanpa Cacat)',
            'Biji Hitam': 'Cacat Hitam Penuh/Parsial',
            'Biji Cokelat': 'Cacat Over-Fermentasi',
            'Berlubang': 'Cacat Lubang Serangga',
            'Pecah': 'Cacat Pecah/Retak',
            'Berjamur': 'Cacat Kontaminasi Jamur',
            'Grade A': 'Kualitas Premium',
            'Grade B': 'Kualitas Standar',
            'Grade C': 'Kualitas Rendah',
        }
        self.grade_descriptions = {}
        for label in grade_labels:
            self.grade_descriptions[label] = _default_descriptions.get(label, label)
        self.grade_descriptions['Uncertain'] = 'Tidak Yakin'
    
    def start(self):
        """Start camera"""
        self.camera.start()
    
    def stop(self):
        """Stop camera"""
        self.camera.stop()
    
    def process_frame(self, frame, use_tta=False):
        """
        Process satu frame untuk grading
        Dengan NMS, temporal smoothing, dan confidence filtering
        
        Args:
            frame: Input frame dari camera
            use_tta: Gunakan Test-Time Augmentation
            
        Returns:
            tuple: (annotated_frame, results)
        """
        t_start = time.time()
        
        # Apply Auto White Balance if enabled
        if self.awb_enabled:
            frame_processed = self.preprocessor.apply_auto_white_balance(frame)
        else:
            frame_processed = frame.copy()
            
        # Check image quality
        quality_ok = True
        quality_issues = []
        if self.quality_check_enabled:
            quality_ok, quality_reasons = self.preprocessor.check_image_quality(frame_processed)
            quality_issues = quality_reasons['issues']
            
        # Detect biji kopi
        detected_beans = self.preprocessor.detect_coffee_beans(frame_processed)
        
        # Apply NMS untuk menghapus deteksi duplikat
        detected_beans = self.preprocessor.apply_nms(detected_beans, iou_threshold=0.3)
        
        t_preproc = (time.time() - t_start) * 1000.0  # ms
        
        t_inf_start = time.time()
        results = []
        annotated_frame = frame_processed.copy()
        active_keys = set()
        
        # Batch inference if there are beans and TTA is disabled
        if len(detected_beans) > 0 and not use_tta:
            rois = []
            valid_beans = []
            for bean in detected_beans:
                roi = self.preprocessor.extract_bean_roi(frame_processed, bean['bbox'])
                if roi is not None:
                    normalized_roi = self.preprocessor.normalize_image(roi)
                    rois.append(normalized_roi)
                    valid_beans.append((bean, roi, normalized_roi))
            
            if len(rois) > 0:
                rois_array = np.array(rois)
                batch_predictions = self.model.predict_batch(rois_array)
                
                for (bean, roi, normalized_roi), (grade, confidence, probs) in zip(valid_beans, batch_predictions):
                    bbox = bean['bbox']
                    contour = bean['contour']
                    
                    # Apply temporal smoothing
                    smoothed_probs = self.smoother.smooth(bbox, probs)
                    predicted_class = np.argmax(smoothed_probs)
                    smoothed_confidence = float(smoothed_probs[predicted_class])
                    
                    # Track active keys
                    smoother_key = self.smoother._get_key(bbox)
                    active_keys.add(smoother_key)
                    
                    # Apply confidence threshold
                    if smoothed_confidence < self.confidence_threshold:
                        final_grade = 'Uncertain'
                    else:
                        final_grade = self.model.grade_labels[predicted_class]
                    
                    # Store result
                    results.append({
                        'bbox': bbox,
                        'grade': final_grade,
                        'confidence': smoothed_confidence,
                        'raw_confidence': confidence,
                        'probabilities': smoothed_probs,
                        'raw_probabilities': probs
                    })
                    
                    # Apply Grad-CAM overlay if enabled and toggled
                    if self.show_gradcam and self.gradcam_enabled:
                        try:
                            heatmap, _ = self.model.get_gradcam_heatmap(normalized_roi, pred_index=predicted_class)
                            gradcam_roi = self.model.overlay_gradcam(roi, heatmap, alpha=0.4)
                            x, y, w, h = bbox
                            gradcam_roi_resized = cv2.resize(gradcam_roi, (w, h))
                            annotated_frame[y:y+h, x:x+w] = gradcam_roi_resized
                        except Exception as e:
                            pass
                    
                    # Draw detection
                    self._draw_detection(annotated_frame, bbox, contour,
                                        final_grade, smoothed_confidence, smoothed_probs)
        else:
            # Fallback to sequential inference if TTA is enabled (TTA logic is not batched yet)
            for bean in detected_beans:
                bbox = bean['bbox']
                contour = bean['contour']
                
                # Extract ROI
                roi = self.preprocessor.extract_bean_roi(frame_processed, bbox)
                
                if roi is not None:
                    # Normalize untuk model
                    normalized_roi = self.preprocessor.normalize_image(roi)
                    
                    # Predict grade
                    if use_tta:
                        grade, confidence, probs = self.model.predict_with_tta(normalized_roi)
                    else:
                        grade, confidence, probs = self.model.predict(normalized_roi)
                    
                    # Apply temporal smoothing
                    smoothed_probs = self.smoother.smooth(bbox, probs)
                    predicted_class = np.argmax(smoothed_probs)
                    smoothed_confidence = float(smoothed_probs[predicted_class])
                    
                    # Track active keys
                    smoother_key = self.smoother._get_key(bbox)
                    active_keys.add(smoother_key)
                    
                    # Apply confidence threshold
                    if smoothed_confidence < self.confidence_threshold:
                        final_grade = 'Uncertain'
                    else:
                        final_grade = self.model.grade_labels[predicted_class]
                    
                    # Store result
                    results.append({
                        'bbox': bbox,
                        'grade': final_grade,
                        'confidence': smoothed_confidence,
                        'raw_confidence': confidence,
                        'probabilities': smoothed_probs,
                        'raw_probabilities': probs
                    })
                    
                    # Apply Grad-CAM overlay if enabled and toggled
                    if self.show_gradcam and self.gradcam_enabled:
                        try:
                            heatmap, _ = self.model.get_gradcam_heatmap(normalized_roi, pred_index=predicted_class)
                            gradcam_roi = self.model.overlay_gradcam(roi, heatmap, alpha=0.4)
                            x, y, w, h = bbox
                            gradcam_roi_resized = cv2.resize(gradcam_roi, (w, h))
                            annotated_frame[y:y+h, x:x+w] = gradcam_roi_resized
                        except Exception as e:
                            pass
                    
                    # Draw detection
                    self._draw_detection(annotated_frame, bbox, contour,
                                        final_grade, smoothed_confidence, smoothed_probs)
        
        t_inf = (time.time() - t_inf_start) * 1000.0  # ms
        
        # Cleanup old tracking entries
        self.smoother.cleanup(active_keys)
        
        # Draw Quality Warning banner if issues found
        if not quality_ok and len(quality_issues) > 0:
            h, w = annotated_frame.shape[:2]
            warning_text = f"WARNING KUALITAS: {', '.join(quality_issues)}"
            # Semi-transparent overlay for banner
            banner_overlay = annotated_frame.copy()
            cv2.rectangle(banner_overlay, (10, h - 85), (w - 10, h - 45), (0, 0, 200), -1)
            annotated_frame = cv2.addWeighted(annotated_frame, 0.7, banner_overlay, 0.3, 0)
            cv2.putText(annotated_frame, warning_text, (25, h - 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
        t_total = (time.time() - t_start) * 1000.0  # ms
        
        self.last_latency = {
            'preproc': t_preproc,
            'inference': t_inf,
            'total': t_total
        }
        
        return annotated_frame, results
    
    def _draw_detection(self, frame, bbox, contour, grade, confidence, probs):
        """
        Draw detection visualization pada frame
        
        Args:
            frame: Frame to draw on
            bbox: Bounding box (x, y, w, h)
            contour: Bean contour
            grade: Predicted grade
            confidence: Confidence score
            probs: Probability array
        """
        x, y, w, h = bbox
        color = self.grade_colors.get(grade, (255, 255, 255))
        
        # Draw contour outline (lebih natural daripada rectangle)
        cv2.drawContours(frame, [contour], -1, color, 2)
        
        # Draw bounding box (semi-transparent)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Draw label background
        label = f"{grade}"
        conf_text = f"{confidence:.0%}"
        
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        conf_size, _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        
        total_width = max(label_size[0], conf_size[0]) + 10
        total_height = label_size[1] + conf_size[1] + 15
        
        # Background rectangle
        bg_y1 = max(0, y - total_height - 5)
        bg_y2 = y - 2
        cv2.rectangle(frame, (x, bg_y1), (x + total_width, bg_y2), color, -1)
        
        # Grade label
        cv2.putText(frame, label, (x + 3, bg_y1 + label_size[1] + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Confidence text
        cv2.putText(frame, conf_text, (x + 3, bg_y2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Draw mini probability bar
        bar_y = y + h + 5
        bar_height = 8
        bar_width = w
        
        if bar_y + bar_height < frame.shape[0]:
            # Build color list dynamically from grade_colors
            grade_labels_for_bar = [l for l in self.grade_colors if l != 'Uncertain']
            current_x = x
            
            for i, prob in enumerate(probs):
                if i < len(grade_labels_for_bar):
                    col = self.grade_colors[grade_labels_for_bar[i]]
                else:
                    col = (200, 200, 200)
                pw = int(prob * bar_width)
                if pw > 0:
                    cv2.rectangle(frame, (current_x, bar_y),
                                (current_x + pw, bar_y + bar_height), col, -1)
                    current_x += pw
    
    def update_statistics(self, results):
        """
        Update statistics berdasarkan hasil grading
        
        Args:
            results: List of grading results
        """
        for result in results:
            grade = result['grade']
            if grade in self.stats:
                self.stats[grade] += 1
                self.stats['total'] += 1
    
    def draw_overlay(self, frame):
        """
        Draw comprehensive overlay: statistics, FPS, info
        
        Args:
            frame: Input frame
            
        Returns:
            Frame dengan overlay
        """
        h, w = frame.shape[:2]
        
        # === Statistics Panel (kiri atas) ===
        # Calculate panel height dynamically based on number of classes
        grade_labels = [l for l in self.stats if l not in ('Uncertain', 'total')]
        n_labels = len(grade_labels)
        panel_h = 70 + n_labels * 25 + 50  # title + rows + total/uncertain
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (380, panel_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Title
        y_offset = 35
        cv2.putText(frame, "STATISTIK GRADING", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Grade stats — dynamic iteration
        y_offset += 30
        total_graded = self.stats['total'] - self.stats.get('Uncertain', 0)
        
        for grade in grade_labels:
            count = self.stats[grade]
            percentage = (count / total_graded * 100) if total_graded > 0 else 0
            color = self.grade_colors.get(grade, (255, 255, 255))
            desc = self.grade_descriptions.get(grade, '')
            
            text = f"{grade}: {count} ({percentage:.1f}%)"
            cv2.putText(frame, text, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            
            # Description (only if panel is wide enough)
            if len(desc) > 0:
                cv2.putText(frame, f"  {desc}", (230, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
            y_offset += 25
        
        # Total
        cv2.putText(frame, f"Total: {self.stats['total']}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y_offset += 20
        
        # Uncertain count
        uncertain = self.stats.get('Uncertain', 0)
        if uncertain > 0:
            cv2.putText(frame, f"Uncertain: {uncertain}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        
        # === Right Status & Performance Panel ===
        panel_w = 210
        panel_x1 = w - panel_w - 10
        panel_x2 = w - 10
        panel_h = 240
        
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (panel_x1, 10), (panel_x2, panel_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay2, 0.3, 0)
        
        # Draw FPS
        fps = self.fps_counter.get_fps()
        fps_text = f"FPS: {fps:.1f}"
        fps_color = (0, 255, 0) if fps >= 20 else (0, 200, 255) if fps >= 10 else (0, 0, 255)
        cv2.putText(frame, fps_text, (panel_x1 + 15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
                    
        # Draw Latency
        lat_text = f"Latency: {self.last_latency['total']:.1f} ms"
        cv2.putText(frame, lat_text, (panel_x1 + 15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        lat_details = f"CV: {self.last_latency['preproc']:.1f}ms | DL: {self.last_latency['inference']:.1f}ms"
        cv2.putText(frame, lat_details, (panel_x1 + 15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
                    
        # Separator line
        cv2.line(frame, (panel_x1 + 10, 95), (panel_x2 - 10, 95), (100, 100, 100), 1)
        
        # System Settings Status
        status_y = 115
        status_font_scale = 0.38
        
        thresh_text = f"Conf Thresh: {self.confidence_threshold:.0%}"
        cv2.putText(frame, thresh_text, (panel_x1 + 15, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, status_font_scale, (200, 200, 200), 1)
                    
        # AWB status
        status_y += 20
        awb_text = f"AWB: {'ON' if self.awb_enabled else 'OFF'}"
        awb_color = (0, 255, 0) if self.awb_enabled else (150, 150, 150)
        cv2.putText(frame, awb_text, (panel_x1 + 15, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, status_font_scale, awb_color, 1)
        
        # Quality check status
        status_y += 20
        qc_text = f"QC: {'ON' if self.quality_check_enabled else 'OFF'}"
        qc_color = (0, 255, 0) if self.quality_check_enabled else (150, 150, 150)
        cv2.putText(frame, qc_text, (panel_x1 + 15, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, status_font_scale, qc_color, 1)
                    
        # Grad-CAM status
        status_y += 20
        gc_text = f"Grad-CAM: {'ON' if self.show_gradcam else 'OFF'}"
        gc_color = (0, 255, 0) if self.show_gradcam else (150, 150, 150)
        cv2.putText(frame, gc_text, (panel_x1 + 15, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, status_font_scale, gc_color, 1)
                    
        # GPU Acceleration check (Direct from TensorFlow device list)
        status_y += 20
        try:
            gpus = tf.config.list_physical_devices('GPU')
            gpu_active = len(gpus) > 0
        except Exception:
            gpu_active = False
        gpu_text = f"GPU Accel: {'ACTIVE' if gpu_active else 'N/A'}"
        gpu_color = (0, 255, 255) if gpu_active else (150, 150, 150)
        cv2.putText(frame, gpu_text, (panel_x1 + 15, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, status_font_scale, gpu_color, 1)
        
        # === Instructions (bawah) ===
        overlay3 = frame.copy()
        cv2.rectangle(overlay3, (10, h - 40), (w - 10, h - 5), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay3, 0.3, 0)
        
        instructions = "q: Quit | r: Reset | s: Screenshot | +/-: Conf | t: TTA | g: Grad-CAM | w: AWB | c: QC"
        cv2.putText(frame, instructions, (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        
        return frame
    
    def run_realtime(self):
        """
        Run sistem grading secara realtime
        """
        print("Memulai sistem grading kopi realtime...")
        print("=" * 50)
        print("Controls:")
        print("  q     : Quit")
        print("  r     : Reset statistics")
        print("  s     : Screenshot")
        print("  +/-   : Adjust confidence threshold")
        print("  t     : Toggle TTA (Test-Time Augmentation)")
        print("  g     : Toggle Grad-CAM overlays")
        print("  w     : Toggle Auto White Balance (AWB)")
        print("  c     : Toggle Image Quality Checker (QC)")
        print("=" * 50)
        
        self.start()
        use_tta = False
        
        try:
            while True:
                # FPS tracking
                self.fps_counter.tick()
                
                # Read frame
                ret, frame = self.camera.read_frame()
                
                if not ret:
                    print("Gagal membaca frame dari camera")
                    break
                
                # Process frame
                annotated_frame, results = self.process_frame(frame, use_tta=use_tta)
                
                # Update statistics
                self.update_statistics(results)
                
                # Draw overlay (stats, FPS, instructions)
                display_frame = self.draw_overlay(annotated_frame)
                
                # TTA indicator
                if use_tta:
                    cv2.putText(display_frame, "TTA: ON", 
                               (display_frame.shape[1] - 100, display_frame.shape[0] - 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Show frame
                cv2.imshow('Coffee Grading System', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nMenghentikan sistem...")
                    break
                elif key == ord('r'):
                    print("Reset statistics")
                    grade_labels_r = [l for l in self.stats if l not in ('Uncertain', 'total')]
                    self.stats = {label: 0 for label in grade_labels_r}
                    self.stats['Uncertain'] = 0
                    self.stats['total'] = 0
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"Screenshot saved: {filename}")
                elif key == ord('+') or key == ord('='):
                    self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
                    print(f"Confidence threshold: {self.confidence_threshold:.0%}")
                elif key == ord('-'):
                    self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
                    print(f"Confidence threshold: {self.confidence_threshold:.0%}")
                elif key == ord('t'):
                    use_tta = not use_tta
                    print(f"TTA: {'ON' if use_tta else 'OFF'}")
                elif key == ord('g'):
                    self.show_gradcam = not self.show_gradcam
                    print(f"Grad-CAM: {'ON' if self.show_gradcam else 'OFF'}")
                elif key == ord('w'):
                    self.awb_enabled = not self.awb_enabled
                    print(f"Auto White Balance: {'ON' if self.awb_enabled else 'OFF'}")
                elif key == ord('c'):
                    self.quality_check_enabled = not self.quality_check_enabled
                    print(f"Quality Checker: {'ON' if self.quality_check_enabled else 'OFF'}")
        
        finally:
            self.stop()
            cv2.destroyAllWindows()
            
            # Print final statistics
            print("\n" + "=" * 50)
            print("STATISTIK AKHIR")
            print("=" * 50)
            
            total_graded = self.stats['total'] - self.stats.get('Uncertain', 0)
            
            grade_labels_final = [l for l in self.stats if l not in ('Uncertain', 'total')]
            for grade in grade_labels_final:
                count = self.stats[grade]
                percentage = (count / total_graded * 100) if total_graded > 0 else 0
                desc = self.grade_descriptions.get(grade, '')
                print(f"  {grade}: {count:>5} ({percentage:>5.1f}%) - {desc}")
            
            print(f"  {'─' * 45}")
            print(f"  Total Graded: {total_graded}")
            
            uncertain = self.stats.get('Uncertain', 0)
            if uncertain > 0:
                print(f"  Uncertain:    {uncertain} (filtered out)")
            
            print(f"  Total Frames: {self.stats['total']}")
