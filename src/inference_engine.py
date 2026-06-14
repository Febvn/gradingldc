"""
SmartGradingEngine — orkestrator Smart Grading end-to-end untuk PoC.

Alur (sesuai proposal "Smart Grading berbasis Computer Vision & AI"):

    Gambar  ->  [AWB + cek kualitas]  ->  [deteksi biji (kontur+NMS)]
            ->  [klasifikasi cacat per-biji: EfficientNet milik tim]
            ->  [agregasi jumlah per kelas]
            ->  [penilaian MUTU lot: SNI 01-2907-2008]  ->  hasil + anotasi

Menyatukan komponen yang sudah ada (ImagePreprocessor, CoffeeGradingModel)
dengan komponen baru (SNIGradeEngine). Dirancang agar tetap berjalan
walau model belum dilatih (mode "deteksi saja").
"""
import os
# Pakai Keras 2 (tf-keras) sebelum modul apa pun meng-import tensorflow.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import time
import base64
from typing import Optional, Dict, List

import cv2
import numpy as np

_SRC = os.path.dirname(__file__)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from preprocessing import ImagePreprocessor
from grading_standard import SNIGradeEngine
from screen_size_grader import ScreenSizeGrader
from config import Config

# Warna BGR per kelas (selaras dengan grading_system / batch_inference).
_COLOR_PALETTE_BGR = [
    (0, 200, 0),      # Normal       -> hijau
    (0, 0, 220),      # Biji Hitam   -> merah
    (0, 120, 220),    # Biji Cokelat -> oranye
    (220, 180, 0),    # Berlubang    -> cyan
    (180, 0, 180),    # Pecah        -> ungu
    (0, 180, 220),    # Berjamur     -> kuning
]


class SmartGradingEngine:
    def __init__(self,
                 model_path: Optional[str] = None,
                 confidence_threshold: Optional[float] = None,
                 awb_enabled: bool = True):
        self.preprocessor = ImagePreprocessor()
        self.sni = SNIGradeEngine()
        self.grade_labels = list(Config.GRADE_LABELS)
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None
            else Config.CONFIDENCE_THRESHOLD
        )
        self.awb_enabled = awb_enabled
        self.screen_grader = ScreenSizeGrader(pixels_per_mm=10.0)

        # Warna per label.
        self.colors = {
            label: _COLOR_PALETTE_BGR[i % len(_COLOR_PALETTE_BGR)]
            for i, label in enumerate(self.grade_labels)
        }
        self.colors["Uncertain"] = (128, 128, 128)

        # Muat model EfficientNet milik tim bila tersedia.
        self.model = None
        self.model_available = False
        path = model_path or Config.MODEL_SAVE_PATH
        if path and os.path.exists(path):
            try:
                from model import CoffeeGradingModel
                self.model = CoffeeGradingModel(path)
                self.model_available = True
            except Exception as e:  # pragma: no cover - defensif
                print(f"[SmartGradingEngine] Gagal memuat model: {e}")
                self.model = None

    # ------------------------------------------------------------------
    @property
    def model_status(self) -> str:
        return "EfficientNetB0 (terlatih)" if self.model_available else "BELUM DILATIH"

    # ------------------------------------------------------------------
    def analyze_image(self,
                      image_bgr: np.ndarray,
                      draw: bool = True,
                      sample_weight_g: Optional[float] = None) -> Dict:
        """Analisis satu gambar -> deteksi, klasifikasi, grade, anotasi."""
        t0 = time.time()
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Gambar kosong / tidak valid")

        frame = (self.preprocessor.apply_auto_white_balance(image_bgr)
                 if self.awb_enabled else image_bgr.copy())

        quality_ok, quality = self.preprocessor.check_image_quality(frame)

        beans = self.preprocessor.detect_coffee_beans(frame)
        beans = self.preprocessor.apply_nms(beans, iou_threshold=Config.NMS_IOU_THRESHOLD)
        t_detect = (time.time() - t0) * 1000.0

        # Klasifikasi per-biji (batch) ------------------------------------
        t1 = time.time()
        counts = {label: 0 for label in self.grade_labels}
        counts["Uncertain"] = 0
        bean_results: List[Dict] = []

        rois, valid = [], []
        for b in beans:
            roi = self.preprocessor.extract_bean_roi(frame, b["bbox"])
            if roi is not None:
                rois.append(self.preprocessor.normalize_image(roi))
                valid.append(b)

        if self.model_available and rois:
            preds = self.model.predict_batch(np.array(rois))
            for b, (grade, conf, probs) in zip(valid, preds):
                cls = int(np.argmax(probs))
                conf = float(probs[cls])
                if conf < self.confidence_threshold:
                    label = "Uncertain"
                else:
                    label = self.grade_labels[cls]
                counts[label] += 1
                
                _, width_mm = self.screen_grader.get_physical_dimensions(b["contour"])
                screen_size = self.screen_grader.determine_screen_size(width_mm)
                
                bean_results.append({
                    "bbox": [int(v) for v in b["bbox"]],
                    "label": label,
                    "confidence": round(conf, 4),
                    "screen_size": screen_size,
                })
        else:
            # Mode deteksi-saja (model belum ada): tandai semua "Uncertain".
            for b in valid:
                counts["Uncertain"] += 1
                
                _, width_mm = self.screen_grader.get_physical_dimensions(b["contour"])
                screen_size = self.screen_grader.determine_screen_size(width_mm)
                
                bean_results.append({
                    "bbox": [int(v) for v in b["bbox"]],
                    "label": "Uncertain",
                    "confidence": 0.0,
                    "screen_size": screen_size,
                })
        t_infer = (time.time() - t1) * 1000.0

        # Penilaian mutu SNI ----------------------------------------------
        grade_result = self.sni.grade_sample(
            {k: v for k, v in counts.items()},
            sample_weight_g=sample_weight_g,
        )

        annotated_b64 = None
        if draw:
            annotated = self._annotate(frame, valid, bean_results, grade_result, quality_ok)
            annotated_b64 = self._encode_png(annotated)

        h_img, w_img = frame.shape[:2]
        return {
            "model_status": self.model_status,
            "model_available": self.model_available,
            "image_size": [int(w_img), int(h_img)],
            "quality_ok": bool(quality_ok),
            "quality": {
                "brightness": round(quality["mean_brightness"], 1),
                "sharpness": round(quality["sharpness"], 1),
                "contrast": round(quality["contrast"], 1),
                "issues": quality["issues"],
            },
            "counts": counts,
            "beans": bean_results,
            "grade": grade_result.to_dict(),
            "timing_ms": {
                "detect": round(t_detect, 1),
                "classify": round(t_infer, 1),
                "total": round((time.time() - t0) * 1000.0, 1),
            },
            "annotated_image_b64": annotated_b64,
        }

    # ------------------------------------------------------------------
    def _annotate(self, frame, beans, bean_results, grade_result, quality_ok):
        out = frame.copy()
        for b, r in zip(beans, bean_results):
            x, y, w, h = b["bbox"]
            color = self.colors.get(r["label"], (255, 255, 255))
            cv2.drawContours(out, [b["contour"]], -1, color, 2)
            
            screen_size = r.get("screen_size", "")
            base_label = r["label"] if r["label"] == "Uncertain" else f'{r["label"]} {r["confidence"]:.0%}'
            tag = f"{base_label} ({screen_size})" if screen_size else base_label
            
            cv2.putText(out, tag, (x, max(12, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # Banner mutu di bagian atas.
        h_img, w_img = out.shape[:2]
        banner = out.copy()
        cv2.rectangle(banner, (0, 0), (w_img, 46), (20, 20, 20), -1)
        out = cv2.addWeighted(banner, 0.65, out, 0.35, 0)
        premium = grade_result.export_premium_eligible
        gcolor = (0, 220, 0) if premium else (0, 165, 255)
        text = (f"MUTU: {grade_result.grade_code}  |  Nilai cacat/300g: "
                f"{grade_result.defect_value_per_300g}  |  Biji: {grade_result.total_beans}  "
                f"|  Cacat: {grade_result.defect_rate_pct}%")
        cv2.putText(out, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, gcolor, 2, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _encode_png(image_bgr) -> str:
        ok, buf = cv2.imencode(".png", image_bgr)
        if not ok:
            return ""
        return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


if __name__ == "__main__":
    # Uji cepat memakai satu gambar sintetis.
    sys.path.insert(0, os.path.dirname(_SRC))
    from analyze_dataset import generate_synthetic_dataset
    tmp = os.path.join(os.path.dirname(_SRC), "data")
    generate_synthetic_dataset(tmp, num_samples_per_class=2)
    sample = os.path.join(tmp, "biji_hitam")
    f = [x for x in os.listdir(sample)][0]
    img = cv2.imread(os.path.join(sample, f))
    eng = SmartGradingEngine()
    res = eng.analyze_image(img)
    print("Model:", res["model_status"])
    print("Counts:", res["counts"])
    print("Grade:", res["grade"]["grade_code"], "| nilai/300g:", res["grade"]["defect_value_per_300g"])
