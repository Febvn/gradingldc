"""
Pelatihan model EfficientNet (milik tim) untuk PROOF OF CONCEPT.

Tujuan: menghasilkan artefak model `models/coffee_grading_model.h5` yang
benar-benar terlatih sehingga PoC Smart Grading dapat dijalankan end-to-end
TANPA dataset asli (memakai dataset sintetis generator tim) dan TANPA GPU.

Skrip ini MEMAKAI ULANG arsitektur & pipeline milik Febrian:
    - CoffeeGradingModel (EfficientNetB0 transfer learning)  -> src/model.py
    - generate_synthetic_dataset()                            -> analyze_dataset.py
    - load_dataset(), create_data_augmentation()              -> train_model.py

Khusus PoC, epoch dipersingkat & mixup dimatikan agar cepat di CPU.
Untuk produksi (data lapangan asli) gunakan `python train_model.py` penuh.

Jalankan:
    python train_poc.py                  (default: 50 sampel sintetis/kelas)
    python train_poc.py --per-class 80   (lebih banyak data sintetis)
"""
import os
import sys

# Pastikan working directory selalu di folder script ini
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import sys
import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('h5py').setLevel(logging.ERROR)

try:
    import absl.logging as _absl_log
    _absl_log._warn_preinit_stderr = 0
    _absl_log.set_verbosity(_absl_log.ERROR)
except Exception:
    pass

import argparse


def _directml_is_functional():
    """
    Test whether tensorflow-directml-plugin is usable (no duplicate kernels).
    Version 0.4.0.dev registers conflicting GPU OpKernels for AssignVariableOp,
    Fill, TruncatedNormal, etc., making tf.Variable creation fail entirely.
    Returns True only if a test variable can be created without errors.
    """
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    try:
        v = tf.Variable([1.0])
        del v
        return True
    except tf.errors.InvalidArgumentError as e:
        if 'Multiple OpKernel registrations' in str(e):
            return False
        raise


def setup_hardware():
    """
    Auto-detect dan konfigurasi hardware terbaik yang tersedia.
    Prioritas: NVIDIA (CUDA) > AMD/Intel (DirectML, jika functional) > CPU
    Mengembalikan: ('gpu'|'directml'|'cpu', batch_size_rekomendasi)
    """
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)

    # --- Coba NVIDIA / CUDA GPU ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

        # Verify GPU is actually usable (DirectML plugin may register conflicts)
        if _directml_is_functional():
            names = [g.name for g in gpus]
            print(f"[GPU] {len(gpus)} GPU ditemukan: {names}")
            print("  -> Training akan menggunakan GPU")
            return 'gpu', 32
        else:
            print("[WARN] GPU terdeteksi tapi tensorflow-directml-plugin tidak kompatibel.")
            print("  -> tensorflow-directml-plugin 0.4.0.dev memiliki bug: duplicate GPU")
            print("     OpKernel registrations untuk AssignVariableOp/Fill/TruncatedNormal.")
            print("  -> Untuk GPU AMD, gunakan PyTorch + torch-directml (bukan TF).")
            print("  -> Melanjutkan training di CPU...")

    # --- Coba AMD / Intel via DirectML (Windows) ---
    try:
        import tensorflow_directml_plugin  # noqa: F401
        dml_gpus = tf.config.list_physical_devices('GPU')
        if dml_gpus and _directml_is_functional():
            print(f"[GPU AMD/DirectML] {len(dml_gpus)} GPU ditemukan")
            print("  -> Training akan menggunakan DirectML GPU")
            return 'directml', 32
    except ImportError:
        pass

    # --- Deteksi GPU AMD/NVIDIA tanpa driver CUDA terpasang ---
    try:
        import subprocess
        result = subprocess.run(
            ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
            capture_output=True, text=True, timeout=5
        )
        gpu_names = [l.strip() for l in result.stdout.splitlines() if l.strip() and 'Name' not in l]
        if gpu_names:
            has_nvidia = any('NVIDIA' in n or 'GeForce' in n or 'RTX' in n or 'GTX' in n for n in gpu_names)
            print(f"[INFO] GPU terdeteksi: {', '.join(gpu_names)}")
            if has_nvidia:
                print("  -> GPU NVIDIA terdeteksi tapi CUDA belum aktif.")
                print("     Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
    except Exception:
        pass

    # --- Fallback CPU ---
    print("[CPU] Training menggunakan CPU.")
    return 'cpu', 16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import Config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="PoC trainer (EfficientNet, sintetis)")
    parser.add_argument("--per-class", type=int, default=50,
                        help="Jumlah gambar sintetis per kelas")
    parser.add_argument("--p1-epochs", type=int, default=12, help="Epoch fase 1 (head)")
    parser.add_argument("--p2-epochs", type=int, default=5, help="Epoch fase 2 (fine-tune)")
    args = parser.parse_args()

    # --- Deteksi hardware ---
    hw, batch_size = setup_hardware()
    Config.MIXUP_ENABLED = False
    Config.LR_SCHEDULE = "reduce_on_plateau"

    import numpy as np
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical

    from analyze_dataset import generate_synthetic_dataset
    from train_model import load_dataset, create_data_augmentation, evaluate_model, compute_class_weights
    from model import CoffeeGradingModel

    data_dir = Config.DATA_DIR
    num_classes = Config.MODEL_NUM_CLASSES

    print("=" * 60)
    print("PoC TRAINING - EfficientNetB3 (dataset real + sintetis)")
    print(f"  hardware={hw}  batch={batch_size}  kelas={num_classes}")
    print(f"  per_class={args.per_class}  p1={args.p1_epochs}  p2={args.p2_epochs}")
    print("=" * 60)

    # 1. Pastikan dataset sintetis tersedia.
    generate_synthetic_dataset(data_dir, num_samples_per_class=args.per_class)

    # 2. Muat dataset.
    X, y = load_dataset(data_dir)
    if len(X) == 0:
        print("ERROR: dataset kosong setelah generate sintetis.")
        return 1

    class_weights = compute_class_weights(y)
    y_cat = to_categorical(y, num_classes=num_classes)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train={len(X_train)}  Val={len(X_val)}")

    aug = create_data_augmentation()
    os.makedirs("models", exist_ok=True)
    model = CoffeeGradingModel()

    # 3. Fase 1: latih classifier head.
    model.train(
        X_train, y_train, X_val, y_val,
        epochs=args.p1_epochs, batch_size=batch_size,
        class_weights=class_weights, data_augmentation=aug,
    )

    # 4. Fase 2: fine-tune sebagian layer atas (opsional, pendek).
    if args.p2_epochs > 0:
        model.fine_tune(
            X_train, y_train, X_val, y_val,
            epochs=args.p2_epochs, batch_size=batch_size,
            n_layers_unfreeze=Config.TRAIN_P2_UNFREEZE_LAYERS,
            class_weights=class_weights, data_augmentation=aug,
        )

    # 5. Simpan & evaluasi.
    model.save_model(Config.MODEL_SAVE_PATH)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            evaluate_model(model, X_val, y_val, save_dir="models")
    except Exception as e:
        print(f"(evaluasi dilewati: {e})")

    print("\nSELESAI. Model tersimpan di", Config.MODEL_SAVE_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
