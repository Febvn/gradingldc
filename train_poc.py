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
# WAJIB sebelum import tensorflow: pakai Keras 2 (tf-keras) agar kode model tim
# (ImageDataGenerator, optimizer.lr, dsb.) kompatibel di TF 2.20 / Python 3.13.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import Config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="PoC trainer (EfficientNet, sintetis)")
    parser.add_argument("--per-class", type=int, default=50,
                        help="Jumlah gambar sintetis per kelas")
    parser.add_argument("--p1-epochs", type=int, default=12, help="Epoch fase 1 (head)")
    parser.add_argument("--p2-epochs", type=int, default=5, help="Epoch fase 2 (fine-tune)")
    args = parser.parse_args()

    # --- Override Config agar pelatihan PoC cepat & deterministik ---
    Config.MIXUP_ENABLED = False          # mixup memperlambat di CPU
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
    print("PoC TRAINING — EfficientNetB0 (dataset sintetis)")
    print(f"  kelas={num_classes}  per_class={args.per_class}  "
          f"p1={args.p1_epochs}  p2={args.p2_epochs}")
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
        epochs=args.p1_epochs, batch_size=16,
        class_weights=class_weights, data_augmentation=aug,
    )

    # 4. Fase 2: fine-tune sebagian layer atas (opsional, pendek).
    if args.p2_epochs > 0:
        model.fine_tune(
            X_train, y_train, X_val, y_val,
            epochs=args.p2_epochs, batch_size=16,
            n_layers_unfreeze=Config.TRAIN_P2_UNFREEZE_LAYERS,
            class_weights=class_weights, data_augmentation=aug,
        )

    # 5. Simpan & evaluasi.
    model.save_model(Config.MODEL_SAVE_PATH)
    try:
        evaluate_model(model, X_val, y_val, save_dir="models")
    except Exception as e:
        print(f"(evaluasi dilewati: {e})")

    print("\nSELESAI. Model tersimpan di", Config.MODEL_SAVE_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
