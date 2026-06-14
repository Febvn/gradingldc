"""
Pelatihan YOLOv8n untuk PoC object-detection grading kopi (CPU).

Contoh:
    python yolo_poc/train_yolo.py --data yolo_poc/dataset_clean/data.yaml --name clean --epochs 20
"""
import os
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("YOLO_VERBOSE", "True")

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="yolov8n.pt")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    results = model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        device="cpu", workers=2, seed=0, deterministic=True,
        project="yolo_poc/runs", name=a.name, exist_ok=True, plots=True, verbose=True,
    )
    # Ringkas metrik kunci.
    try:
        m = model.val(data=a.data, device="cpu", project="yolo_poc/runs",
                      name=a.name + "_val", exist_ok=True)
        print("\n==== METRIK VALIDASI ====")
        print(f"mAP50    : {m.box.map50:.4f}")
        print(f"mAP50-95 : {m.box.map:.4f}")
        print(f"precision: {m.box.mp:.4f}")
        print(f"recall   : {m.box.mr:.4f}")
    except Exception as e:
        print("val ringkas dilewati:", e)


if __name__ == "__main__":
    main()
