"""
Demo CLI end-to-end Smart Grading (tanpa browser).

Menjalankan seluruh pipeline pada gambar sampel lot, mencetak hasil mutu SNI,
menyimpan gambar anotasi, dan mencatat sesi ke basis data analitik.

Contoh:
    python poc_demo.py                              # semua sampel demo
    python poc_demo.py --image foto_kopi.jpg        # gambar sendiri
    python poc_demo.py --image foto.jpg --weight 300  # berat sampel diketahui

Setelah itu buka dashboard:  python webapp/app.py
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTHONUTF8", "1")

import sys
import argparse
import cv2

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "webapp"))

BAR = "=" * 64


def _print_result(name, res):
    g = res["grade"]
    print(f"\n{BAR}\nSAMPEL: {name}\n{BAR}")
    print(f"  Model klasifikasi : {res['model_status']}")
    print(f"  Biji terdeteksi   : {g['total_beans']} (cacat {g['defect_beans']}, "
          f"{g['defect_rate_pct']}%)")
    counts = {k: v for k, v in res["counts"].items() if v > 0}
    print(f"  Komposisi         : {counts}")
    if g["defect_value_breakdown"]:
        print(f"  Rincian nilai cacat: {g['defect_value_breakdown']}")
    print(f"  Nilai cacat / 300g: {g['defect_value_per_300g']}")
    print(f"  >> MUTU           : {g['grade_code']} ({g['grade_label']})  "
          f"{'[LAYAK PREMIUM]' if g['export_premium_eligible'] else '[KOMERSIAL/RE-SORTASI]'}")
    print(f"  Rekomendasi       : {g['recommendation']}")


def main():
    ap = argparse.ArgumentParser(description="Demo CLI Smart Grading Kopi")
    ap.add_argument("--image", help="Path gambar (default: pakai sampel lot demo)")
    ap.add_argument("--weight", type=float, default=None, help="Berat sampel (gram), opsional")
    ap.add_argument("--out", default="results", help="Folder output anotasi")
    ap.add_argument("--no-record", action="store_true", help="Jangan catat ke basis data")
    args = ap.parse_args()

    from inference_engine import SmartGradingEngine
    import make_samples
    from analytics import GradingStore

    os.makedirs(args.out, exist_ok=True)
    engine = SmartGradingEngine()
    store = None if args.no_record else GradingStore()

    # Tentukan daftar gambar yang diproses.
    if args.image:
        targets = [(os.path.basename(args.image), args.image)]
    else:
        sdir = os.path.join(_ROOT, "webapp", "sample_images")
        if not os.path.isdir(sdir) or not os.listdir(sdir):
            make_samples.generate_all(sdir)
        targets = [(f, os.path.join(sdir, f)) for f in sorted(os.listdir(sdir))
                   if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    print(f"\n{BAR}\n SMART GRADING KOPI — DEMO CLI (SNI 01-2907-2008)\n{BAR}")
    for name, path in targets:
        img = cv2.imread(path)
        if img is None:
            print(f"  ! Gagal membaca {path}")
            continue
        res = engine.analyze_image(img, draw=True, sample_weight_g=args.weight)
        _print_result(name, res)

        # Simpan anotasi (decode dari base64).
        if res["annotated_image_b64"]:
            import base64
            import numpy as np
            b = base64.b64decode(res["annotated_image_b64"].split(",", 1)[1])
            arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            out_path = os.path.join(args.out, f"graded_{name}")
            cv2.imwrite(out_path, arr)
            print(f"  Anotasi disimpan  : {out_path}")

        if store is not None:
            store.add_session(res["grade"], source=name, line="CLI", operator="demo")

    print(f"\n{BAR}\nSelesai. Buka dashboard analitik:  python webapp/app.py\n{BAR}")


if __name__ == "__main__":
    main()
