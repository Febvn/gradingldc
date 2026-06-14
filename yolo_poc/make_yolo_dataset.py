"""
Generator dataset OBJECT DETECTION (format YOLO) untuk PoC grading kopi.

Karena posisi tiap biji digambar sendiri, kita tahu PERSIS kotak (bbox) dan
kelasnya — sehingga menghasilkan label YOLO ground-truth yang akurat tanpa
anotasi manual. Dua varian:

  - clean    : latar polos terang, biji terpisah, tanpa noise  -> "terlalu mudah"
               (analog data internet/Roboflow yang rapi -> metrik menggelembung)
  - hardened : latar bertekstur/acak, pencahayaan & bayangan bervariasi,
               oklusi (biji bertumpuk), blur/noise, gambar negatif (tanpa objek)
               -> lebih mendekati kondisi lapangan -> metrik lebih jujur

6 kelas (selaras Config.GRADE_LABELS):
  0 Normal  1 Biji Hitam  2 Biji Cokelat  3 Berlubang  4 Pecah  5 Berjamur
"""
import os
import shutil
import numpy as np
import cv2

CLASSES = ["Normal", "Biji Hitam", "Biji Cokelat", "Berlubang", "Pecah", "Berjamur"]
CLASS_COLORS = {  # BGR, selaras generator sintetis tim
    "Normal": (35, 75, 140), "Biji Hitam": (25, 25, 30), "Biji Cokelat": (20, 50, 95),
    "Berlubang": (35, 75, 140), "Pecah": (38, 78, 142), "Berjamur": (65, 85, 95),
}
# Distribusi kemunculan kelas (Normal dominan, seperti lot nyata).
CLASS_WEIGHTS = np.array([0.45, 0.12, 0.13, 0.12, 0.13, 0.05])
IMG = 512


def _ellipse_bbox(cx, cy, a, b, angle_deg):
    """Bounding box axis-aligned dari elips terrotasi."""
    t = np.radians(angle_deg)
    hw = np.sqrt((a * np.cos(t)) ** 2 + (b * np.sin(t)) ** 2)
    hh = np.sqrt((a * np.sin(t)) ** 2 + (b * np.cos(t)) ** 2)
    return cx - hw, cy - hh, cx + hw, cy + hh


def _draw_bean(img, cx, cy, label, rng, scale=1.0):
    if label == "Pecah":
        a, b = int(rng.integers(15, 21) * scale), int(rng.integers(10, 13) * scale)
    else:
        a, b = int(rng.integers(22, 30) * scale), int(rng.integers(15, 19) * scale)
    angle = int(rng.integers(0, 180))
    base = CLASS_COLORS[label]
    color = tuple(int(np.clip(c + rng.integers(-12, 12), 0, 255)) for c in base)

    cv2.ellipse(img, (cx, cy), (a, b), angle, 0, 360, color, -1)
    rad = np.radians(angle)
    dx, dy = int(a * np.cos(rad) * 0.8), int(a * np.sin(rad) * 0.8)
    groove = tuple(max(0, c - 35) for c in color)
    cv2.line(img, (cx - dx, cy - dy), (cx + dx, cy + dy), groove, max(1, int(scale)))

    if label == "Berlubang":
        for _ in range(int(rng.integers(1, 3))):
            cv2.circle(img, (cx + int(rng.integers(-a//2, a//2)), cy + int(rng.integers(-b//2, b//2))),
                       max(2, int(3 * scale)), (10, 10, 10), -1)
    elif label == "Pecah":
        pts = np.array([[cx - a, cy - b], [cx + a, cy - b], [cx - 2, cy + 2]], dtype=np.int32)
        cv2.fillPoly(img, [pts], (225, 225, 225))
    elif label == "Berjamur":
        for _ in range(2):
            ov = img.copy()
            cv2.circle(ov, (cx + int(rng.integers(-a//2, a//2)), cy + int(rng.integers(-b//2, b//2))),
                       int(rng.integers(5, 9)), (210, 230, 230), -1)
            cv2.addWeighted(ov, 0.45, img, 0.55, 0, dst=img)

    x1, y1, x2, y2 = _ellipse_bbox(cx, cy, a, b, angle)
    return x1, y1, x2, y2


def _background(rng, hard):
    if not hard:
        img = np.ones((IMG, IMG, 3), np.uint8) * 228
        return np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    # Hardened: warna dasar acak + tekstur + gradien pencahayaan.
    base = rng.integers(80, 210)
    img = np.ones((IMG, IMG, 3), np.uint8) * base
    img = np.clip(img + rng.normal(0, rng.integers(6, 22), img.shape), 0, 255).astype(np.uint8)
    # Bercak/tekstur latar.
    for _ in range(rng.integers(6, 18)):
        c = tuple(int(x) for x in rng.integers(60, 230, 3))
        cv2.circle(img, (int(rng.integers(0, IMG)), int(rng.integers(0, IMG))),
                   int(rng.integers(10, 60)), c, -1)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(2, 6))
    # Gradien pencahayaan (vignette tidak merata).
    gx = np.tile(np.linspace(rng.uniform(0.6, 1.0), rng.uniform(0.6, 1.0), IMG), (IMG, 1))
    img = np.clip(img * gx[..., None], 0, 255).astype(np.uint8)
    return img


def _gen_image(rng, hard):
    img = _background(rng, hard)
    labels = []
    # Gambar negatif (tanpa objek) hanya pada mode hardened.
    if hard and rng.random() < 0.10:
        return img, labels

    n = int(rng.integers(8, 20))
    placed = []  # (cx,cy,r) untuk kontrol oklusi
    attempts = 0
    while len(placed) < n and attempts < n * 6:
        attempts += 1
        scale = rng.uniform(0.8, 1.4) if hard else 1.0
        r = int(34 * scale)
        cx, cy = int(rng.integers(r, IMG - r)), int(rng.integers(r, IMG - r))
        min_gap = r * (0.55 if hard else 1.05)  # hardened mengizinkan oklusi
        if any((cx - px) ** 2 + (cy - py) ** 2 < (min_gap + pr * 0.55) ** 2 for px, py, pr in placed):
            continue
        ci = int(rng.choice(len(CLASSES), p=CLASS_WEIGHTS))
        x1, y1, x2, y2 = _draw_bean(img, cx, cy, CLASSES[ci], rng, scale)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(IMG - 1, x2), min(IMG - 1, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        cxn, cyn = (x1 + x2) / 2 / IMG, (y1 + y2) / 2 / IMG
        wn, hn = (x2 - x1) / IMG, (y2 - y1) / IMG
        labels.append((ci, cxn, cyn, wn, hn))
        placed.append((cx, cy, r))

    if hard:
        if rng.random() < 0.5:
            k = int(rng.choice([3, 5, 7]))
            img = cv2.GaussianBlur(img, (k, k), 0)
        img = np.clip(img + rng.normal(0, rng.integers(3, 12), img.shape), 0, 255).astype(np.uint8)
    return img, labels


def generate(out_dir, n_train=100, n_val=25, hard=False, seed=0):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    rng = np.random.default_rng(seed)
    for split, count in [("train", n_train), ("val", n_val)]:
        for i in range(count):
            img, labels = _gen_image(rng, hard)
            stem = f"{split}_{i:04d}"
            cv2.imwrite(os.path.join(out_dir, "images", split, stem + ".jpg"), img)
            with open(os.path.join(out_dir, "labels", split, stem + ".txt"), "w") as f:
                for ci, cx, cy, w, h in labels:
                    f.write(f"{ci} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\n")
        f.write("train: images/train\nval: images/val\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write("names: [" + ", ".join(f"'{c}'" for c in CLASSES) + "]\n")
    return yaml_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=100)
    ap.add_argument("--val", type=int, default=25)
    ap.add_argument("--hard", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    y = generate(a.out, a.train, a.val, a.hard, a.seed)
    print("data.yaml:", y, "| hard=", a.hard)
