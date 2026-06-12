"""
Membuat gambar SAMPEL LOT multi-biji untuk demo grading (premium/komersial/rendah).

Setiap gambar berisi puluhan biji kopi tersebar di latar terang, dengan
komposisi cacat terkontrol. Resep menggambar biji DISELARASKAN dengan
generator dataset sintetis tim (analyze_dataset.generate_synthetic_dataset)
agar model EfficientNet yang dilatih pada biji sintetis tetap mengenalinya.
"""
import os
import numpy as np
import cv2

# Skema warna BGR per kelas (identik dengan generator tim).
CLASS_COLORS = {
    "Normal": (35, 75, 140),
    "Biji Hitam": (25, 25, 30),
    "Biji Cokelat": (20, 50, 95),
    "Berlubang": (35, 75, 140),
    "Pecah": (38, 78, 142),
    "Berjamur": (65, 85, 95),
}

# Komposisi lot demo (proporsi per kelas).
LOTS = {
    "lot_premium": {
        "title": "Lot Premium",
        "n": 64,
        "mix": {"Normal": 0.94, "Pecah": 0.03, "Biji Cokelat": 0.03},
    },
    "lot_komersial": {
        "title": "Lot Komersial",
        "n": 64,
        "mix": {"Normal": 0.80, "Pecah": 0.07, "Berlubang": 0.06,
                "Biji Cokelat": 0.04, "Biji Hitam": 0.03},
    },
    "lot_rendah": {
        "title": "Lot Mutu Rendah",
        "n": 64,
        "mix": {"Normal": 0.45, "Biji Hitam": 0.20, "Berjamur": 0.15,
                "Pecah": 0.12, "Berlubang": 0.08},
    },
}


def _draw_bean(img, cx, cy, label, rng):
    """Gambar satu biji kopi kecil sesuai kelas pada (cx, cy)."""
    if label == "Pecah":
        axes = (rng.integers(13, 18), rng.integers(8, 11))
    else:
        axes = (rng.integers(18, 23), rng.integers(12, 15))
    angle = int(rng.integers(0, 180))
    base = CLASS_COLORS.get(label, CLASS_COLORS["Normal"])
    color = tuple(int(np.clip(c + rng.integers(-10, 10), 0, 255)) for c in base)

    cv2.ellipse(img, (cx, cy), tuple(int(a) for a in axes), angle, 0, 360, color, -1)

    # Alur tengah biji.
    rad = np.radians(angle)
    dx = int(axes[0] * np.cos(rad) * 0.8)
    dy = int(axes[0] * np.sin(rad) * 0.8)
    groove = tuple(max(0, c - 35) for c in color)
    cv2.line(img, (cx - dx, cy - dy), (cx + dx, cy + dy), groove, 1)

    if label == "Berlubang":
        for _ in range(int(rng.integers(1, 3))):
            hx = cx + int(rng.integers(-8, 8))
            hy = cy + int(rng.integers(-6, 6))
            cv2.circle(img, (hx, hy), int(rng.integers(2, 3)), (10, 10, 10), -1)
    elif label == "Pecah":
        pts = np.array([[cx - 22, cy - 22], [cx + 22, cy - 22], [cx - 2, cy + 2]],
                       dtype=np.int32)
        cv2.fillPoly(img, [pts], (225, 225, 225))
    elif label == "Berjamur":
        for _ in range(2):
            px = cx + int(rng.integers(-9, 9))
            py = cy + int(rng.integers(-9, 9))
            overlay = img.copy()
            cv2.circle(overlay, (px, py), int(rng.integers(5, 8)), (210, 230, 230), -1)
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, dst=img)


def _labels_for_mix(n, mix, rng):
    labels = []
    for lab, frac in mix.items():
        labels += [lab] * int(round(n * frac))
    while len(labels) < n:
        labels.append("Normal")
    labels = labels[:n]
    rng.shuffle(labels)
    return labels


def make_lot_image(spec, seed=0, w=820, h=560):
    rng = np.random.default_rng(seed)
    img = np.ones((h, w, 3), dtype=np.uint8) * 228
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)

    labels = _labels_for_mix(spec["n"], spec["mix"], rng)
    # Tata biji pada grid jitter agar tidak bertumpukan.
    cols = 10
    rows = int(np.ceil(spec["n"] / cols))
    mx, my = 60, 60
    gx = (w - 2 * mx) / (cols - 1)
    gy = (h - 2 * my) / (rows - 1)
    i = 0
    for r in range(rows):
        for c in range(cols):
            if i >= len(labels):
                break
            cx = int(mx + c * gx + rng.integers(-12, 12))
            cy = int(my + r * gy + rng.integers(-12, 12))
            _draw_bean(img, cx, cy, labels[i], rng)
            i += 1
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, labels


def generate_all(out_dir, seed=7):
    os.makedirs(out_dir, exist_ok=True)
    created = []
    for i, (key, spec) in enumerate(LOTS.items()):
        img, labels = make_lot_image(spec, seed=seed + i)
        path = os.path.join(out_dir, f"{key}.jpg")
        cv2.imwrite(path, img)
        created.append({"key": key, "title": spec["title"], "file": f"{key}.jpg",
                        "n_beans": spec["n"]})
    return created


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "sample_images")
    created = generate_all(out)
    for c in created:
        print("dibuat:", c["file"], "(", c["n_beans"], "biji )")
