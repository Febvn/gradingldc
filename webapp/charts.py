"""
Helper grafik berbasis SVG murni (tanpa pustaka eksternal / CDN).

Dipakai dashboard analitik agar tetap tampil walau offline. Mengembalikan
string SVG siap-tempel ke template (markup di-mark safe oleh Jinja autoescape
melalui filter |safe).
"""
from __future__ import annotations

from typing import List, Tuple
from html import escape

AMBER = "#d97706"
AMBER_LIGHT = "#fbbf24"
GREEN = "#10b981"
GRID = "#222c37"
TEXT = "#9ca3af"


def _x_positions(n: int, w: float, pad_l: float, pad_r: float) -> List[float]:
    if n <= 1:
        return [pad_l + (w - pad_l - pad_r) / 2.0]
    step = (w - pad_l - pad_r) / (n - 1)
    return [pad_l + i * step for i in range(n)]


def line_chart(labels: List[str], values: List[float], *,
               width: int = 720, height: int = 260, color: str = AMBER,
               fill: bool = True, y_suffix: str = "", title: str = "") -> str:
    """Grafik garis/area sederhana. labels=tanggal, values=angka."""
    if not values:
        return '<svg viewBox="0 0 100 40"></svg>'
    pad_l, pad_r, pad_t, pad_b = 44.0, 16.0, 20.0, 28.0
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        vmax = vmin + 1.0
    span = vmax - vmin

    xs = _x_positions(len(values), width, pad_l, pad_r)
    plot_h = height - pad_t - pad_b

    def y_of(v: float) -> float:
        return pad_t + plot_h * (1.0 - (v - vmin) / span)

    pts = [(xs[i], y_of(values[i])) for i in range(len(values))]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    # Gridlines + label sumbu Y (4 garis).
    grid, ylab = [], []
    for k in range(5):
        gv = vmin + span * k / 4.0
        gy = y_of(gv)
        grid.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r}" y2="{gy:.1f}" '
                    f'stroke="{GRID}" stroke-width="1"/>')
        ylab.append(f'<text x="{pad_l - 6}" y="{gy + 3:.1f}" text-anchor="end" '
                    f'font-size="10" fill="{TEXT}">{gv:.0f}{y_suffix}</text>')

    area = ""
    if fill:
        area = (f'<polygon points="{xs[0]:.1f},{pad_t + plot_h:.1f} {poly} '
                f'{xs[-1]:.1f},{pad_t + plot_h:.1f}" fill="{color}" opacity="0.12"/>')

    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{color}"/>' for x, y in pts)

    # Label X: tampilkan beberapa saja agar tidak berdesakan.
    n = len(labels)
    step = max(1, n // 6)
    xlab = []
    for i in range(0, n, step):
        xlab.append(f'<text x="{xs[i]:.1f}" y="{height - 8}" text-anchor="middle" '
                    f'font-size="9" fill="{TEXT}">{escape(labels[i][5:])}</text>')

    ttl = (f'<text x="{pad_l}" y="13" font-size="11" fill="{TEXT}">{escape(title)}</text>'
           if title else "")
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" '
            f'preserveAspectRatio="xMidYMid meet" role="img">'
            f'{ttl}{"".join(grid)}{area}'
            f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.5"/>'
            f'{dots}{"".join(ylab)}{"".join(xlab)}</svg>')


def pareto_chart(items: List[dict], *, width: int = 720, height: int = 280) -> str:
    """Diagram Pareto: bar count + garis kumulatif %. items=[{label,count,cum_pct}]."""
    if not items:
        return '<svg viewBox="0 0 100 40"></svg>'
    pad_l, pad_r, pad_t, pad_b = 44.0, 44.0, 16.0, 52.0
    plot_h = height - pad_t - pad_b
    plot_w = width - pad_l - pad_r
    n = len(items)
    cmax = max(it["count"] for it in items) or 1
    bw = plot_w / n * 0.6
    gap = plot_w / n

    bars, blabels, cumpts = [], [], []
    for i, it in enumerate(items):
        cx = pad_l + gap * i + gap * 0.2
        bh = plot_h * it["count"] / cmax
        by = pad_t + plot_h - bh
        bars.append(f'<rect x="{cx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                    f'rx="3" fill="{AMBER}" opacity="0.85"/>')
        bars.append(f'<text x="{cx + bw/2:.1f}" y="{by - 4:.1f}" text-anchor="middle" '
                    f'font-size="9" fill="#e5e7eb">{it["count"]}</text>')
        blabels.append(f'<text x="{cx + bw/2:.1f}" y="{height - 30:.1f}" text-anchor="middle" '
                       f'font-size="9" fill="{TEXT}">{escape(it["label"])}</text>')
        # Titik kumulatif (skala 0..100% pada tinggi plot).
        cy = pad_t + plot_h * (1.0 - it["cum_pct"] / 100.0)
        cumpts.append((cx + bw / 2, cy, it["cum_pct"]))

    line = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in cumpts)
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{GREEN}"/>'
                   f'<text x="{x:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="8" '
                   f'fill="{GREEN}">{p:.0f}%</text>' for x, y, p in cumpts)

    return (f'<svg viewBox="0 0 {width} {height}" width="100%" '
            f'preserveAspectRatio="xMidYMid meet" role="img">'
            f'{"".join(bars)}'
            f'<polyline points="{line}" fill="none" stroke="{GREEN}" stroke-width="2"/>'
            f'{dots}{"".join(blabels)}</svg>')


def donut_chart(segments: List[Tuple[str, int, str]], *, size: int = 200) -> str:
    """Donut sederhana. segments=[(label, value, color)]."""
    total = sum(v for _, v, _ in segments) or 1
    cx = cy = size / 2.0
    r = size / 2.0 - 12
    circ = 2 * 3.141592653589793 * r
    offset = 0.0
    rings = []
    for _, v, color in segments:
        frac = v / total
        dash = circ * frac
        rings.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" '
            f'stroke-width="20" stroke-dasharray="{dash:.2f} {circ - dash:.2f}" '
            f'stroke-dashoffset="{-offset:.2f}" transform="rotate(-90 {cx} {cy})"/>'
        )
        offset += dash
    return (f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}">'
            f'{"".join(rings)}'
            f'<text x="{cx}" y="{cy - 2}" text-anchor="middle" font-size="22" '
            f'font-weight="700" fill="#f3f4f6">{total}</text>'
            f'<text x="{cx}" y="{cy + 16}" text-anchor="middle" font-size="10" '
            f'fill="{TEXT}">SESI</text></svg>')
