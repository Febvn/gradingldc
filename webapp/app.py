"""
Smart Grading Kopi — Web Dashboard (Proof of Concept).

Server Flask yang menyatukan seluruh pipeline menjadi aplikasi yang bisa
didemokan di browser, sesuai amanat proposal: "dashboard pemantauan / aplikasi
grading yang intuitif untuk staf di lapangan".

Halaman:
    /            Grading   : pilih sampel lot / unggah gambar -> mutu SNI
    /dashboard   Analitik  : KPI & tren operasional dari riwayat grading
    /about       Tentang   : pemetaan PoC ke proposal & arsitektur

Jalankan:
    python webapp/app.py            (lalu buka http://127.0.0.1:5000)
"""
import os
# Keras 2 (tf-keras) sebelum tensorflow di-import oleh inference_engine.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONUTF8", "1")

import sys
import io
import time

import numpy as np
import cv2
from flask import (Flask, render_template, request, redirect, url_for,
                   send_from_directory, jsonify, flash)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

from analytics import GradingStore, OperationalAnalytics, seed_simulated_history, GRADE_ORDER
import charts
import make_samples

SAMPLE_DIR = os.path.join(_HERE, "sample_images")
# Vercel filesystem is read-only except /tmp; fall back to /tmp when running serverless.
if os.environ.get("VERCEL") or not os.access(_ROOT, os.W_OK):
    DB_PATH = "/tmp/grading_sessions.db"
else:
    DB_PATH = os.path.join(_ROOT, "data_store", "grading_sessions.db")

app = Flask(__name__)
app.secret_key = "smart-grading-poc-ldc"
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB

# Palet warna mutu untuk UI.
GRADE_COLORS = {
    "Mutu 1": "#10b981", "Mutu 2": "#22c55e", "Mutu 3": "#eab308",
    "Mutu 4a": "#f59e0b", "Mutu 4b": "#f97316", "Mutu 5": "#ef4444",
    "Mutu 6": "#dc2626", "Di luar mutu": "#7f1d1d",
}
CLASS_COLORS = {
    "Normal": "#10b981", "Biji Hitam": "#ef4444", "Biji Cokelat": "#f97316",
    "Berlubang": "#06b6d4", "Pecah": "#a855f7", "Berjamur": "#eab308",
    "Uncertain": "#6b7280",
}

_engine = None  # lazy: hindari memuat TensorFlow sampai benar-benar dibutuhkan.
_demo_seeded = False


@app.before_request
def _ensure_demo():
    global _demo_seeded
    if not _demo_seeded:
        _demo_seeded = True
        ensure_demo_data()


def get_engine():
    global _engine
    if _engine is None:
        from inference_engine import SmartGradingEngine
        _engine = SmartGradingEngine()
    return _engine


def get_store():
    return GradingStore(DB_PATH)


def ensure_demo_data():
    """Siapkan sampel lot & seed riwayat simulasi bila belum ada."""
    if not os.path.isdir(SAMPLE_DIR) or not os.listdir(SAMPLE_DIR):
        make_samples.generate_all(SAMPLE_DIR)
    store = get_store()
    if store.count() == 0:
        seed_simulated_history(store, days=30, per_day=6)


def list_samples():
    titles = {s["key"]: s["title"] for s in
              [{"key": k, "title": v["title"]} for k, v in make_samples.LOTS.items()]}
    items = []
    if os.path.isdir(SAMPLE_DIR):
        for f in sorted(os.listdir(SAMPLE_DIR)):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                key = os.path.splitext(f)[0]
                items.append({"file": f, "title": titles.get(key, key)})
    return items


# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", samples=list_samples(),
                           grade_colors=GRADE_COLORS, class_colors=CLASS_COLORS,
                           result=None, active="grade")


@app.route("/grade", methods=["GET", "POST"])
def grade():
    img = None
    source = "upload"

    # GET dengan ?sample=... -> demo yang dapat di-bookmark.
    sample = request.form.get("sample") or request.args.get("sample")
    if sample:
        path = os.path.join(SAMPLE_DIR, os.path.basename(sample))
        if os.path.exists(path):
            img = cv2.imread(path)
            source = sample
    elif "image" in request.files and request.files["image"].filename:
        f = request.files["image"]
        data = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        source = f.filename

    if img is None:
        flash("Tidak ada gambar valid. Pilih sampel atau unggah file gambar.")
        return redirect(url_for("index"))

    line = request.form.get("line", "Demo")
    try:
        engine = get_engine()
        result = engine.analyze_image(img, draw=True)
    except Exception as e:
        flash(f"Model ML tidak tersedia di environment ini: {e}")
        return redirect(url_for("index"))

    # Catat sesi (kecuali bila model belum dilatih -> tetap dicatat sbg indikatif).
    try:
        get_store().add_session(result["grade"], source=source, line=line,
                                operator="Web", simulated=False)
    except Exception as e:  # pragma: no cover
        print("Gagal mencatat sesi:", e)

    return render_template("index.html", samples=list_samples(),
                           grade_colors=GRADE_COLORS, class_colors=CLASS_COLORS,
                           result=result, source=source, active="grade")


@app.route("/dashboard")
def dashboard():
    store = get_store()
    an = OperationalAnalytics(store)
    overview = an.overview()
    trend = an.daily_trend()
    pareto = an.defect_pareto()
    by_line = an.by_line()
    recent = store.recent(12)

    # Grafik SVG.
    trend_labels = [t["date"] for t in trend]
    chart_defect_value = charts.line_chart(
        trend_labels, [t["avg_defect_value"] for t in trend],
        title="Rata-rata nilai cacat / 300 g", color=charts.AMBER)
    chart_premium = charts.line_chart(
        trend_labels, [t["premium_pct"] for t in trend],
        title="Yield mutu premium (%)", color=charts.GREEN, y_suffix="%")
    chart_pareto = charts.pareto_chart(pareto)

    # Donut distribusi mutu.
    donut_segments = []
    if not overview.get("empty"):
        for g in GRADE_ORDER:
            c = overview["grade_distribution"].get(g, 0)
            if c > 0:
                donut_segments.append((g, c, GRADE_COLORS.get(g, "#888")))
    chart_donut = charts.donut_chart(donut_segments) if donut_segments else ""

    return render_template("dashboard.html", active="dashboard",
                           overview=overview, trend=trend, pareto=pareto,
                           by_line=by_line, recent=recent,
                           grade_colors=GRADE_COLORS,
                           chart_defect_value=chart_defect_value,
                           chart_premium=chart_premium,
                           chart_pareto=chart_pareto, chart_donut=chart_donut,
                           donut_segments=donut_segments)


@app.route("/live")
def live():
    return render_template("live.html", active="live",
                           grade_colors=GRADE_COLORS, class_colors=CLASS_COLORS)


@app.route("/about")
def about():
    return render_template("about.html", active="about")


@app.route("/api/grade", methods=["POST"])
def api_grade():
    """Endpoint JSON dipakai halaman Live & integrasi sistem lain.

    Param opsional: ?record=1 untuk mencatat sesi ke dashboard,
    ?annotate=1 untuk menyertakan gambar anotasi base64.
    """
    if "image" not in request.files:
        return jsonify({"error": "field 'image' (file) wajib"}), 400
    data = np.frombuffer(request.files["image"].read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "gambar tidak valid"}), 400
    annotate = request.args.get("annotate") == "1"
    try:
        res = get_engine().analyze_image(img, draw=annotate)
    except Exception as e:
        return jsonify({"error": f"Model tidak tersedia: {e}"}), 503
    if request.args.get("record") == "1":
        try:
            get_store().add_session(res["grade"], source="live-camera",
                                    line=request.form.get("line", "Live"),
                                    operator="Live")
        except Exception as e:  # pragma: no cover
            print("Gagal mencatat sesi live:", e)
    return jsonify(res)


@app.route("/reseed", methods=["POST"])
def reseed():
    store = get_store()
    store.clear()
    n = seed_simulated_history(store, days=30, per_day=6)
    flash(f"Riwayat simulasi dibuat ulang: {n} sesi.")
    return redirect(url_for("dashboard"))


@app.route("/sample_images/<path:fname>")
def sample_image(fname):
    return send_from_directory(SAMPLE_DIR, fname)


if __name__ == "__main__":
    import argparse
    from sslcert import ensure_cert, get_lan_ip

    ap = argparse.ArgumentParser(description="Smart Grading Kopi — Web Dashboard")
    ap.add_argument("--host", default=None,
                    help="Host bind (default: 127.0.0.1; otomatis 0.0.0.0 bila --https)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--https", action="store_true",
                    help="Aktifkan HTTPS (WAJIB agar kamera HP bisa diakses via LAN)")
    args = ap.parse_args()

    ensure_demo_data()

    ssl_context = None
    scheme = "http"
    host = args.host or "127.0.0.1"
    if args.https:
        scheme = "https"
        host = args.host or "0.0.0.0"  # agar bisa diakses dari HP di jaringan sama
        cert, key = ensure_cert(os.path.join(_HERE, ".certs"))
        ssl_context = (cert, key)

    lan = get_lan_ip()
    print("=" * 64)
    print("  Smart Grading Kopi — Web Dashboard (PoC)")
    print(f"  Laptop ini :  {scheme}://127.0.0.1:{args.port}")
    if args.https:
        print(f"  Dari HP    :  {scheme}://{lan}:{args.port}/live")
        print("  (HP & laptop harus di Wi-Fi yang sama; terima peringatan sertifikat)")
    else:
        print(f"  Live scan  :  {scheme}://127.0.0.1:{args.port}/live")
        print("  Untuk kamera HP via LAN, jalankan dengan:  python webapp/app.py --https")
    print("=" * 64)
    app.run(host=host, port=args.port, debug=False, ssl_context=ssl_context, threaded=True)
