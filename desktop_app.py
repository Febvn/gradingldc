"""
Smart Grading Kopi — Desktop Application (Native Window).

Membungkus Flask webapp menjadi aplikasi Desktop standalone menggunakan
`flaskwebgui`. Fitur:
  - Splash screen saat loading model TensorFlow
  - Auto-detect Chrome / Edge / Chromium
  - Graceful shutdown & error handling
  - Desktop-optimized window (1280×800, no address bar)

Jalankan:
    python desktop_app.py              (mode development)
    dist/SmartGradingKopi.exe          (setelah di-build dengan PyInstaller)
"""
import os
import sys
import time
import threading
import webbrowser
import logging

# ── Environment setup (SEBELUM TensorFlow/Keras di-import) ──────────────
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONUTF8", "1")

# ── Path resolution (frozen EXE vs script) ──────────────────────────────
if getattr(sys, "frozen", False):
    APPLICATION_PATH = sys._MEIPASS
    os.environ["SMART_GRADING_ROOT"] = sys._MEIPASS
else:
    APPLICATION_PATH = os.path.dirname(os.path.abspath(__file__))
    os.environ["SMART_GRADING_ROOT"] = APPLICATION_PATH

sys.path.insert(0, APPLICATION_PATH)
sys.path.insert(0, os.path.join(APPLICATION_PATH, "src"))
sys.path.insert(0, os.path.join(APPLICATION_PATH, "webapp"))

# ── Logging ─────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(APPLICATION_PATH, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "desktop_app.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("SmartGradingDesktop")

# ── Splash Screen (tkinter-based, shown while Flask loads) ──────────────

_splash_root = None


def _show_splash():
    """Tampilkan splash screen sederhana saat aplikasi memuat model."""
    global _splash_root
    try:
        import tkinter as tk
    except ImportError:
        log.warning("tkinter tidak tersedia — skip splash screen.")
        return

    _splash_root = tk.Tk()
    _splash_root.overrideredirect(True)  # tanpa title bar
    _splash_root.attributes("-topmost", True)
    _splash_root.configure(bg="#0c0f12")

    sw, sh = 520, 280
    x = (_splash_root.winfo_screenwidth() - sw) // 2
    y = (_splash_root.winfo_screenheight() - sh) // 2
    _splash_root.geometry(f"{sw}x{sh}+{x}+{y}")

    # ── Canvas splash ──
    canvas = tk.Canvas(_splash_root, width=sw, height=sh, bg="#0c0f12",
                       highlightthickness=0, bd=0)
    canvas.pack()

    # Border gradient effect
    canvas.create_rectangle(0, 0, sw, sh, outline="#d97706", width=2)

    # Logo
    canvas.create_text(sw // 2, 60, text="☕", font=("Segoe UI Emoji", 40),
                       fill="#fbbf24")

    # Title
    canvas.create_text(sw // 2, 120, text="Smart Grading Kopi",
                       font=("Segoe UI", 22, "bold"), fill="#f3f4f6")

    # Subtitle
    canvas.create_text(sw // 2, 155,
                       text="SNI 01-2907-2008 · Computer Vision · PT LDC",
                       font=("Segoe UI", 10), fill="#9ca3af")

    # Loading bar background
    bar_y = 200
    bar_w = 360
    bar_h = 6
    bar_x = (sw - bar_w) // 2
    canvas.create_rectangle(bar_x, bar_y, bar_x + bar_w, bar_y + bar_h,
                            fill="#1a2129", outline="")

    # Animated loading bar
    loading_bar = canvas.create_rectangle(bar_x, bar_y,
                                          bar_x + 60, bar_y + bar_h,
                                          fill="#d97706", outline="")

    # Status text
    status_text = canvas.create_text(sw // 2, 230,
                                     text="Memuat komponen sistem...",
                                     font=("Segoe UI", 9), fill="#9ca3af")

    # Footer
    canvas.create_text(sw // 2, 260,
                       text="Kerja Praktik — Teknik Informatika ITERA 2026",
                       font=("Segoe UI", 8), fill="#4b5563")

    # ── Animasi loading bar ──
    _anim_state = {"pos": 0, "direction": 1}

    def animate():
        if _splash_root is None:
            return
        try:
            _anim_state["pos"] += _anim_state["direction"] * 8
            if _anim_state["pos"] + 60 >= bar_x + bar_w:
                _anim_state["direction"] = -1
            elif _anim_state["pos"] <= bar_x:
                _anim_state["direction"] = 1
            canvas.coords(loading_bar,
                          _anim_state["pos"], bar_y,
                          _anim_state["pos"] + 80, bar_y + bar_h)
            _splash_root.after(30, animate)
        except tk.TclError:
            pass

    animate()
    _splash_root.update()


def _close_splash():
    """Tutup splash screen."""
    global _splash_root
    if _splash_root is not None:
        try:
            _splash_root.destroy()
        except Exception:
            pass
        _splash_root = None


# ── Browser detection ───────────────────────────────────────────────────

def _find_browser():
    """Auto-detect Chrome, Edge, atau Chromium untuk flaskwebgui."""
    import shutil

    # Windows registry paths for Chrome/Edge
    possible_paths = [
        # Chrome
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
        # Edge
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
    ]

    for p in possible_paths:
        if os.path.isfile(p):
            log.info(f"Browser ditemukan: {p}")
            return p

    # Fallback: cari di PATH
    for exe in ("chrome", "google-chrome", "chromium", "msedge"):
        found = shutil.which(exe)
        if found:
            log.info(f"Browser ditemukan di PATH: {found}")
            return found

    log.warning("Tidak ditemukan Chrome/Edge — flaskwebgui akan mencari sendiri.")
    return None


# ── Main application ────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  Smart Grading Kopi — Desktop Application")
    log.info("=" * 60)

    # 1. Tampilkan splash screen
    splash_thread = threading.Thread(target=_show_splash, daemon=True)
    splash_thread.start()
    time.sleep(0.3)  # beri waktu splash muncul

    try:
        # 2. Import Flask app (ini akan memicu import TensorFlow dll.)
        log.info("Mengimpor modul Flask app...")
        from webapp.app import app, ensure_demo_data

        # 3. Siapkan demo data (sampel lot + riwayat simulasi)
        log.info("Menyiapkan data demo...")
        ensure_demo_data()

        # 4. Inject desktop-mode flag ke template context
        @app.context_processor
        def inject_desktop_mode():
            return {"desktop_mode": True}

        # 5. Tutup splash screen
        _close_splash()

        # 6. Cari browser
        browser_path = _find_browser()

        # 7. Jalankan UI desktop
        log.info("Meluncurkan antarmuka Desktop...")
        log.info("Jangan tutup terminal ini jika Anda menjalankannya dari script.")

        from flaskwebgui import FlaskUI

        ui_kwargs = {
            "app": app,
            "server": "flask",
            "port": 5050,
            "width": 1280,
            "height": 820,
        }
        if browser_path:
            ui_kwargs["browser_path"] = browser_path

        ui = FlaskUI(**ui_kwargs)
        ui.run()

    except ImportError as e:
        _close_splash()
        error_msg = f"Dependensi tidak ditemukan: {e}"
        log.error(error_msg)
        log.error("Pastikan semua dependensi terinstall: pip install -r requirements.txt")
        _show_error_dialog(
            "Dependensi Tidak Ditemukan",
            f"{error_msg}\n\nJalankan:\n  pip install -r requirements.txt"
        )
        sys.exit(1)

    except Exception as e:
        _close_splash()
        error_msg = f"Gagal menjalankan aplikasi: {e}"
        log.error(error_msg, exc_info=True)
        _show_error_dialog("Error", error_msg)
        sys.exit(1)


def _show_error_dialog(title, message):
    """Tampilkan dialog error menggunakan tkinter."""
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
    except Exception:
        # Fallback: print ke konsol
        print(f"\n{'='*50}")
        print(f"ERROR: {title}")
        print(f"{'='*50}")
        print(message)
        print(f"{'='*50}")
        input("Tekan Enter untuk keluar...")


if __name__ == "__main__":
    main()
