# Sistem Grading Kopi Realtime

Sistem klasifikasi kualitas biji kopi secara realtime menggunakan computer vision dan machine learning.

> 📄 **Dokumentasi PoC lengkap** (Smart Grading SNI, dashboard web, live scan kamera/mobile,
> dan evaluasi PoC YOLO object detection): lihat [docs/DOKUMENTASI_POC.md](docs/DOKUMENTASI_POC.md).

## Fitur
- Deteksi dan klasifikasi biji kopi dari camera realtime
- Grading berdasarkan warna, ukuran, dan cacat visual
- Interface GUI untuk monitoring
- Model machine learning untuk klasifikasi otomatis

## Teknologi
- Python 3.8+
- OpenCV untuk computer vision
- TensorFlow/Keras untuk machine learning
- NumPy untuk processing data

## Struktur Project
```
coffee-grading-system/
├── src/
│   ├── camera_capture.py      # Modul capture dari camera
│   ├── preprocessing.py        # Preprocessing image
│   ├── feature_extraction.py  # Ekstraksi fitur biji kopi
│   ├── model.py               # Model ML untuk klasifikasi
│   └── grading_system.py      # Sistem utama
├── models/                     # Saved models
├── data/                       # Dataset training
│   ├── grade_a/
│   ├── grade_b/
│   └── grade_c/
├── requirements.txt
└── main.py                     # Entry point aplikasi
```

## Instalasi
```bash
pip install -r requirements.txt
```

## Cara Pakai
```bash
python main.py
```

## Grade Kopi
- **Grade A**: Biji utuh, warna seragam, tidak ada cacat
- **Grade B**: Biji utuh, sedikit variasi warna
- **Grade C**: Biji pecah, warna tidak seragam, ada cacat

abc