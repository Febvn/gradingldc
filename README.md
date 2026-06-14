# Sistem Grading Kopi Realtime

Sistem klasifikasi kualitas biji kopi secara realtime menggunakan computer vision dan machine learning.

## Fitur
- Deteksi dan klasifikasi biji kopi dari camera realtime
- **7 Kelas Defect**: Normal, Biji Hitam, Biji Cokelat, **Biji Hijau (Belum Matang)**, Berlubang, Pecah, Berjamur
- **Deteksi Biji Hijau**: Color-based detection khusus untuk biji kopi yang belum matang
- Interface GUI untuk monitoring
- Model machine learning dengan EfficientNetB0 backbone
- Feature extraction komprehensif (HSV, LAB, GLCM, LBP, Shape, Texture)

## Teknologi
- Python 3.8+
- OpenCV untuk computer vision
- TensorFlow/Keras untuk machine learning
- NumPy untuk processing data
- EfficientNetB0 (transfer learning) + Fine-tuning

## Struktur Project
```
coffee-grading-system/
├── src/
│   ├── camera_capture.py      # Modul capture dari camera
│   ├── preprocessing.py        # Preprocessing image + green detection
│   ├── feature_extraction.py  # Ekstraksi fitur + green bean analysis
│   ├── model.py               # Model ML untuk klasifikasi
│   ├── grading_system.py      # Sistem utama
│   └── config.py              # Konfigurasi terpusat
├── models/                     # Saved models
├── data/                       # Dataset training
│   ├── normal/                # Biji kopi normal (matang, tidak cacat)
│   ├── biji_hitam/            # Biji hitam (over-fermented)
│   ├── biji_cokelat/          # Biji cokelat muda (under-roasted)
│   ├── biji_hijau/            # Biji hijau / belum matang ⭐ NEW
│   ├── berlubang/             # Biji berlubang (insect damage)
│   ├── pecah/                 # Biji pecah (broken)
│   └── berjamur/              # Biji berjamur (moldy)
├── requirements.txt
├── main.py                     # Entry point aplikasi
├── train_model.py             # Training script
└── augment_green_beans.py     # Augmentasi data biji hijau ⭐ NEW
```

## Instalasi
```bash
pip install -r requirements.txt
```

## Cara Pakai

### 1. Siapkan Data Training
Letakkan gambar biji kopi sesuai kelas di folder `data/`:
- `data/normal/` - Biji kopi matang sempurna
- `data/biji_hijau/` - Biji kopi hijau/belum matang ⭐
- `data/biji_hitam/` - Biji hitam
- dst.

**Tips untuk Biji Hijau:**
- Gunakan `augment_green_beans.py` untuk generate data synthetic
- Atau foto biji kopi hijau asli dengan berbagai sudut pencahayaan

### 2. Augmentasi Data Biji Hijau (Opsional)
```bash
python augment_green_beans.py
```
Script ini akan:
- Augmentasi existing biji hijau dengan berbagai transformasi
- Simulasi biji hijau dari biji normal (color transformation)
- Generate sample "mixed green" (setengah matang)

### 3. Train Model
```bash
python train_model.py
```

Model akan training dengan:
- **Phase 1**: Train classifier head (50 epochs)
- **Phase 2**: Fine-tune EfficientNetB0 top layers (30 epochs)
- Data augmentation komprehensif
- Class weighting untuk handle imbalanced data

### 4. Run Aplikasi Realtime
```bash
python main.py
```

## Deteksi Biji Hijau - Cara Kerja

### Color Features (HSV/LAB)
- **Hue Range**: Biji hijau memiliki Hue 80-160° (hijau-kuning)
- **Saturation**: Lebih tinggi dari biji matang
- **LAB a-channel**: Negatif (< -10) untuk warna hijau

### Green Detection Algorithm
1. Convert image ke HSV dan LAB color space
2. Create mask untuk pixel di range hijau
3. Calculate percentage pixel hijau
4. Analyze saturation dan LAB a-channel
5. Compute green color ratio (G / (R+B))

### Feature Extraction untuk Biji Hijau
```python
{
    'green_pixel_percentage': 0.35,      # 35% pixel hijau
    'green_saturation_mean': 145.2,      # Saturation tinggi
    'green_lab_a_percentage': 0.28,      # 28% pixel LAB-a negatif
    'dominant_hue_degrees': 120,         # Hue dominan di range hijau
    'green_color_ratio': 1.42,           # G channel > R+B
    'is_likely_green': 1.0               # Flag: kemungkinan hijau
}
```

## Model Architecture

**Backbone**: EfficientNetB0 (pretrained ImageNet)
- Input: 224x224x3
- GlobalAveragePooling2D
- Dense(256) + Dropout(0.4)
- Dense(128) + Dropout(0.3)
- Dense(7, softmax) - 7 kelas output

**Training Strategy**:
- Label smoothing (0.1)
- L2 regularization (0.01)
- Mixup augmentation (α=0.2)
- Cosine annealing with warmup
- Class weighting untuk imbalanced data

## Performance Improvement Tips

### Untuk Deteksi Biji Hijau Lebih Akurat:
1. **Kumpulkan data real biji hijau** - paling penting!
2. **Berbagai kondisi pencahayaan** - indoor, outdoor, diffused light
3. **Berbagai tingkat "kehijauan"**:
   - Hijau penuh (totally unripe)
   - Setengah hijau (partially ripe)
   - Hijau kekuningan (almost ripe)
4. **Augmentasi warna** - gunakan `augment_green_beans.py`
5. **Balance dataset** - jumlah sample tiap kelas sebanding
6. **Fine-tune threshold** - edit `Config` di `src/config.py`

### Hyperparameter Tuning:
Edit `src/config.py`:
```python
# Tambah epochs jika model belum converge
TRAIN_P1_EPOCHS = 100
TRAIN_P2_EPOCHS = 50

# Kurangi learning rate jika training tidak stabil
TRAIN_P1_LEARNING_RATE = 5e-4

# Adjust augmentation intensity
AUG_ROTATION_RANGE = 45
AUG_BRIGHTNESS_RANGE = (0.6, 1.4)
```

## Grade Kopi (7 Kelas)

### 🟢 Normal
Biji matang sempurna, warna coklat seragam, tidak ada cacat

### ⚫ Biji Hitam
Biji over-fermented, warna hitam, kualitas buruk

### 🟤 Biji Cokelat
Biji under-roasted, warna coklat muda

### 🟢 Biji Hijau ⭐ NEW
Biji belum matang, warna hijau/hijau kekuningan, harus dipisahkan

### 🕳️ Berlubang
Biji dengan lubang (insect damage, borer beetle)

### 💥 Pecah
Biji pecah/patah, broken beans

### 🟢🔴 Berjamur
Biji dengan jamur, moldy appearance

## Troubleshooting

### Model Tidak Bisa Detect Biji Hijau
- Cek apakah folder `data/biji_hijau/` ada dan terisi
- Pastikan jumlah sample biji hijau ≥ 50 images
- Run `python augment_green_beans.py` untuk generate data
- Retrain model: `python train_model.py`

### Akurasi Rendah untuk Biji Hijau
- **Tambah data real** - synthetic data hanya supplement
- **Balance dataset** - semua kelas punya jumlah sample serupa
- **Increase epochs** - model belum fully learned
- **Adjust color range** - edit `detect_green_region()` di `preprocessing.py`

### False Positive (Normal Dianggap Hijau)
- Cek pencahayaan - hindari green-tinted lighting
- Enable Auto White Balance: `AWB_ENABLED = True` di config
- Adjust threshold di `detect_green_bean_characteristics()`

## Contributing
Untuk meningkatkan akurasi deteksi biji hijau, kontribusi dataset sangat dihargai!

## License
MIT License
