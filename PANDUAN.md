# Panduan Lengkap Sistem Grading Kopi Realtime

## 📋 Daftar Isi
1. [Instalasi](#instalasi)
2. [Persiapan Data Training](#persiapan-data-training)
3. [Training Model](#training-model)
4. [Menjalankan Sistem](#menjalankan-sistem)
5. [Tips dan Troubleshooting](#tips-dan-troubleshooting)

---

## 🚀 Instalasi

### 1. Install Python
Pastikan Python 3.8 atau lebih baru sudah terinstall:
```bash
python --version
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Catatan untuk Windows:**
- Jika ada error saat install TensorFlow, coba install versi CPU:
  ```bash
  pip install tensorflow-cpu==2.15.0
  ```

### 3. Test Camera
Jalankan script test untuk memastikan camera berfungsi:
```bash
python test_camera.py
```

---

## 📸 Persiapan Data Training

### Struktur Folder
```
data/
├── normal/        <- Biji kopi normal (tanpa cacat)
├── biji_hitam/    <- Biji hitam penuh/parsial
├── biji_cokelat/  <- Biji cokelat (over-fermentasi)
├── biji_hijau/    <- Biji hijau/belum matang ⭐ NEW
├── berlubang/     <- Biji berlubang (serangga)
├── pecah/         <- Biji pecah/retak
└── berjamur/      <- Biji berjamur
```

### Kriteria Klasifikasi Defect (Berdasarkan SCAA/SNI)

**Normal**
- Biji utuh, tidak pecah
- Warna coklat seragam
- Tidak ada cacat (hitam, lubang, jamur)
- Ukuran konsisten

**Biji Hitam**
- Biji berwarna hitam seluruhnya (full black) atau sebagian (partial black)
- Disebabkan oleh fermentasi berlebihan atau buah terinfeksi

**Biji Cokelat**
- Biji berwarna cokelat gelap akibat over-fermentasi
- Tekstur lebih lunak dari normal

**Biji Hijau (Belum Matang)** ⭐ **NEW - PENTING!**
- Biji berwarna hijau, hijau-kuning, atau kehijauan
- Belum matang sempurna saat panen
- **Karakteristik:**
  - Hue: 80-160° (HSV color space)
  - Saturation tinggi (>100)
  - LAB a-channel negatif (warna hijau)
  - Tekstur halus, mengkilap
- **Dampak jika tidak dipisahkan:**
  - ❌ Rasa pahit dan asam yang tidak diinginkan
  - ❌ Aroma tidak berkembang sempurna saat roasting
  - ❌ Menurunkan kualitas dan harga jual batch kopi
  - ❌ Menyebabkan kopi tidak lulus standar mutu

**Berlubang**
- Terdapat lubang yang disebabkan oleh serangan serangga (coffee berry borer)
- Lubang bisa tunggal atau multiple

**Pecah**
- Biji terbelah, retak, atau patah
- Biasanya terjadi saat proses pengolahan/pengeringan

**Berjamur**
- Terdapat pertumbuhan jamur pada permukaan biji
- Warna keputihan atau kehijauan

### Cara Mengumpulkan Data

#### Opsi 1: Foto Manual
1. Siapkan background putih/netral
2. Letakkan biji kopi satu per satu
3. Foto dengan pencahayaan yang baik
4. Simpan ke folder sesuai grade

**Tips Foto:**
- Gunakan pencahayaan natural atau lampu putih (hindari lampu hijau/kuning!)
- Jarak camera konsisten (20-30 cm)
- Background kontras dengan biji kopi
- Minimal 100 foto per grade untuk hasil baik

**⭐ Tips Khusus untuk Biji Hijau:**
- Foto di berbagai kondisi lighting (indoor/outdoor)
- Include berbagai tingkat "kehijauan":
  - Hijau penuh (totally unripe)
  - Setengah hijau (partially ripe)
  - Hijau kekuningan (almost ripe)
- Gunakan white balance yang akurat (jangan sampai biji normal kelihatan hijau)

#### Opsi 2: Capture dari Camera
Gunakan script test_camera.py:
```bash
python test_camera.py
```
- Tekan 's' untuk save image
- Pindahkan image ke folder grade yang sesuai

#### Opsi 3: Augmentasi Data Biji Hijau ⭐ NEW
Jika Anda kesulitan mendapat banyak sample biji hijau real:
```bash
python augment_green_beans.py
```

**Script ini akan:**
1. **Augmentasi biji hijau existing** (rotasi, flip, brightness, noise)
2. **Simulasi biji hijau dari biji normal** (hue shift, saturation boost)
3. **Generate "mixed green" samples** (setengah matang)

**Output folders:**
- `data/biji_hijau_augmented/` - Review & copy yang bagus ke `data/biji_hijau/`
- `data/biji_hijau_simulated/` - Synthetic green beans
- `data/biji_hijau_mixed/` - Mixed samples (half-ripe)

**⚠️ PENTING:**
- Synthetic data adalah **SUPPLEMENT** saja, bukan pengganti!
- Tetap usahakan kumpulkan data real sebanyak mungkin
- Review hasil augmentasi, hapus yang tidak realistis
- Copy yang bagus ke folder `data/biji_hijau/` untuk training

### Jumlah Data Recommended
- **Minimum**: 50 images per kelas (300 total)
- **Good**: 200 images per kelas (1200 total)
- **Excellent**: 500+ images per kelas (3000+ total)

---

## 🎓 Training Model

### 1. Pastikan Data Sudah Siap
Cek folder data:
```bash
dir data\normal
dir data\biji_hitam
dir data\biji_cokelat
dir data\berlubang
dir data\pecah
dir data\berjamur
```

### 2. Jalankan Training
```bash
python train_model.py
```

**Proses Training:**
- Model akan load semua images
- Split data menjadi training (80%) dan validation (20%)
- Training dengan early stopping (otomatis stop jika tidak improve)
- Save model ke `models/coffee_grading_model.h5`

**Waktu Training:**
- Tergantung jumlah data dan hardware
- CPU: 5-30 menit
- GPU: 1-5 menit

### 3. Monitor Training
Perhatikan output:
- **Training Accuracy**: Akurasi pada data training
- **Validation Accuracy**: Akurasi pada data validation

**Target:**
- Validation accuracy > 85% = Good
- Validation accuracy > 90% = Excellent

---

## 🎯 Menjalankan Sistem

### Mode Realtime
```bash
python main.py
```

### Kontrol Keyboard
- **q**: Quit/keluar dari aplikasi
- **r**: Reset statistics
- **s**: Screenshot (save current frame)

### Tampilan Interface
- **Bounding Box**: Kotak di sekitar biji kopi yang terdeteksi
  - Hijau = Normal
  - Merah = Biji Hitam
  - Orange = Biji Cokelat
  - Cyan = Berlubang
  - Ungu = Pecah
  - Kuning = Berjamur
  - Abu-abu = Uncertain (confidence rendah)
- **Label**: Jenis defect dan confidence score
- **Statistics Panel**: Jumlah dan persentase setiap kelas defect

---

## 🔍 Analisis Unsupervised & Segmentasi Warna (K-Means & PCA)

Untuk memperkuat usulan judul **Smart Grading**, sistem ini mengintegrasikan teknik **Unsupervised Learning** (Clustering & Segmentasi) berdampingan dengan model **Supervised Learning** (EfficientNetB0).

### 1. Analisis Unsupervised Dataset (PCA & K-Means)
Aplikasi menyediakan script `analyze_dataset.py` untuk mengelompokkan biji kopi secara otomatis berdasarkan fitur visualnya tanpa menggunakan label training.

#### Cara Menjalankan:
```bash
python analyze_dataset.py
```

#### Cara Kerja:
1. **Ekstraksi Fitur**: Sistem mendeteksi biji kopi lalu mengekstrak fitur warna (LAB & HSV), bentuk (Hu Moments, circularity, solidity), dan tekstur (GLCM & LBP).
2. **Reduksi Dimensi (PCA)**: Mengurangi dimensi fitur yang tinggi menjadi 2 dimensi (Principal Components) agar dapat divisualisasikan.
3. **Clustering (K-Means)**: Mengelompokkan seluruh biji kopi ke dalam 6 cluster ($K=6$).
4. **Metrik Evaluasi**:
   - **Silhouette Score**: Mengukur seberapa terpisah klaster yang terbentuk (mendekati 1 = sangat terpisah).
   - **Adjusted Rand Index (ARI)**: Mengukur kesesuaian klaster unsupervised dengan label defect yang sebenarnya (mendekati 1 = cocok sempurna).

#### Hasil Output:
- **Visualisasi**: File plot `models/clustering_analysis.png` yang menampilkan perbandingan visual 2D antara sebaran label asli vs hasil clustering K-Means.
- **Data Metrik**: JSON file `models/clustering_metrics.json`.

---

### 2. Segmentasi Warna Piksel dengan K-Means
Di modul `src/preprocessing.py`, terdapat fungsi `segment_kmeans(image, k=3)` yang membagi piksel gambar biji kopi ke dalam $k$ kelompok warna menggunakan algoritma K-Means di LAB color space.

- **Kegunaan**: Berguna untuk memisahkan daerah cacat (seperti bintik hitam, lubang, atau spora jamur) dari bagian biji kopi yang normal secara dinamis berdasarkan kontras warna.
- **Implementasi**: Otomatis dipanggil dalam pipeline pra-pemrosesan jika diperlukan analisis visual mendalam pada biji yang terdeteksi.

---

## 💡 Tips dan Troubleshooting

### Tips Penggunaan

**1. Setup Camera yang Baik**
- Gunakan pencahayaan yang cukup dan merata
- Background kontras (putih/hitam)
- Camera tegak lurus dengan permukaan
- Jarak optimal: 30-50 cm dari objek

**2. Optimasi Deteksi**
- Letakkan biji kopi terpisah (tidak menumpuk)
- Hindari bayangan yang terlalu kuat
- Gunakan conveyor belt untuk sistem otomatis

**3. Meningkatkan Akurasi**
- Tambah data training
- Pastikan data training berkualitas baik
- Balance jumlah data per grade
- Re-train model secara berkala

### Troubleshooting

**Problem: Camera tidak terdeteksi**
```
Error: Tidak bisa membuka camera 0
```
**Solusi:**
- Cek apakah camera terhubung
- Coba ganti camera_id (0, 1, 2, dst)
- Pastikan tidak ada aplikasi lain yang menggunakan camera

**Problem: Deteksi tidak akurat**
**Solusi:**
- Cek pencahayaan
- Adjust threshold di `preprocessing.py`
- Tambah data training
- Re-train model

**Problem: Model tidak ditemukan**
```
WARNING: Model tidak ditemukan
```
**Solusi:**
- Jalankan `python train_model.py` terlebih dahulu
- Pastikan file `models/coffee_grading_model.h5` ada

**Problem: FPS rendah/lag**
**Solusi:**
- Kurangi resolusi camera di `camera_capture.py`
- Gunakan model yang lebih ringan
- Upgrade hardware (GPU recommended)

**Problem: Import error**
```
ModuleNotFoundError: No module named 'cv2'
```
**Solusi:**
```bash
pip install opencv-python
```

### Kustomisasi

**Mengubah Kelas Klasifikasi**
Edit di `src/config.py`:
```python
MODEL_NUM_CLASSES = 6
GRADE_LABELS = ['Normal', 'Biji Hitam', 'Biji Cokelat', 'Berlubang', 'Pecah', 'Berjamur']
# Nama folder data = label lowercase, spasi diganti underscore
# Semua file lain otomatis mengikuti config ini
```

**Mengubah Resolusi Camera**
Edit di `camera_capture.py`:
```python
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Ubah sesuai kebutuhan
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
```

**Mengubah Threshold Deteksi**
Edit di `preprocessing.py`:
```python
if area > 500 and area < 50000:  # Adjust nilai ini
```

---

## 📊 Pengembangan Lanjutan

### Fitur yang Bisa Ditambahkan
1. **Export Data**: Save hasil grading ke CSV/Excel
2. **Database Integration**: Simpan hasil ke database
3. **Web Interface**: Akses sistem via browser
4. **Multi-camera**: Support multiple cameras
5. **Sorting Automation**: Integrasi dengan mesin sorting
6. **Cloud Deployment**: Deploy model ke cloud
7. **Mobile App**: Aplikasi mobile untuk monitoring

### Integrasi Hardware
- **Conveyor Belt**: Untuk sistem continuous
- **Pneumatic Sorter**: Untuk sorting otomatis
- **Industrial Camera**: Untuk kualitas image lebih baik
- **Lighting System**: Untuk pencahayaan konsisten

---

## 📞 Support

Jika ada pertanyaan atau masalah:
1. Cek dokumentasi ini terlebih dahulu
2. Review error message dengan teliti
3. Cek file log jika ada
4. Test dengan data sample sederhana

---

## 📝 Lisensi

Project ini dibuat untuk tujuan edukasi dan komersial.

**Selamat menggunakan sistem grading kopi! ☕**
