"""
Script untuk analisis Unsupervised Learning (Clustering & Dimensionality Reduction)
- Ekstraksi fitur visual (warna, bentuk, tekstur) dari dataset
- Reduksi dimensi menggunakan PCA (Principal Component Analysis)
- Clustering menggunakan K-Means
- Evaluasi Silhouette Score & Adjusted Rand Index (ARI)
- Visualisasi sebaran cluster vs kelas asli
"""
import sys
import os
import numpy as np
import cv2
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from config import Config
except ImportError:
    Config = None

from preprocessing import ImagePreprocessor
from feature_extraction import FeatureExtractor

# Try import sklearn
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("ERROR: scikit-learn tidak terinstall. Pastikan requirements sudah dipenuhi.")

# Try import seaborn
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def generate_synthetic_dataset(data_dir, num_samples_per_class=20):
    """
    Menghasilkan dataset sintetis biji kopi untuk testing awal.
    Membantu user agar bisa langsung menjalankan training/analisis secara mandiri.
    """
    grade_labels = Config.GRADE_LABELS if Config is not None else ['Normal', 'Biji Hitam', 'Biji Cokelat', 'Berlubang', 'Pecah', 'Berjamur']
    folder_names = [label.lower().replace(' ', '_') for label in grade_labels]
    
    print("=" * 60)
    print(f"DATASET TIDAK DITEMUKAN / KOSONG")
    print(f"Menghasilkan {num_samples_per_class} gambar sintetis per kelas di '{data_dir}' untuk simulasi...")
    print("=" * 60)
    
    # Random seed untuk konsistensi
    np.random.seed(42)
    
    for folder, label in zip(folder_names, grade_labels):
        target_folder = os.path.join(data_dir, folder)
        os.makedirs(target_folder, exist_ok=True)
        
        for i in range(num_samples_per_class):
            # 1. Background piring abu-abu terang dengan variasi noise halus
            img = np.ones((224, 224, 3), dtype=np.uint8) * 225
            noise = np.random.normal(0, 3, img.shape).astype(np.int16)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            # 2. Gambar biji kopi di tengah dengan orientasi acak
            cx, cy = 112, 112
            
            if label == 'Pecah':
                # Biji pecah ukurannya lebih kecil atau berbentuk patahan
                axes = (np.random.randint(28, 38), np.random.randint(18, 24))
            else:
                axes = (np.random.randint(46, 56), np.random.randint(32, 38))
                
            angle = np.random.randint(0, 180)
            
            # Skema warna BGR
            if label == 'Normal':
                color = (35, 75, 140)  # Cokelat terang segar
            elif label == 'Biji Hitam':
                color = (25, 25, 30)   # Hitam gosong/busuk
            elif label == 'Biji Cokelat':
                color = (20, 50, 95)   # Cokelat gelap asam/over-fermentasi
            elif label == 'Berlubang':
                color = (35, 75, 140)  # Cokelat terang
            elif label == 'Pecah':
                color = (38, 78, 142)  # Cokelat terang
            elif label == 'Berjamur':
                color = (65, 85, 95)   # Cokelat keabu-abuan/pucat
            else:
                color = (35, 75, 140)
            
            # Variasi warna kecil agar tampak alami
            color = tuple(int(np.clip(c + np.random.randint(-10, 10), 0, 255)) for c in color)
            
            # Menggambar bentuk biji kopi
            cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, color, -1)
            
            # Menggambar alur garis tengah biji kopi
            rad = np.radians(angle)
            dx = int(axes[0] * np.cos(rad) * 0.8)
            dy = int(axes[0] * np.sin(rad) * 0.8)
            groove_color = tuple(max(0, c - 35) for c in color)
            cv2.line(img, (cx - dx, cy - dy), (cx + dx, cy + dy), groove_color, 2)
            
            # 3. Menggambar detail cacat yang khas
            if label == 'Berlubang':
                # Lubang serangga (titik hitam kecil)
                for _ in range(np.random.randint(1, 3)):
                    hx = cx + np.random.randint(-20, 20)
                    hy = cy + np.random.randint(-15, 15)
                    cv2.circle(img, (hx, hy), np.random.randint(3, 5), (10, 10, 10), -1)
            elif label == 'Pecah':
                # Potong sebagian biji kopi menggunakan poligon sewarna background
                pts = np.array([
                    [cx - 65, cy - 65],
                    [cx + 65, cy - 65],
                    [cx - 5, cy + 5]
                ], dtype=np.int32)
                cv2.fillPoly(img, [pts], (225, 225, 225))
            elif label == 'Berjamur':
                # Bercak jamur (overlay putih/kuning keputihan)
                for _ in range(3):
                    px = cx + np.random.randint(-22, 22)
                    py = cy + np.random.randint(-22, 22)
                    overlay = img.copy()
                    cv2.circle(overlay, (px, py), np.random.randint(10, 18), (210, 230, 230), -1)
                    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, dst=img)
            
            # Tambahkan blur sedikit agar natural
            img = cv2.GaussianBlur(img, (3, 3), 0)
            
            filename = f"sync_coffee_{i:03d}.jpg"
            cv2.imwrite(os.path.join(target_folder, filename), img)
            
    print(f"✓ Berhasil menulis {num_samples_per_class * len(grade_labels)} gambar sintetis ke '{data_dir}/'")
    print()


def load_and_extract_features(data_dir):
    """
    Load dataset dan lakukan ekstraksi semua fitur visual.
    
    Args:
        data_dir: Path ke folder data
        
    Returns:
        tuple: (feature_vectors, true_labels, true_label_names, image_paths)
    """
    grade_labels = Config.GRADE_LABELS if Config is not None else ['Normal', 'Biji Hitam', 'Biji Cokelat', 'Berlubang', 'Pecah', 'Berjamur']
    folder_names = [label.lower().replace(' ', '_') for label in grade_labels]
    
    # Cek apakah folder data kosong
    is_empty = True
    supported_formats = Config.SUPPORTED_FORMATS if Config is not None else ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    
    if os.path.exists(data_dir):
        for folder in folder_names:
            folder_path = os.path.join(data_dir, folder)
            if os.path.exists(folder_path):
                files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_formats)]
                if len(files) > 0:
                    is_empty = False
                    break
                    
    if is_empty:
        generate_synthetic_dataset(data_dir, num_samples_per_class=25)
        
    preprocessor = ImagePreprocessor()
    extractor = FeatureExtractor()
    
    feature_vectors = []
    true_labels = []
    true_label_names = []
    image_paths = []
    
    print("Mulai ekstraksi fitur...")
    for idx, (folder, label_name) in enumerate(zip(folder_names, grade_labels)):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            continue
            
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_formats)]
        print(f"  Memproses '{folder_path}' ({len(files)} file)...")
        
        for filename in files:
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            
            if img is not None:
                # 1. Gunakan preprocessor untuk mencari contour biji kopi
                beans = preprocessor.detect_coffee_beans(img)
                
                if len(beans) > 0:
                    # Ambil bean contour terbesar
                    bean = beans[0]
                    contour = bean['contour']
                    # Gunakan ROI yang di-crop
                    roi = preprocessor.extract_bean_roi(img, bean['bbox'], padding=5)
                else:
                    # Fallback jika tidak terdeteksi contour (gunakan seluruh gambar)
                    h, w = img.shape[:2]
                    contour = np.array([[[0, 0]], [[0, h-1]], [[w-1, h-1]], [[w-1, 0]]])
                    roi = cv2.resize(img, preprocessor.target_size)
                
                if roi is not None:
                    # 2. Ekstrak all features
                    features = extractor.extract_all_features(roi, contour)
                    # Convert to flat vector
                    vec = extractor.features_to_vector(features)
                    
                    feature_vectors.append(vec)
                    true_labels.append(idx)
                    true_label_names.append(label_name)
                    image_paths.append(img_path)
                    
    print(f"Fitur berhasil diekstrak untuk {len(feature_vectors)} gambar.")
    return np.array(feature_vectors), np.array(true_labels), true_label_names, image_paths


def run_clustering_analysis(X, y_true, label_names, save_dir='models'):
    """
    Jalankan standarisasi, PCA, K-Means clustering, dan visualisasi.
    """
    if not HAS_SKLEARN:
        print("ERROR: scikit-learn diperlukan untuk clustering.")
        return
        
    os.makedirs(save_dir, exist_ok=True)
    num_classes = len(np.unique(y_true))
    
    # 1. Standarisasi Fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Reduksi Dimensi dengan PCA (2 Komponen utama)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained Variance Ratio PCA: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}")
    
    # 3. K-Means Clustering (Unsupervised)
    kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X_pca)
    
    # 4. Evaluasi Metrik Clustering
    sil_score = silhouette_score(X_pca, y_kmeans)
    ari_score = adjusted_rand_score(y_true, y_kmeans)
    
    print("\n" + "=" * 50)
    print("UNSUPERVISED CLUSTERING METRICS")
    print("=" * 50)
    print(f"Silhouette Score (K-Means): {sil_score:.4f}")
    print("  * Dekat 1: Cluster terpisah dengan baik.")
    print("  * Dekat 0: Cluster saling tumpang tindih.")
    print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
    print("  * Dekat 1: Cluster K-Means cocok sempurna dengan label defect (Supervised).")
    print("  * Dekat 0: Pengelompokan acak/tidak ada korelasi.")
    print("=" * 50 + "\n")
    
    # 5. Visualisasi Perbandingan (2 Panel)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Mapping warna untuk defect labels
    # Palette warna kustom premium
    color_palette = ['#2ECC71', '#E74C3C', '#E67E22', '#3498DB', '#9B59B6', '#F1C40F']
    if num_classes > len(color_palette):
        cmap = plt.get_cmap('tab10')
        colors_true = [cmap(i) for i in y_true]
        colors_kmeans = [cmap(i) for i in y_kmeans]
    else:
        colors_true = [color_palette[i] for i in y_true]
        colors_kmeans = [color_palette[i] for i in y_kmeans]
        
    # Scatter plot 1: True Labels (Supervised)
    ax1 = axes[0]
    for idx in range(num_classes):
        mask = y_true == idx
        if mask.sum() > 0:
            ax1.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                label=label_names[idx],
                color=color_palette[idx % len(color_palette)],
                alpha=0.8, edgecolors='w', s=60
            )
    ax1.set_title('Visualisasi Fitur Berdasarkan Label Cacat Asli\n(Supervised Classes)', fontsize=13, fontweight='bold')
    ax1.set_xlabel(f'Principal Component 1 ({explained_variance[0]:.1%})')
    ax1.set_ylabel(f'Principal Component 2 ({explained_variance[1]:.1%})')
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Scatter plot 2: K-Means Clusters (Unsupervised)
    ax2 = axes[1]
    for idx in range(num_classes):
        mask = y_kmeans == idx
        if mask.sum() > 0:
            ax2.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                label=f'Cluster {idx + 1}',
                alpha=0.8, edgecolors='w', s=60
            )
            
    # Plot centroid cluster
    centroids = kmeans.cluster_centers_
    ax2.scatter(
        centroids[:, 0], centroids[:, 1],
        marker='X', s=200, linewidths=2,
        color='black', label='Centroids'
    )
    
    ax2.set_title(f'Hasil Clustering K-Means Warna/Bentuk/Tekstur\n(Unsupervised - Sil Score: {sil_score:.3f}, ARI: {ari_score:.3f})', fontsize=13, fontweight='bold')
    ax2.set_xlabel(f'Principal Component 1 ({explained_variance[0]:.1%})')
    ax2.set_ylabel(f'Principal Component 2 ({explained_variance[1]:.1%})')
    ax2.legend(loc='best')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'clustering_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualisasi analisis clustering disimpan di: {plot_path}")
    
    # Save metrics data to JSON
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'silhouette_score': float(sil_score),
        'adjusted_rand_index': float(ari_score),
        'pca_explained_variance_ratio': [float(v) for v in explained_variance]
    }
    metrics_path = os.path.join(save_dir, 'clustering_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Data metrik clustering disimpan di: {metrics_path}")


def main():
    print("=" * 60)
    print("UNSUPERVISED CLUSTERING & PCA ANALYSIS")
    print("=" * 60)
    
    data_dir = Config.DATA_DIR if Config is not None else "data"
    save_dir = Config.MODELS_DIR if Config is not None else "models"
    
    X, y, label_names, _ = load_and_extract_features(data_dir)
    
    if len(X) == 0:
        print("ERROR: Tidak ada data untuk dianalisis.")
        return
        
    run_clustering_analysis(X, y, Config.GRADE_LABELS if Config is not None else label_names, save_dir)
    
    print("\nProses analisis clustering selesai dengan sukses!")
    print("=" * 60)


if __name__ == "__main__":
    main()
