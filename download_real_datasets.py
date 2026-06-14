import os
import sys
import shutil

# Pastikan working directory selalu di folder script ini
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def setup_kaggle():
    print("\n" + "="*50)
    print(" 1. Setup Kaggle Dataset (Untuk Model EfficientNet)")
    print("="*50)
    
    # Menulis token akses kaggle
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    with open(os.path.join(kaggle_dir, "access_token"), "w") as f:
        f.write("KGAT_33f13bdd32c286f01715ace5e9593e36")
    
    os.environ["KAGGLE_API_TOKEN"] = "KGAT_33f13bdd32c286f01715ace5e9593e36"

    try:
        import kaggle
        print("Mengautentikasi Kaggle...")
        kaggle.api.authenticate()
        
        target_dir = "data_raw/kaggle_17_defects"
        os.makedirs(target_dir, exist_ok=True)
        print("Mengunduh dataset Kaggle... (Mungkin butuh waktu beberapa menit)")
        kaggle.api.dataset_download_files(
            'sujitraarw/coffee-green-bean-with-17-defects-original',
            path=target_dir, 
            unzip=True
        )
        print(f"[OK] Kaggle dataset berhasil diunduh ke folder '{target_dir}'.")
    except Exception as e:
        print(f"[ERROR] Gagal mengunduh dataset Kaggle: {e}")

def setup_roboflow():
    print("\n" + "="*50)
    print(" 2. Setup Roboflow Dataset (Untuk Model YOLOv8)")
    print("="*50)
    
    api_key = "favqW2UcABSKEJHlBMGu"

    try:
        from roboflow import Roboflow
        print("Mengautentikasi Roboflow...")
        rf = Roboflow(api_key=api_key)
        
        project = rf.workspace("roasted-coffee-bean-defect-detectionandy").project("coffee-bean-defect")
        version = project.version(1)
        
        target_dir = "yolo_poc/dataset_real"
        print(f"Mengunduh dataset Roboflow (YOLOv8 format)...")
        current_dir = os.getcwd()
        os.makedirs(target_dir, exist_ok=True)
        os.chdir(target_dir)
        
        dataset = version.download("yolov8")
        
        os.chdir(current_dir)
        print(f"[OK] Roboflow dataset berhasil diunduh ke folder '{target_dir}/{dataset.name}'.")
    except Exception as e:
        print(f"[ERROR] Gagal mengunduh dataset Roboflow: {e}")
        if "current_dir" in locals():
            os.chdir(current_dir)

def main():
    print("============================================================")
    print("  Smart Grading Kopi - Real Datasets Downloader")
    print("============================================================")
    
    setup_kaggle()
    setup_roboflow()
    
    print("\nSelesai! Silakan cek folder 'data_raw' dan 'yolo_poc/dataset_real'.")

if __name__ == "__main__":
    main()
