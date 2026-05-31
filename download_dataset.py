"""
Script untuk download dataset kopi dari internet
Menggunakan berbagai sumber dataset publik
"""
import os
import urllib.request
import zipfile
import json


def download_file(url, destination):
    """Download file dari URL"""
    print(f"Downloading from {url}...")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"✓ Downloaded to {destination}")
        return True
    except Exception as e:
        print(f"✗ Error downloading: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"✗ Error extracting: {e}")
        return False


def main():
    """Main function"""
    print("=" * 70)
    print("DOWNLOAD DATASET KOPI")
    print("=" * 70)
    print()
    
    print("SUMBER DATASET YANG BISA DIGUNAKAN:")
    print()
    print("1. Kaggle - Coffee Bean Dataset")
    print("   URL: https://www.kaggle.com/datasets")
    print("   Search: 'coffee bean classification' atau 'coffee quality'")
    print()
    print("2. Roboflow - Coffee Bean Dataset")
    print("   URL: https://universe.roboflow.com/")
    print("   Search: 'coffee bean' atau 'coffee grading'")
    print()
    print("3. Google Images - Manual Download")
    print("   Search: 'coffee bean grade A', 'coffee bean defects', dll")
    print()
    print("4. GitHub - Public Datasets")
    print("   Search: 'coffee bean dataset' di GitHub")
    print()
    
    print("=" * 70)
    print("CARA DOWNLOAD DARI KAGGLE:")
    print("=" * 70)
    print()
    print("1. Install Kaggle CLI:")
    print("   pip install kaggle")
    print()
    print("2. Setup Kaggle API:")
    print("   - Login ke kaggle.com")
    print("   - Go to Account > API > Create New API Token")
    print("   - Download kaggle.json")
    print("   - Letakkan di: ~/.kaggle/kaggle.json (Linux/Mac)")
    print("                  C:\\Users\\<username>\\.kaggle\\kaggle.json (Windows)")
    print()
    print("3. Download dataset:")
    print("   kaggle datasets download -d <dataset-name>")
    print()
    print("   Contoh:")
    print("   kaggle datasets download -d vencerlanz09/coffee-bean-classification")
    print()
    
    print("=" * 70)
    print("CARA DOWNLOAD DARI ROBOFLOW:")
    print("=" * 70)
    print()
    print("1. Buka: https://universe.roboflow.com/")
    print("2. Search: 'coffee bean'")
    print("3. Pilih dataset yang sesuai")
    print("4. Klik 'Download' > Pilih format 'Folder Structure'")
    print("5. Extract ke folder 'data/'")
    print()
    
    print("=" * 70)
    print("ALTERNATIF: GUNAKAN SCRIPT collect_data.py")
    print("=" * 70)
    print()
    print("Jika tidak ada dataset publik yang sesuai,")
    print("gunakan script collect_data.py untuk capture sendiri:")
    print()
    print("   python collect_data.py")
    print()
    print("Keuntungan:")
    print("- Data sesuai dengan kondisi real Anda")
    print("- Pencahayaan dan background konsisten")
    print("- Lebih akurat untuk use case spesifik")
    print()
    
    print("=" * 70)
    print("DATASET SAMPLE UNTUK TESTING")
    print("=" * 70)
    print()
    print("Untuk testing awal, Anda bisa:")
    print("1. Download 10-20 gambar biji kopi dari Google Images")
    print("2. Klasifikasikan manual ke grade A/B/C")
    print("3. Letakkan di folder data/grade_a, data/grade_b, data/grade_c")
    print("4. Train model untuk testing")
    print("5. Collect data lebih banyak untuk production")
    print()


if __name__ == "__main__":
    main()
