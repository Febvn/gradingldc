"""
Script untuk augmentasi data khusus biji kopi hijau
Karena biji hijau kemungkinan data-nya sedikit, script ini akan
generate synthetic green bean images dengan color augmentation
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Pastikan working directory selalu di folder script ini
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def simulate_green_bean(normal_image, green_intensity=0.3):
    """
    Simulasi biji kopi hijau dari biji normal dengan color transformation
    
    Args:
        normal_image: Image biji kopi normal
        green_intensity: Intensitas efek hijau (0.0-1.0)
        
    Returns:
        numpy.ndarray: Simulated green bean image
    """
    # Convert ke HSV
    hsv = cv2.cvtColor(normal_image, cv2.COLOR_BGR2HSV)
    
    # Shift Hue ke arah hijau (40-80 degrees)
    # Original coffee brown adalah sekitar 10-30 degrees
    hue_shift = int(green_intensity * 50)  # Shift 0-50 degrees
    hsv[:, :, 0] = np.clip(hsv[:, :, 0].astype(np.int16) + hue_shift, 0, 179).astype(np.uint8)
    
    # Increase saturation untuk warna lebih vivid
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * (1.0 + green_intensity * 0.5), 0, 255).astype(np.uint8)
    
    # Slightly increase brightness
    hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.float32) * (1.0 + green_intensity * 0.2), 0, 255).astype(np.uint8)
    
    # Convert back to BGR
    green_bean = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return green_bean


def augment_green_beans(source_dir, output_dir, num_variants=10):
    """
    Generate augmented green bean images
    
    Args:
        source_dir: Directory dengan sample biji hijau (atau biji normal untuk simulasi)
        output_dir: Directory output untuk augmented images
        num_variants: Jumlah varian per image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported formats
    formats = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    
    image_files = []
    for fmt in formats:
        image_files.extend(Path(source_dir).glob(f'*{fmt}'))
    
    if len(image_files) == 0:
        print(f"Tidak ada image ditemukan di {source_dir}")
        return
    
    print(f"Ditemukan {len(image_files)} images untuk augmentasi")
    
    count = 0
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        base_name = img_path.stem
        
        for i in range(num_variants):
            # Random green intensity
            green_intensity = np.random.uniform(0.2, 0.6)
            
            # Simulate green bean
            augmented = simulate_green_bean(img, green_intensity)
            
            # Additional random augmentations
            # 1. Random rotation
            if np.random.random() > 0.5:
                angle = np.random.uniform(-30, 30)
                h, w = augmented.shape[:2]
                M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
                augmented = cv2.warpAffine(augmented, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            # 2. Random brightness
            if np.random.random() > 0.5:
                brightness = np.random.uniform(0.8, 1.2)
                augmented = np.clip(augmented.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
            
            # 3. Random flip
            if np.random.random() > 0.5:
                augmented = cv2.flip(augmented, 1)
            
            # 4. Random noise
            if np.random.random() > 0.7:
                noise = np.random.normal(0, 5, augmented.shape).astype(np.int16)
                augmented = np.clip(augmented.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Save
            output_name = f"{base_name}_aug_{i:03d}.jpg"
            output_path = os.path.join(output_dir, output_name)
            cv2.imwrite(output_path, augmented, [cv2.IMWRITE_JPEG_QUALITY, 95])
            count += 1
        
        print(f"Processed: {img_path.name} -> {num_variants} variants")
    
    print(f"\nTotal {count} augmented images generated in {output_dir}")


def create_mixed_green_samples(normal_dir, output_dir, num_samples=50):
    """
    Create biji kopi dengan campuran area normal dan hijau (partial green)
    Ini simulasi biji yang setengah matang
    
    Args:
        normal_dir: Directory biji normal
        output_dir: Directory output
        num_samples: Jumlah sample yang mau dibuat
    """
    os.makedirs(output_dir, exist_ok=True)
    
    formats = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    image_files = []
    for fmt in formats:
        image_files.extend(Path(normal_dir).glob(f'*{fmt}'))
    
    if len(image_files) == 0:
        print(f"Tidak ada image ditemukan di {normal_dir}")
        return
    
    print(f"Creating {num_samples} mixed green samples...")
    
    for i in range(num_samples):
        # Random pilih source image
        img_path = np.random.choice(image_files)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Create green version
        green_intensity = np.random.uniform(0.3, 0.7)
        green_img = simulate_green_bean(img, green_intensity)
        
        # Create mask untuk blend (random gradient atau patches)
        mask = np.zeros((h, w), dtype=np.float32)
        
        if np.random.random() > 0.5:
            # Gradient mask (horizontal atau vertical)
            if np.random.random() > 0.5:
                # Horizontal gradient
                mask = np.linspace(0, 1, w).reshape(1, w)
                mask = np.repeat(mask, h, axis=0)
            else:
                # Vertical gradient
                mask = np.linspace(0, 1, h).reshape(h, 1)
                mask = np.repeat(mask, w, axis=1)
        else:
            # Random patches
            num_patches = np.random.randint(2, 5)
            for _ in range(num_patches):
                cx = np.random.randint(0, w)
                cy = np.random.randint(0, h)
                radius = np.random.randint(h//4, h//2)
                cv2.circle(mask, (cx, cy), radius, 1.0, -1)
        
        # Smooth mask
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask = np.clip(mask, 0, 1)
        
        # Blend
        mask_3ch = np.stack([mask, mask, mask], axis=2)
        blended = (img.astype(np.float32) * (1 - mask_3ch) + 
                   green_img.astype(np.float32) * mask_3ch)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        # Save
        output_name = f"mixed_green_{i:04d}.jpg"
        output_path = os.path.join(output_dir, output_name)
        cv2.imwrite(output_path, blended, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    print(f"Generated {num_samples} mixed green samples in {output_dir}")


def main():
    """Main function"""
    print("=" * 60)
    print("AUGMENTASI DATA BIJI KOPI HIJAU")
    print("=" * 60)
    print()
    
    # Opsi 1: Augmentasi dari existing biji hijau
    source_green = "data/biji_hijau"
    if os.path.exists(source_green):
        print("Augmentasi existing biji hijau...")
        augment_green_beans(
            source_dir=source_green,
            output_dir="data/biji_hijau_augmented",
            num_variants=10
        )
    else:
        print(f"Folder {source_green} tidak ditemukan, skip augmentasi existing")
    
    print()
    
    # Opsi 2: Simulasi biji hijau dari biji normal
    # Cari source: data/normal, atau fallback ke folder Kaggle
    source_normal = None
    for candidate in ["data/normal", "data_raw/kaggle_17_defects/Normal"]:
        if os.path.exists(candidate) and any(
            Path(candidate).glob("*.jpg")
        ):
            source_normal = candidate
            break

    if source_normal:
        print(f"Simulasi biji hijau dari '{source_normal}'...")
        augment_green_beans(
            source_dir=source_normal,
            output_dir="data/biji_hijau_simulated",
            num_variants=5
        )

        print()
        print("Membuat sampel mixed green (setengah matang)...")
        create_mixed_green_samples(
            normal_dir=source_normal,
            output_dir="data/biji_hijau_mixed",
            num_samples=30
        )
    else:
        print("Folder data/normal dan data_raw/kaggle_17_defects/Normal tidak ditemukan.")
        print("Jalankan dulu: python download_real_datasets.py")
    
    print()
    print("=" * 60)
    print("SELESAI")
    print("=" * 60)
    print("\nCara pakai hasil augmentasi:")
    print("1. Review images di folder output")
    print("2. Copy yang bagus ke 'data/biji_hijau/' untuk training")
    print("3. Hapus yang tidak sesuai")
    print("4. Run 'python train_model.py' untuk train dengan data baru")


if __name__ == "__main__":
    main()
