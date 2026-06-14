"""
Script untuk training model grading kopi
- Data augmentation komprehensif
- Class weighting otomatis untuk imbalanced data
- 2-phase training: classifier head + fine-tuning
- Evaluasi lengkap: confusion matrix, classification report, training curves
"""
import sys
import os
import numpy as np
import cv2
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import CoffeeGradingModel
from preprocessing import ImagePreprocessor
try:
    from config import Config
except ImportError:
    Config = None

# Try import seaborn
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("WARNING: seaborn tidak terinstall, confusion matrix heatmap akan menggunakan matplotlib.")


def load_dataset(data_dir):
    """
    Load dataset dari folder structure yang dinamis berdasarkan Config.GRADE_LABELS.
    Nama folder = label yang di-lowercase dan spasi diganti underscore.
    Contoh untuk 6 kelas defect:
        data/
          normal/
          biji_hitam/
          biji_cokelat/
          berlubang/
          pecah/
          berjamur/
    
    Args:
        data_dir: Path ke data directory
        
    Returns:
        tuple: (images, labels)
    """
    images = []
    labels = []
    
    # Dynamic mapping dari Config
    grade_labels = Config.GRADE_LABELS if Config is not None else ['Grade A', 'Grade B', 'Grade C']
    grade_mapping = {}
    for idx, label_name in enumerate(grade_labels):
        folder_name = label_name.lower().replace(' ', '_')
        grade_mapping[folder_name] = idx
    
    preprocessor = ImagePreprocessor()
    supported_formats = Config.SUPPORTED_FORMATS if Config is not None else ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    
    for grade_folder, label in grade_mapping.items():
        folder_path = os.path.join(data_dir, grade_folder)
        
        if not os.path.exists(folder_path):
            print(f"WARNING: Folder {folder_path} tidak ditemukan")
            continue
        
        file_count = 0
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(supported_formats):
                img_path = os.path.join(folder_path, filename)
                
                # Load image
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Resize ke target size
                    img_resized = cv2.resize(
                        img, preprocessor.target_size,
                        interpolation=cv2.INTER_LANCZOS4
                    )
                    
                    # Normalize (EfficientNet preprocessing: [-1, 1])
                    img_normalized = preprocessor.normalize_image(img_resized)
                    
                    images.append(img_normalized)
                    labels.append(label)
                    file_count += 1
        
        print(f"  {grade_folder}: {file_count} images loaded")
    
    print(f"Total images loaded: {len(images)}")
    
    return np.array(images), np.array(labels)


def create_data_augmentation():
    """
    Buat data augmentation pipeline yang komprehensif
    Termasuk rotasi, flip, zoom, brightness, dan shift
    
    Returns:
        ImageDataGenerator: Augmentation generator
    """
    if Config is not None:
        train_datagen = ImageDataGenerator(
            rotation_range=Config.AUG_ROTATION_RANGE,
            width_shift_range=Config.AUG_WIDTH_SHIFT,
            height_shift_range=Config.AUG_HEIGHT_SHIFT,
            shear_range=Config.AUG_SHEAR_RANGE,
            zoom_range=Config.AUG_ZOOM_RANGE,
            horizontal_flip=Config.AUG_HORIZONTAL_FLIP,
            vertical_flip=Config.AUG_VERTICAL_FLIP,
            brightness_range=Config.AUG_BRIGHTNESS_RANGE,
            channel_shift_range=Config.AUG_CHANNEL_SHIFT,
            fill_mode='reflect',
        )
    else:
        train_datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.7, 1.3],
            channel_shift_range=20,
            fill_mode='reflect',
        )
    
    return train_datagen


def compute_class_weights(y):
    """
    Hitung class weights untuk handle data yang tidak seimbang
    
    Args:
        y: Array of labels (bukan one-hot)
        
    Returns:
        dict: Class weights {class_index: weight}
    """
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    
    grade_names = Config.GRADE_LABELS if Config is not None else ['Grade A', 'Grade B', 'Grade C']
    print("Class weights:")
    for c, w in class_weight_dict.items():
        name = grade_names[c] if c < len(grade_names) else f'Class {c}'
        count = np.sum(y == c)
        print(f"  {name}: {w:.3f} (n={count})")
    
    return class_weight_dict


def plot_training_history(history, history_ft=None, save_dir='models'):
    """
    Plot training history (accuracy dan loss curves)
    
    Args:
        history: History dari phase 1 training
        history_ft: History dari phase 2 fine-tuning (optional)
        save_dir: Directory untuk menyimpan plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Accuracy Plot ---
    ax1 = axes[0]
    acc = history.history['accuracy']
    epochs_range = range(1, len(acc) + 1)
    
    ax1.plot(epochs_range, acc, 'b-', label='Train Accuracy (P1)', linewidth=2)
    
    if 'val_accuracy' in history.history:
        val_acc = history.history['val_accuracy']
        ax1.plot(epochs_range, val_acc, 'b--', label='Val Accuracy (P1)', linewidth=2)
    
    offset = len(acc)
    
    if history_ft is not None:
        acc_ft = history_ft.history['accuracy']
        ft_range = range(offset + 1, offset + len(acc_ft) + 1)
        ax1.plot(ft_range, acc_ft, 'r-', label='Train Accuracy (P2)', linewidth=2)
        
        if 'val_accuracy' in history_ft.history:
            val_acc_ft = history_ft.history['val_accuracy']
            ax1.plot(ft_range, val_acc_ft, 'r--', label='Val Accuracy (P2)', linewidth=2)
        
        ax1.axvline(x=offset + 0.5, color='gray', linestyle=':', label='Fine-tune start')
    
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # --- Loss Plot ---
    ax2 = axes[1]
    loss = history.history['loss']
    
    ax2.plot(epochs_range, loss, 'b-', label='Train Loss (P1)', linewidth=2)
    
    if 'val_loss' in history.history:
        val_loss = history.history['val_loss']
        ax2.plot(epochs_range, val_loss, 'b--', label='Val Loss (P1)', linewidth=2)
    
    if history_ft is not None:
        loss_ft = history_ft.history['loss']
        ft_range = range(offset + 1, offset + len(loss_ft) + 1)
        ax2.plot(ft_range, loss_ft, 'r-', label='Train Loss (P2)', linewidth=2)
        
        if 'val_loss' in history_ft.history:
            val_loss_ft = history_ft.history['val_loss']
            ax2.plot(ft_range, val_loss_ft, 'r--', label='Val Loss (P2)', linewidth=2)
        
        ax2.axvline(x=offset + 0.5, color='gray', linestyle=':', label='Fine-tune start')
    
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_dir}/training_curves.png")


def plot_confusion_matrix(y_true, y_pred, save_dir='models'):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels (class indices)
        y_pred: Predicted labels (class indices)
        save_dir: Directory untuk menyimpan plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    grade_names = Config.GRADE_LABELS if Config is not None else ['Grade A', 'Grade B', 'Grade C']
    cm = confusion_matrix(y_true, y_pred)
    
    n_classes = len(grade_names)
    fig_size = max(8, n_classes * 1.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.75))
    
    if HAS_SEABORN:
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=grade_names, yticklabels=grade_names,
            ax=ax, square=True, linewidths=1, linecolor='white',
            annot_kws={'size': 12 if n_classes <= 4 else 10}
        )
    else:
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax)
        
        font_size = 14 if n_classes <= 4 else 10
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                       ha='center', va='center',
                       fontsize=font_size, fontweight='bold',
                       color='white' if cm[i, j] > cm.max() / 2 else 'black')
        
        ax.set_xticks(range(len(grade_names)))
        ax.set_yticks(range(len(grade_names)))
        ax.set_xticklabels(grade_names, rotation=45, ha='right')
        ax.set_yticklabels(grade_names)
    
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_dir}/confusion_matrix.png")


def evaluate_model(model, X_val, y_val, save_dir='models'):
    """
    Evaluasi komprehensif: confusion matrix, classification report, per-class metrics
    
    Args:
        model: Trained CoffeeGradingModel
        X_val: Validation images
        y_val: Validation labels (one-hot)
        save_dir: Directory untuk menyimpan hasil
    """
    os.makedirs(save_dir, exist_ok=True)
    
    grade_names = Config.GRADE_LABELS if Config is not None else ['Grade A', 'Grade B', 'Grade C']
    
    # Predict
    y_pred_probs = model.model.predict(X_val, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)
    
    # Classification report
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    report = classification_report(
        y_true_classes, y_pred_classes,
        target_names=grade_names,
        digits=4
    )
    print(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true_classes, y_pred_classes, save_dir)
    
    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for i, name in enumerate(grade_names):
        mask = y_true_classes == i
        if mask.sum() > 0:
            class_acc = (y_pred_classes[mask] == i).mean()
            print(f"  {name}: {class_acc:.4f} ({mask.sum()} samples)")
    
    # Save metrics to JSON
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'classification_report': classification_report(
            y_true_classes, y_pred_classes,
            target_names=grade_names,
            output_dict=True
        ),
        'confusion_matrix': confusion_matrix(y_true_classes, y_pred_classes).tolist(),
    }
    
    metrics_path = os.path.join(save_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    return metrics


def main():
    """Main training function"""
    print("=" * 60)
    print("TRAINING MODEL GRADING KOPI")
    print(f"Backbone: {Config.BACKBONE if Config is not None else 'efficientnet'} + Fine-tuning + Data Augmentation")
    print("=" * 60)
    print()
    
    # Load dataset
    data_dir = Config.DATA_DIR if Config is not None else "data"
    num_classes = Config.MODEL_NUM_CLASSES if Config is not None else 3
    grade_labels = Config.GRADE_LABELS if Config is not None else ['Grade A', 'Grade B', 'Grade C']
    
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory '{data_dir}' tidak ditemukan")
        print("\nBuat struktur folder berikut:")
        print(f"{data_dir}/")
        for label_name in grade_labels:
            folder_name = label_name.lower().replace(' ', '_')
            print(f"  {folder_name}/  <- Letakkan gambar {label_name} di sini")
        return
    
    print("Loading dataset...")
    X, y = load_dataset(data_dir)
    
    if len(X) == 0:
        print("Dataset kosong. Mencoba menghasilkan gambar sintetis untuk training...")
        try:
            from analyze_dataset import generate_synthetic_dataset
            generate_synthetic_dataset(data_dir, num_samples_per_class=25)
            # Reload dataset
            X, y = load_dataset(data_dir)
        except Exception as e:
            print(f"Gagal menghasilkan dataset sintetis: {e}")
            
    if len(X) == 0:
        print("ERROR: Tidak ada data untuk training")
        print("Letakkan gambar biji kopi di sub-folder berikut:")
        for label_name in grade_labels:
            folder_name = label_name.lower().replace(' ', '_')
            print(f"  {data_dir}/{folder_name}/")
        return
    
    if len(X) < 10:
        print(f"WARNING: Hanya {len(X)} images. Minimal direkomendasikan 100 images per kelas.")
    
    # Compute class weights
    print("\nComputing class weights...")
    class_weights = compute_class_weights(y)
    
    # Convert labels ke one-hot encoding
    y_categorical = to_categorical(y, num_classes=num_classes)
    
    # Split dataset
    test_size = Config.TRAIN_TEST_SPLIT if Config is not None else 0.2
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create data augmentation
    print("\nCreating data augmentation pipeline...")
    train_datagen = create_data_augmentation()
    
    # Initialize model
    os.makedirs("models", exist_ok=True)
    model = CoffeeGradingModel()
    
    # Show model summary
    print("\nModel Architecture:")
    model.get_model_summary()
    
    # Hyperparameters from config
    epochs_p1 = Config.TRAIN_P1_EPOCHS if Config is not None else 50
    batch_size_p1 = Config.TRAIN_P1_BATCH_SIZE if Config is not None else 32
    
    epochs_p2 = Config.TRAIN_P2_EPOCHS if Config is not None else 30
    batch_size_p2 = Config.TRAIN_P2_BATCH_SIZE if Config is not None else 16
    n_layers_unfreeze = Config.TRAIN_P2_UNFREEZE_LAYERS if Config is not None else 20
    
    model_path = Config.MODEL_SAVE_PATH if Config is not None else "models/coffee_grading_model.h5"
    
    # =============================================
    # Phase 1: Train classifier head
    # =============================================
    print("\n\nMemulai Phase 1: Training classifier head...")
    history_p1 = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs_p1,
        batch_size=batch_size_p1,
        class_weights=class_weights,
        data_augmentation=train_datagen
    )
    
    # =============================================
    # Phase 2: Fine-tuning
    # =============================================
    print("\n\nMemulai Phase 2: Fine-tuning top layers...")
    history_p2 = model.fine_tune(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs_p2,
        batch_size=batch_size_p2,
        n_layers_unfreeze=n_layers_unfreeze,
        class_weights=class_weights,
        data_augmentation=train_datagen
    )
    
    # =============================================
    # Save model
    # =============================================
    model.save_model(model_path)
    
    # =============================================
    # Plot training curves
    # =============================================
    print("\nGenerating training plots...")
    plot_training_history(history_p1, history_p2, save_dir='models')
    
    # =============================================
    # Comprehensive evaluation
    # =============================================
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_val, y_val, save_dir='models')
    
    # =============================================
    # Final summary
    # =============================================
    print("\n" + "=" * 60)
    print("TRAINING SELESAI")
    print("=" * 60)
    print(f"Model disimpan di: {model_path}")
    
    # Print final metrics
    if history_p1 is not None:
        final_train_acc = history_p1.history['accuracy'][-1]
        print(f"\nPhase 1 Final Training Accuracy: {final_train_acc:.4f}")
        if 'val_accuracy' in history_p1.history:
            final_val_acc = history_p1.history['val_accuracy'][-1]
            print(f"Phase 1 Final Validation Accuracy: {final_val_acc:.4f}")
    
    if history_p2 is not None:
        final_train_acc_ft = history_p2.history['accuracy'][-1]
        print(f"\nPhase 2 Final Training Accuracy: {final_train_acc_ft:.4f}")
        if 'val_accuracy' in history_p2.history:
            final_val_acc_ft = history_p2.history['val_accuracy'][-1]
            print(f"Phase 2 Final Validation Accuracy: {final_val_acc_ft:.4f}")
    
    print("\nFiles yang dihasilkan:")
    print(f"  - {model_path}")
    print(f"  - models/training_curves.png")
    print(f"  - models/confusion_matrix.png")
    print(f"  - models/evaluation_metrics.json")
    print(f"  - models/best_model_phase1.h5")
    print(f"  - models/best_model_finetuned.h5")


if __name__ == "__main__":
    main()
