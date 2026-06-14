"""
Konfigurasi terpusat untuk seluruh sistem grading kopi.
Semua parameter yang sebelumnya hardcoded dikumpulkan di sini
agar mudah di-tweak tanpa mengubah source code.
"""


class Config:
    """Konfigurasi global untuk Coffee Grading System"""
    
    # =============================================
    # MODEL
    # =============================================
    MODEL_INPUT_SHAPE = (224, 224, 3)
    MODEL_NUM_CLASSES = 6
    GRADE_LABELS = ['Normal', 'Biji Hitam', 'Biji Cokelat', 'Berlubang', 'Pecah', 'Berjamur']
    MODEL_SAVE_PATH = "models/coffee_grading_model.h5"
    MODEL_BEST_P1_PATH = "models/best_model_phase1.h5"
    MODEL_BEST_FT_PATH = "models/best_model_finetuned.h5"
    
    # Backbone: 'efficientnet' atau 'mobilenet' atau 'custom'
    BACKBONE = 'efficientnet'
    
    # =============================================
    # TRAINING - PHASE 1 (Classifier Head)
    # =============================================
    TRAIN_P1_EPOCHS = 50
    TRAIN_P1_BATCH_SIZE = 32
    TRAIN_P1_LEARNING_RATE = 1e-3
    TRAIN_P1_PATIENCE_EARLY_STOP = 10
    TRAIN_P1_PATIENCE_LR_REDUCE = 5
    
    # =============================================
    # TRAINING - PHASE 2 (Fine-tuning)
    # =============================================
    TRAIN_P2_EPOCHS = 30
    TRAIN_P2_BATCH_SIZE = 16
    TRAIN_P2_LEARNING_RATE = 1e-5
    TRAIN_P2_UNFREEZE_LAYERS = 20
    TRAIN_P2_PATIENCE_EARLY_STOP = 8
    TRAIN_P2_PATIENCE_LR_REDUCE = 3
    
    # =============================================
    # TRAINING - LEARNING RATE SCHEDULE
    # =============================================
    # 'reduce_on_plateau', 'cosine_annealing', 'cosine_warmup'
    LR_SCHEDULE = 'cosine_warmup'
    LR_WARMUP_EPOCHS = 5
    LR_MIN = 1e-7
    
    # =============================================
    # TRAINING - REGULARIZATION
    # =============================================
    LABEL_SMOOTHING = 0.1
    L2_REGULARIZATION = 0.01
    DROPOUT_DENSE_1 = 0.4
    DROPOUT_DENSE_2 = 0.3
    
    # =============================================
    # DATA AUGMENTATION
    # =============================================
    AUG_ROTATION_RANGE = 30
    AUG_WIDTH_SHIFT = 0.15
    AUG_HEIGHT_SHIFT = 0.15
    AUG_SHEAR_RANGE = 0.15
    AUG_ZOOM_RANGE = 0.2
    AUG_HORIZONTAL_FLIP = True
    AUG_VERTICAL_FLIP = True
    AUG_BRIGHTNESS_RANGE = (0.7, 1.3)
    AUG_CHANNEL_SHIFT = 20
    
    # Mixup augmentation
    MIXUP_ENABLED = True
    MIXUP_ALPHA = 0.2  # Beta distribution alpha parameter
    
    # =============================================
    # DATA
    # =============================================
    DATA_DIR = "data"
    TRAIN_TEST_SPLIT = 0.2
    RANDOM_SEED = 42
    SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    
    # =============================================
    # PREPROCESSING / COMPUTER VISION
    # =============================================
    TARGET_SIZE = (224, 224)
    
    # CLAHE
    CLAHE_CLIP_LIMIT = 3.0
    CLAHE_TILE_SIZE = (8, 8)
    
    # Bean detection
    BEAN_MIN_AREA = 500
    BEAN_MAX_AREA = 50000
    BEAN_MIN_ASPECT_RATIO = 0.4
    BEAN_MAX_ASPECT_RATIO = 2.5
    BEAN_MIN_SOLIDITY = 0.6
    BEAN_MIN_CIRCULARITY = 0.3
    
    # NMS
    NMS_IOU_THRESHOLD = 0.3
    
    # =============================================
    # IMAGE QUALITY
    # =============================================
    QUALITY_CHECK_ENABLED = True
    QUALITY_MIN_BRIGHTNESS = 30      # Minimum mean brightness (0-255)
    QUALITY_MAX_BRIGHTNESS = 240     # Maximum mean brightness
    QUALITY_MIN_SHARPNESS = 50.0     # Minimum Laplacian variance
    QUALITY_MIN_CONTRAST = 20.0      # Minimum std deviation
    
    # =============================================
    # REALTIME INFERENCE
    # =============================================
    CONFIDENCE_THRESHOLD = 0.6
    SMOOTHER_WINDOW_SIZE = 7
    SMOOTHER_ALPHA = 0.6
    FPS_AVG_WINDOW = 30
    
    # Camera
    CAMERA_ID = 0
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    
    # Auto White Balance
    AWB_ENABLED = True
    
    # =============================================
    # FEATURE EXTRACTION
    # =============================================
    # GLCM
    GLCM_DISTANCES = [1, 2]
    GLCM_LEVELS = 16
    
    # LBP
    LBP_RADIUS = 2
    
    # =============================================
    # GRAD-CAM
    # =============================================
    GRADCAM_ENABLED = True
    GRADCAM_LAYER_NAME = None  # None = auto-detect last conv layer
    GRADCAM_ALPHA = 0.4        # Overlay transparency
    
    # =============================================
    # OUTPUT
    # =============================================
    MODELS_DIR = "models"
    SCREENSHOTS_DIR = "screenshots"
    BATCH_OUTPUT_DIR = "results"
