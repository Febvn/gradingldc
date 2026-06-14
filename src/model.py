"""
Modul untuk model machine learning klasifikasi kopi
- EfficientNetB0 sebagai backbone (transfer learning)
- Fine-tuning strategy (2-phase training)
- Label smoothing untuk mengurangi overconfidence
- Test-Time Augmentation (TTA) untuk prediksi lebih robust
"""
import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import os
import math

try:
    from config import Config
except ImportError:
    Config = None


class CosineWarmupDecayScheduler(keras.callbacks.Callback):
    """
    Callback untuk scheduling learning rate menggunakan Cosine Annealing dengan Warmup.
    """
    def __init__(self, total_epochs, warmup_epochs, lr_max, lr_min=1e-7, verbose=0):
        super(CosineWarmupDecayScheduler, self).__init__()
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr") and not hasattr(self.model.optimizer, "learning_rate"):
            raise ValueError('Optimizer must have a "learning_rate" or "lr" attribute.')
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = (self.lr_max - self.lr_min) * (epoch + 1) / self.warmup_epochs + self.lr_min
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + math.cos(math.pi * progress))
            
        if hasattr(self.model.optimizer, "learning_rate"):
            try:
                keras.backend.set_value(self.model.optimizer.learning_rate, lr)
            except:
                self.model.optimizer.learning_rate = lr
        elif hasattr(self.model.optimizer, "lr"):
            try:
                keras.backend.set_value(self.model.optimizer.lr, lr)
            except:
                self.model.optimizer.lr = lr
            
        if self.verbose > 0:
            print(f'\nEpoch {epoch+1}: CosineWarmupDecayScheduler setting learning rate to {lr:.3e}.')


def mixup_generator(generator, alpha=0.2):
    """
    Wrap an ImageDataGenerator flow to perform Mixup data augmentation.
    """
    while True:
        # Get a batch
        X1, y1 = next(generator)
        # Get another batch
        X2, y2 = next(generator)
        
        batch_size = min(len(X1), len(X2))
        if batch_size == 0:
            continue
            
        X1, y1 = X1[:batch_size], y1[:batch_size]
        X2, y2 = X2[:batch_size], y2[:batch_size]
        
        # Sample lambda from Beta distribution
        if alpha > 0:
            l = np.random.beta(alpha, alpha, batch_size)
        else:
            l = np.ones(batch_size)
            
        # Reshape lambda for broadcasting
        l_x = l.reshape(batch_size, 1, 1, 1)
        l_y = l.reshape(batch_size, 1)
        
        # Mix images and labels
        X_mix = l_x * X1 + (1.0 - l_x) * X2
        y_mix = l_y * y1 + (1.0 - l_y) * y2
        
        yield X_mix, y_mix


class CoffeeGradingModel:
    def __init__(self, model_path=None):
        """
        Initialize model
        
        Args:
            model_path: Path ke saved model (optional)
        """
        self.model = None
        self.model_path = model_path
        
        if Config is not None:
            self.grade_labels = Config.GRADE_LABELS
            self.input_shape = Config.MODEL_INPUT_SHAPE
        else:
            self.grade_labels = ['Grade A', 'Grade B', 'Grade C']
            self.input_shape = (224, 224, 3)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()
    
    def build_model(self, use_pretrained=True):
        """
        Build model untuk klasifikasi
        
        Args:
            use_pretrained: Gunakan transfer learning dengan pretrained model
        """
        backbone = 'efficientnet'
        if Config is not None:
            backbone = Config.BACKBONE
            
        if use_pretrained:
            if backbone == 'efficientnet' or backbone == 'efficientnetb0':
                self._build_efficientnet_model(version='b0')
            elif backbone == 'efficientnetb3':
                self._build_efficientnet_model(version='b3')
            elif backbone == 'mobilenet':
                self._build_mobilenet_model()
            else:
                self._build_efficientnet_model(version='b0')
        else:
            self._build_custom_cnn_model()
        
        return self.model
    
    def _build_efficientnet_model(self, version='b0'):
        """
        Build model dengan EfficientNet sebagai backbone
        EfficientNetB3 lebih kuat dari B0 untuk dataset kompleks
        """
        # include_preprocessing=False: hindari konflik layers.Normalization
        # dengan DirectML / GPU kernel pada TF 2.10. Input tetap di-normalize
        # oleh preprocessing pipeline sebelum masuk model.
        try:
            preproc_kwargs = dict(include_preprocessing=False)
            if version == 'b3':
                base_model = keras.applications.EfficientNetB3(
                    input_shape=self.input_shape, include_top=False,
                    weights='imagenet', **preproc_kwargs)
                model_name = "EfficientNetB3"
            else:
                base_model = keras.applications.EfficientNetB0(
                    input_shape=self.input_shape, include_top=False,
                    weights='imagenet', **preproc_kwargs)
                model_name = "EfficientNetB0"
        except TypeError:
            # TF versi lama tidak mendukung include_preprocessing
            if version == 'b3':
                base_model = keras.applications.EfficientNetB3(
                    input_shape=self.input_shape, include_top=False,
                    weights='imagenet')
                model_name = "EfficientNetB3"
            else:
                base_model = keras.applications.EfficientNetB0(
                    input_shape=self.input_shape, include_top=False,
                    weights='imagenet')
                model_name = "EfficientNetB0"
        
        # Phase 1: Freeze base model layers
        base_model.trainable = False
        
        # Build classifier head dengan Functional API
        inputs = keras.Input(shape=self.input_shape)
        
        # Base model
        x = base_model(inputs, training=False)
        
        # Classifier head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        
        l2_reg = Config.L2_REGULARIZATION if Config is not None else 0.01
        dropout_1 = Config.DROPOUT_DENSE_1 if Config is not None else 0.4
        dropout_2 = Config.DROPOUT_DENSE_2 if Config is not None else 0.3
        label_smooth = Config.LABEL_SMOOTHING if Config is not None else 0.1
        lr_p1 = Config.TRAIN_P1_LEARNING_RATE if Config is not None else 1e-3
        
        x = layers.Dense(
            256, activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
        x = layers.Dropout(dropout_1)(x)
        x = layers.Dense(
            128, activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
        x = layers.Dropout(dropout_2)(x)
        num_classes = Config.MODEL_NUM_CLASSES if Config is not None else 3
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile dengan label smoothing
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_p1),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smooth),
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        self.model = model
        self._base_model = base_model
        print(f"Model dengan Transfer Learning ({model_name}) berhasil dibuat")
        print(f"  Total params: {model.count_params():,}")

    def _build_mobilenet_model(self):
        """
        Build model dengan MobileNetV2 sebagai backbone
        """
        # Base model: MobileNetV2
        base_model = keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Phase 1: Freeze base model layers
        base_model.trainable = False
        
        # Build classifier head dengan Functional API
        inputs = keras.Input(shape=self.input_shape)
        
        # Base model
        x = base_model(inputs, training=False)
        
        # Classifier head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        
        l2_reg = Config.L2_REGULARIZATION if Config is not None else 0.01
        dropout_1 = Config.DROPOUT_DENSE_1 if Config is not None else 0.4
        dropout_2 = Config.DROPOUT_DENSE_2 if Config is not None else 0.3
        label_smooth = Config.LABEL_SMOOTHING if Config is not None else 0.1
        lr_p1 = Config.TRAIN_P1_LEARNING_RATE if Config is not None else 1e-3
        
        x = layers.Dense(
            256, activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
        x = layers.Dropout(dropout_1)(x)
        x = layers.Dense(
            128, activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
        x = layers.Dropout(dropout_2)(x)
        num_classes = Config.MODEL_NUM_CLASSES if Config is not None else 3
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile dengan label smoothing
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_p1),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smooth),
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        self.model = model
        self._base_model = base_model
        print("Model dengan Transfer Learning (MobileNetV2) berhasil dibuat")
        print(f"  Total params: {model.count_params():,}")
    
    def _build_custom_cnn_model(self):
        """
        Custom CNN model yang lebih dalam untuk kasus tanpa pretrained weights
        """
        model = keras.Sequential([
            # Input
            keras.Input(shape=self.input_shape),
            
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Block 5 (tambahan)
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(Config.MODEL_NUM_CLASSES if Config is not None else 3, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        self.model = model
        self._base_model = None
        print("Custom CNN model berhasil dibuat")
        print(f"  Total params: {model.count_params():,}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=50, batch_size=32, class_weights=None,
              data_augmentation=None):
        """
        Train model (Phase 1: classifier head only)
        
        Args:
            X_train: Training images
            y_train: Training labels (one-hot encoded)
            X_val: Validation images (optional)
            y_val: Validation labels (optional)
            epochs: Number of epochs
            batch_size: Batch size
            class_weights: Dictionary of class weights untuk handle imbalanced data
            data_augmentation: ImageDataGenerator instance (optional)
            
        Returns:
            History object
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        monitor = 'val_loss' if X_val is not None else 'loss'
        
        # Check LR Schedule type from Config
        lr_schedule = Config.LR_SCHEDULE if Config is not None else 'reduce_on_plateau'
        patience_stop = Config.TRAIN_P1_PATIENCE_EARLY_STOP if Config is not None else 10
        patience_lr = Config.TRAIN_P1_PATIENCE_LR_REDUCE if Config is not None else 5
        lr_p1 = Config.TRAIN_P1_LEARNING_RATE if Config is not None else 1e-3
        min_lr = Config.LR_MIN if Config is not None else 1e-7
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience_stop,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        if lr_schedule in ['cosine_annealing', 'cosine_warmup']:
            warmup_epochs = Config.LR_WARMUP_EPOCHS if (Config is not None and lr_schedule == 'cosine_warmup') else 0
            callbacks.append(
                CosineWarmupDecayScheduler(
                    total_epochs=epochs,
                    warmup_epochs=warmup_epochs,
                    lr_max=lr_p1,
                    lr_min=min_lr,
                    verbose=1
                )
            )
        else:
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor=monitor,
                    factor=0.5,
                    patience=patience_lr,
                    min_lr=min_lr,
                    verbose=1
                )
            )
            
        checkpoint_path = Config.MODEL_BEST_P1_PATH if Config is not None else 'models/best_model_phase1.h5'
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=monitor,
                save_best_only=True,
                verbose=1
            )
        )
        
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        print("\n" + "=" * 50)
        print("PHASE 1: Training Classifier Head")
        print("=" * 50)
        
        if data_augmentation is not None:
            # Training dengan augmentasi
            flow_gen = data_augmentation.flow(X_train, y_train, batch_size=batch_size)
            
            # Wrap with Mixup if enabled
            mixup_enabled = Config.MIXUP_ENABLED if Config is not None else False
            if mixup_enabled:
                alpha = Config.MIXUP_ALPHA if Config is not None else 0.2
                flow_gen = mixup_generator(flow_gen, alpha)
                print(f"Mixup Augmentation diaktifkan dengan alpha={alpha}")
                
            history = self.model.fit(
                flow_gen,
                validation_data=validation_data,
                epochs=epochs,
                steps_per_epoch=len(X_train) // batch_size,
                callbacks=callbacks,
                class_weight=None if mixup_enabled else class_weights,
                verbose=1
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
        
        return history
    
    def fine_tune(self, X_train, y_train, X_val=None, y_val=None,
                  epochs=30, batch_size=32, n_layers_unfreeze=20,
                  class_weights=None, data_augmentation=None):
        """
        Fine-tune model (Phase 2: unfreeze top layers dari base model)
        Harus dipanggil SETELAH train()
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            n_layers_unfreeze: Jumlah layer terakhir yang di-unfreeze
            class_weights: Dictionary of class weights
            data_augmentation: ImageDataGenerator instance (optional)
            
        Returns:
            History object
        """
        if self._base_model is None:
            print("Fine-tuning hanya tersedia untuk pretrained model")
            return None
        
        print("\n" + "=" * 50)
        print(f"PHASE 2: Fine-tuning (unfreeze top {n_layers_unfreeze} layers)")
        print("=" * 50)
        
        # Unfreeze top N layers dari base model
        self._base_model.trainable = True
        
        # Freeze semua kecuali top N layers
        for layer in self._base_model.layers[:-n_layers_unfreeze]:
            layer.trainable = False
        
        # Recompile dengan learning rate lebih kecil
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        # Count trainable params
        trainable = sum(
            tf.keras.backend.count_params(w)
            for w in self.model.trainable_weights
        )
        print(f"  Trainable params setelah fine-tune: {trainable:,}")
        
        # Callbacks
        monitor = 'val_loss' if X_val is not None else 'loss'
        
        lr_schedule = Config.LR_SCHEDULE if Config is not None else 'reduce_on_plateau'
        patience_stop = Config.TRAIN_P2_PATIENCE_EARLY_STOP if Config is not None else 8
        patience_lr = Config.TRAIN_P2_PATIENCE_LR_REDUCE if Config is not None else 3
        lr_p2 = Config.TRAIN_P2_LEARNING_RATE if Config is not None else 1e-5
        min_lr = Config.LR_MIN if Config is not None else 1e-8
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience_stop,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        if lr_schedule in ['cosine_annealing', 'cosine_warmup']:
            warmup_epochs = Config.LR_WARMUP_EPOCHS if (Config is not None and lr_schedule == 'cosine_warmup') else 0
            callbacks.append(
                CosineWarmupDecayScheduler(
                    total_epochs=epochs,
                    warmup_epochs=warmup_epochs,
                    lr_max=lr_p2,
                    lr_min=min_lr,
                    verbose=1
                )
            )
        else:
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor=monitor,
                    factor=0.3,
                    patience=patience_lr,
                    min_lr=min_lr,
                    verbose=1
                )
            )
            
        checkpoint_path = Config.MODEL_BEST_FT_PATH if Config is not None else 'models/best_model_finetuned.h5'
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=monitor,
                save_best_only=True,
                verbose=1
            )
        )
        
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        if data_augmentation is not None:
            # Fine-tuning dengan augmentasi
            flow_gen = data_augmentation.flow(X_train, y_train, batch_size=batch_size)
            
            # Wrap with Mixup if enabled
            mixup_enabled = Config.MIXUP_ENABLED if Config is not None else False
            if mixup_enabled:
                alpha = Config.MIXUP_ALPHA if Config is not None else 0.2
                flow_gen = mixup_generator(flow_gen, alpha)
                print(f"Mixup Augmentation (Fine-tuning) diaktifkan dengan alpha={alpha}")
                
            history = self.model.fit(
                flow_gen,
                validation_data=validation_data,
                epochs=epochs,
                steps_per_epoch=len(X_train) // batch_size,
                callbacks=callbacks,
                class_weight=None if mixup_enabled else class_weights,
                verbose=1
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
        
        return history
    
    def predict(self, image, confidence_threshold=0.0):
        """
        Predict grade untuk satu image
        
        Args:
            image: Input image (224x224x3, normalized)
            confidence_threshold: Minimum confidence untuk valid prediction
            
        Returns:
            tuple: (predicted_grade, confidence, probabilities)
        """
        if self.model is None:
            raise Exception("Model belum dibuat atau dimuat")
        
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Predict
        predictions = self.model.predict(image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Check confidence threshold
        if confidence < confidence_threshold:
            grade = 'Uncertain'
        else:
            grade = self.grade_labels[predicted_class]
        
        return grade, confidence, predictions[0]
    
    def predict_with_tta(self, image, n_augments=5):
        """
        Predict dengan Test-Time Augmentation (TTA)
        Rata-rata prediksi dari beberapa versi augmented untuk hasil lebih robust
        
        Args:
            image: Input image (224x224x3, normalized)
            n_augments: Jumlah augmentasi untuk TTA
            
        Returns:
            tuple: (predicted_grade, confidence, probabilities)
        """
        if self.model is None:
            raise Exception("Model belum dibuat atau dimuat")
        
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Original prediction
        all_predictions = [self.model.predict(image, verbose=0)[0]]
        
        # Augmented predictions
        img = image[0]
        
        # Horizontal flip
        flipped = np.fliplr(img)
        all_predictions.append(
            self.model.predict(np.expand_dims(flipped, axis=0), verbose=0)[0]
        )
        
        # Slight brightness variations
        for factor in [0.9, 1.1]:
            bright = np.clip(img * factor, -1.0, 1.0)
            all_predictions.append(
                self.model.predict(np.expand_dims(bright, axis=0), verbose=0)[0]
            )
        
        # Slight rotation via transpose/flip combo
        rotated = np.flipud(img)
        all_predictions.append(
            self.model.predict(np.expand_dims(rotated, axis=0), verbose=0)[0]
        )
        
        # Average predictions
        avg_predictions = np.mean(all_predictions[:n_augments + 1], axis=0)
        predicted_class = np.argmax(avg_predictions)
        confidence = float(avg_predictions[predicted_class])
        grade = self.grade_labels[predicted_class]
        
        return grade, confidence, avg_predictions
    
    def predict_batch(self, images):
        """
        Predict grade untuk multiple images
        
        Args:
            images: Batch of images
            
        Returns:
            list: List of (grade, confidence, probabilities) tuples
        """
        if self.model is None:
            raise Exception("Model belum dibuat atau dimuat")
        
        predictions = self.model.predict(images, verbose=0)
        results = []
        
        for pred in predictions:
            predicted_class = np.argmax(pred)
            confidence = float(pred[predicted_class])
            grade = self.grade_labels[predicted_class]
            results.append((grade, confidence, pred))
        
        return results
    
    def save_model(self, path):
        """
        Save model ke file
        
        Args:
            path: Path untuk save model
        """
        if self.model is None:
            raise Exception("Tidak ada model untuk disave")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model berhasil disave ke {path}")
    
    def load_model(self, path):
        """
        Load model dari file
        
        Args:
            path: Path ke saved model
        """
        self.model = keras.models.load_model(path)
        self._base_model = None  # Cannot fine-tune loaded model
        print(f"Model berhasil dimuat dari {path}")
    
    def get_model_summary(self):
        """Get model summary"""
        if self.model is not None:
            return self.model.summary()
        return None
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model pada test data
        
        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise Exception("Model belum dibuat atau dimuat")
        
        results = self.model.evaluate(X_test, y_test, verbose=1)
        
        metrics = {}
        for name, value in zip(self.model.metrics_names, results):
            metrics[name] = float(value)
        
        return metrics

    def get_gradcam_heatmap(self, image, pred_index=None, last_conv_layer_name=None):
        """
        Generate Grad-CAM heatmap for the given image.
        
        Args:
            image: Preprocessed/normalized image (shape 224x224x3)
            pred_index: Index of class to compute heatmap for (None = predicted class)
            last_conv_layer_name: Name of last convolutional layer (None = auto-detect)
            
        Returns:
            tuple: (heatmap, pred_index)
        """
        if self.model is None:
            raise Exception("Model belum dimuat/dibuat")
            
        if len(image.shape) == 3:
            img_array = np.expand_dims(image, axis=0)
        else:
            img_array = image
            
        # Detect if there's a nested base model (transfer learning)
        base_model = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model) or (hasattr(layer, 'layers') and len(layer.layers) > 0):
                base_model = layer
                break
                
        if base_model is not None:
            # Nested model (EfficientNet / MobileNet)
            if last_conv_layer_name is None:
                # Find last conv layer in base model
                for l in reversed(base_model.layers):
                    if 'conv' in l.name.lower() or 'project' in l.name.lower() or 'activation' in l.name.lower():
                        last_conv_layer_name = l.name
                        break
            
            base_conv_layer = base_model.get_layer(last_conv_layer_name)
            base_model_conv_output = base_conv_layer.output
            
            # Find the classifier head layers in the main model
            classifier_layers = []
            found_base = False
            for layer in self.model.layers:
                if layer == base_model:
                    found_base = True
                    continue
                if found_base:
                    classifier_layers.append(layer)
            
            # Reconstruct path from base model output to final prediction
            base_model_out = base_model.output
            x = base_model_out
            for layer in classifier_layers:
                x = layer(x)
            
            # Build custom grad model
            grad_model = tf.keras.Model(
                inputs=base_model.input,
                outputs=[base_model_conv_output, x]
            )
        else:
            # Custom CNN (flat model)
            if last_conv_layer_name is None:
                for l in reversed(self.model.layers):
                    if 'conv' in l.name.lower():
                        last_conv_layer_name = l.name
                        break
            grad_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=[self.model.get_layer(last_conv_layer_name).output, self.model.output]
            )
            
        # Record gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
            
        # Get gradients of prediction wrt conv output
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Mean intensity of gradients per channel (GAP over spatial dimensions)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply feature map by importance and sum
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Apply ReLU to keep positive influence and normalize
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy(), int(pred_index)

    @staticmethod
    def overlay_gradcam(image, heatmap, alpha=0.4):
        """
        Overlay a Grad-CAM heatmap on top of a BGR image.
        
        Args:
            image: Original uint8 image (BGR)
            heatmap: 2D float array (0.0 to 1.0)
            alpha: Transparency factor for overlay
            
        Returns:
            numpy.ndarray: Annotated BGR image
        """
        # Rescale heatmap to 0-255
        heatmap_im = np.uint8(255 * heatmap)
        
        # Apply jet colormap
        jet = cv2.applyColorMap(heatmap_im, cv2.COLORMAP_JET)

        # Resize colormap to original image size
        jet = cv2.resize(jet, (image.shape[1], image.shape[0]))
        
        # Superimpose
        superimposed = jet * alpha + image * (1 - alpha)
        superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
        
        return superimposed
