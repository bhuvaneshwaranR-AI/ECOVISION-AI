"""
Model ensemble implementation for land cover classification.
"""

import os
import numpy as np
import tensorflow as tf
from typing import List, Dict, Optional, Union, Tuple, Any
import json
from unet_model import UNetModel
from config import Config

class EnsembleModel:
    def __init__(self, input_shape: Tuple[int, int, int] = (256, 256, 6), num_classes: int = 6):
        """
        Initialize the ensemble model with multiple U-Net variants.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of land cover classes to predict
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.models: List[tf.keras.Model] = []
        self.model_weights: List[float] = []
        
    def add_model(self, model_type: str, filters_base: int = 64, weight: float = 1.0) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            model_type: Type of U-Net variant ('standard', 'resnet', 'efficientnet')
            filters_base: Number of base filters for the model
            weight: Weight for this model in the ensemble
        """
        if model_type == 'standard':
            model = UNetModel(
                input_shape=self.input_shape,
                filters_base=filters_base,
                num_classes=self.num_classes
            ).model
        elif model_type == 'resnet':
            model = self._create_resnet_unet(filters_base)
        elif model_type == 'efficientnet':
            model = self._create_efficientnet_unet(filters_base)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.models.append(model)
        self.model_weights.append(weight)
        
    def _create_resnet_unet(self, filters_base: int) -> tf.keras.Model:
        """Create U-Net with ResNet-style blocks."""
        def residual_block(x: tf.Tensor, filters: int, kernel_size: int = 3) -> tf.Tensor:
            skip = x
            x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            
            if skip.shape[-1] != filters:
                skip = tf.keras.layers.Conv2D(filters, 1, padding='same')(skip)
            
            x = tf.keras.layers.Add()([x, skip])
            x = tf.keras.layers.ReLU()(x)
            return x
        
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        
        # Encoder
        x = residual_block(inputs, filters_base)
        skip1 = x
        x = tf.keras.layers.MaxPooling2D()(x)
        
        x = residual_block(x, filters_base * 2)
        skip2 = x
        x = tf.keras.layers.MaxPooling2D()(x)
        
        x = residual_block(x, filters_base * 4)
        skip3 = x
        x = tf.keras.layers.MaxPooling2D()(x)
        
        x = residual_block(x, filters_base * 8)
        skip4 = x
        x = tf.keras.layers.MaxPooling2D()(x)
        
        # Bridge
        x = residual_block(x, filters_base * 16)
        
        # Decoder
        x = tf.keras.layers.Conv2DTranspose(filters_base * 8, 3, strides=2, padding='same')(x)
        x = tf.keras.layers.Concatenate()([x, skip4])
        x = residual_block(x, filters_base * 8)
        
        x = tf.keras.layers.Conv2DTranspose(filters_base * 4, 3, strides=2, padding='same')(x)
        x = tf.keras.layers.Concatenate()([x, skip3])
        x = residual_block(x, filters_base * 4)
        
        x = tf.keras.layers.Conv2DTranspose(filters_base * 2, 3, strides=2, padding='same')(x)
        x = tf.keras.layers.Concatenate()([x, skip2])
        x = residual_block(x, filters_base * 2)
        
        x = tf.keras.layers.Conv2DTranspose(filters_base, 3, strides=2, padding='same')(x)
        x = tf.keras.layers.Concatenate()([x, skip1])
        x = residual_block(x, filters_base)
        
        # Output
        class_logits = tf.keras.layers.Conv2D(self.num_classes, 1)(x)
        class_probs = tf.keras.layers.Softmax(axis=-1, name='class_probabilities')(class_logits)
        confidence = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', name='confidence_scores')(x)
        outputs = tf.keras.layers.Concatenate(name='model_outputs')([class_probs, confidence])
        
        return tf.keras.Model(inputs=inputs, outputs=outputs, name='resnet_unet')
    
    def _create_efficientnet_unet(self, filters_base: int) -> tf.keras.Model:
        """Create U-Net with EfficientNet backbone."""
        # Load pre-trained EfficientNetB0 without top layers
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )
        
        # Get intermediate layers for skip connections
        skip_names = [
            'block2a_expand_activation',
            'block3a_expand_activation',
            'block4a_expand_activation',
            'block6a_expand_activation'
        ]
        skips = [backbone.get_layer(name).output for name in skip_names]
        
        # Create decoder
        x = backbone.output
        
        for skip in reversed(skips):
            x = tf.keras.layers.Conv2DTranspose(x.shape[-1] // 2, 3, strides=2, padding='same')(x)
            x = tf.keras.layers.Concatenate()([x, skip])
            x = tf.keras.layers.Conv2D(x.shape[-1] // 2, 3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
        
        # Final upsampling to match input resolution
        x = tf.keras.layers.Conv2DTranspose(filters_base, 3, strides=2, padding='same')(x)
        x = tf.keras.layers.Conv2D(filters_base, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        
        # Output
        class_logits = tf.keras.layers.Conv2D(self.num_classes, 1)(x)
        class_probs = tf.keras.layers.Softmax(axis=-1, name='class_probabilities')(class_logits)
        confidence = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', name='confidence_scores')(x)
        outputs = tf.keras.layers.Concatenate(name='model_outputs')([class_probs, confidence])
        
        return tf.keras.Model(inputs=backbone.input, outputs=outputs, name='efficientnet_unet')
    
    def compile_models(self, learning_rate: float = 0.001, class_weights: Optional[Dict[int, float]] = None) -> None:
        """
        Compile all models in the ensemble.
        
        Args:
            learning_rate: Learning rate for the Adam optimizer
            class_weights: Optional dictionary of class weights for imbalanced datasets
        """
        for model in self.models:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss={
                    'class_probabilities': 'categorical_crossentropy',
                    'confidence_scores': 'binary_crossentropy'
                },
                loss_weights={
                    'class_probabilities': 1.0,
                    'confidence_scores': 0.5
                },
                metrics=['accuracy']
            )
    
    def train(self, train_data: tf.data.Dataset, validation_data: Optional[tf.data.Dataset] = None,
             epochs: int = 50, callbacks: Optional[List[tf.keras.callbacks.Callback]] = None) -> List[Dict[str, Any]]:
        """
        Train all models in the ensemble.
        
        Args:
            train_data: Training dataset
            validation_data: Optional validation dataset
            epochs: Number of training epochs
            callbacks: Optional list of Keras callbacks
            
        Returns:
            List of training histories for each model
        """
        histories = []
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{len(self.models)}")
            history = model.fit(
                train_data,
                validation_data=validation_data,
                epochs=epochs,
                callbacks=callbacks
            )
            histories.append(history.history)
        return histories
    
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ensemble predictions.
        
        Args:
            x: Input data
            
        Returns:
            Tuple of (class probabilities, confidence scores)
        """
        predictions = []
        for model, weight in zip(self.models, self.model_weights):
            pred = model.predict(x)
            predictions.append((pred[..., :self.num_classes] * weight, pred[..., -1:] * weight))
        
        # Average predictions weighted by model weights
        total_weight = sum(self.model_weights)
        class_probs = sum(p[0] for p in predictions) / total_weight
        confidence = sum(p[1] for p in predictions) / total_weight
        
        return class_probs, confidence
    
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the ensemble model.
        
        Args:
            x: Input data
            y: Ground truth labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        class_probs, confidence = self.predict(x)
        y_pred = np.argmax(class_probs, axis=-1)
        y_true = np.argmax(y[..., :self.num_classes], axis=-1)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_true)
        iou = self._calculate_iou(y_true, y_pred)
        confidence_error = np.mean(np.abs(confidence - y[..., -1:]))
        
        return {
            'accuracy': float(accuracy),
            'mean_iou': float(np.mean(iou)),
            'confidence_error': float(confidence_error)
        }
    
    def _calculate_iou(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate IoU for each class."""
        ious = []
        for class_id in range(self.num_classes):
            intersection = np.sum((y_true == class_id) & (y_pred == class_id))
            union = np.sum((y_true == class_id) | (y_pred == class_id))
            iou = intersection / (union + 1e-7)
            ious.append(iou)
        return np.array(ious)
    
    def save_models(self, save_dir: str) -> None:
        """
        Save all models in the ensemble.
        
        Args:
            save_dir: Directory to save the models
        """
        os.makedirs(save_dir, exist_ok=True)
        for i, model in enumerate(self.models):
            model_path = os.path.join(save_dir, f'model_{i+1}.h5')
            model.save(model_path)
        
        # Save ensemble configuration
        config = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'model_weights': self.model_weights
        }
        config_path = os.path.join(save_dir, 'ensemble_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)
    
    @classmethod
    def load_models(cls, load_dir: str) -> 'EnsembleModel':
        """
        Load ensemble models from directory.
        
        Args:
            load_dir: Directory containing saved models
            
        Returns:
            Loaded EnsembleModel instance
        """
        # Load configuration
        config_path = os.path.join(load_dir, 'ensemble_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create ensemble instance
        ensemble = cls(
            input_shape=tuple(config['input_shape']),
            num_classes=config['num_classes']
        )
        
        # Load models
        model_files = [f for f in os.listdir(load_dir) if f.startswith('model_') and f.endswith('.h5')]
        for model_file in sorted(model_files):
            model_path = os.path.join(load_dir, model_file)
            model = tf.keras.models.load_model(model_path)
            ensemble.models.append(model)
        
        ensemble.model_weights = config['model_weights']
        return ensemble