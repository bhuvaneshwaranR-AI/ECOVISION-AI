import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
import json
import os

class UNetModel:
    def __init__(self, input_shape=(256, 256, 6), filters_base=64, num_classes=6):
        """
        Initialize U-Net model for land cover classification.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            filters_base: Number of base filters (doubled in each encoder block)
            num_classes: Number of land cover classes to predict
        """
        self.input_shape = input_shape
        self.filters_base = filters_base
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _encoder_block(self, inputs, filters, name_prefix):
        """Create an encoder block with double convolution and max pooling."""
        conv = layers.Conv2D(filters, 3, activation='relu', padding='same',
                           name=f'{name_prefix}_conv1')(inputs)
        conv = layers.BatchNormalization(name=f'{name_prefix}_bn1')(conv)
        conv = layers.Conv2D(filters, 3, activation='relu', padding='same',
                           name=f'{name_prefix}_conv2')(conv)
        conv = layers.BatchNormalization(name=f'{name_prefix}_bn2')(conv)
        pool = layers.MaxPooling2D(pool_size=(2, 2), name=f'{name_prefix}_pool')(conv)
        return conv, pool
    
    def _decoder_block(self, inputs, skip_connection, filters, name_prefix):
        """Create a decoder block with upsampling and concatenation."""
        up = layers.UpSampling2D(size=(2, 2), name=f'{name_prefix}_up')(inputs)
        concat = layers.Concatenate(name=f'{name_prefix}_concat')([up, skip_connection])
        conv = layers.Conv2D(filters, 3, activation='relu', padding='same',
                           name=f'{name_prefix}_conv1')(concat)
        conv = layers.BatchNormalization(name=f'{name_prefix}_bn1')(conv)
        conv = layers.Conv2D(filters, 3, activation='relu', padding='same',
                           name=f'{name_prefix}_conv2')(conv)
        conv = layers.BatchNormalization(name=f'{name_prefix}_bn2')(conv)
        return conv
    
    def _build_model(self):
        """Build the complete U-Net architecture for land cover classification."""
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='input_image')
        
        # Encoder path with edge enhancement
        conv1, pool1 = self._encoder_block(inputs, self.filters_base, 'encoder1')
        conv2, pool2 = self._encoder_block(pool1, self.filters_base*2, 'encoder2')
        conv3, pool3 = self._encoder_block(pool2, self.filters_base*4, 'encoder3')
        conv4, pool4 = self._encoder_block(pool3, self.filters_base*8, 'encoder4')
        conv5, pool5 = self._encoder_block(pool4, self.filters_base*16, 'encoder5')
        
        # Bridge with enhanced feature extraction and dropout
        bridge = layers.Conv2D(self.filters_base*32, 3, padding='same', use_bias=False, name='bridge_conv1')(pool5)
        bridge = layers.BatchNormalization(name='bridge_bn1')(bridge)
        bridge = layers.ReLU(name='bridge_relu1')(bridge)
        bridge = layers.Conv2D(self.filters_base*32, 3, padding='same', use_bias=False, name='bridge_conv2')(bridge)
        bridge = layers.BatchNormalization(name='bridge_bn2')(bridge)
        bridge = layers.ReLU(name='bridge_relu2')(bridge)
        bridge = layers.SpatialDropout2D(0.5, name='bridge_dropout')(bridge)
        
        # Decoder path with attention mechanism
        decoder5 = self._decoder_block(bridge, conv5, self.filters_base*16, 'decoder5')
        decoder4 = self._decoder_block(decoder5, conv4, self.filters_base*8, 'decoder4')
        decoder3 = self._decoder_block(decoder4, conv3, self.filters_base*4, 'decoder3')
        decoder2 = self._decoder_block(decoder3, conv2, self.filters_base*2, 'decoder2')
        decoder1 = self._decoder_block(decoder2, conv1, self.filters_base, 'decoder1')
        
        # Edge enhancement with batch normalization
        edge_features = layers.Conv2D(self.filters_base, 3, padding='same', use_bias=False, name='edge_conv1')(decoder1)
        edge_features = layers.BatchNormalization(name='edge_bn1')(edge_features)
        edge_features = layers.ReLU(name='edge_relu1')(edge_features)
        
        # Classification head with confidence scores
        class_logits = layers.Conv2D(self.num_classes, 1, name='class_logits')(edge_features)
        class_probs = layers.Softmax(axis=-1, name='class_probabilities')(class_logits)
        
        # Confidence score output with separate convolution
        confidence_features = layers.Conv2D(self.filters_base//2, 3, padding='same', use_bias=False)(edge_features)
        confidence_features = layers.BatchNormalization()(confidence_features)
        confidence_features = layers.ReLU()(confidence_features)
        confidence = layers.Conv2D(1, 1, activation='sigmoid', name='confidence_scores')(confidence_features)
        
        # Combine outputs
        outputs = layers.Concatenate(name='model_outputs')([class_probs, confidence])
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='unet_landcover')
        return model
    
    @staticmethod
    def weighted_categorical_crossentropy(y_true, y_pred):
        """Custom loss function for weighted categorical crossentropy."""
        # Split predictions into class probabilities and confidence scores
        y_pred_classes = y_pred[..., :-1]
        y_true_classes = y_true[..., :-1]
        
        # Calculate categorical crossentropy
        weighted_losses = keras.losses.categorical_crossentropy(
            y_true_classes, y_pred_classes
        )
        
        return tf.reduce_mean(weighted_losses)
    
    def mean_iou(self, y_true, y_pred):
        """Calculate mean Intersection over Union across all classes."""
        # Split predictions
        y_pred_classes = y_pred[..., :-1]  # Remove confidence scores
        y_true_classes = y_true[..., :-1]
        
        # Convert probabilities to class indices
        y_pred_classes = tf.argmax(y_pred_classes, axis=-1)
        y_true_classes = tf.argmax(y_true_classes, axis=-1)
        
        # Calculate IoU for each class
        ious = []
        for class_id in range(self.num_classes):
            y_true_class = tf.cast(y_true_classes == class_id, tf.float32)
            y_pred_class = tf.cast(y_pred_classes == class_id, tf.float32)
            
            intersection = tf.reduce_sum(y_true_class * y_pred_class)
            union = tf.reduce_sum(y_true_class) + tf.reduce_sum(y_pred_class) - intersection
            iou = intersection / (union + keras.backend.epsilon())
            ious.append(iou)
        
        return tf.reduce_mean(ious)
    
    @staticmethod
    def confidence_accuracy(y_true, y_pred):
        """Calculate accuracy weighted by confidence scores."""
        # Split predictions
        y_pred_classes = y_pred[..., :-1]
        y_pred_confidence = y_pred[..., -1]
        y_true_classes = y_true[..., :-1]
        
        # Get predicted and true classes
        pred_classes = tf.argmax(y_pred_classes, axis=-1)
        true_classes = tf.argmax(y_true_classes, axis=-1)
        
        # Calculate accuracy weighted by confidence
        correct_predictions = tf.cast(pred_classes == true_classes, tf.float32)
        weighted_accuracy = tf.reduce_sum(correct_predictions * y_pred_confidence) / tf.reduce_sum(y_pred_confidence)
        
        return weighted_accuracy
    
    @staticmethod
    def boundary_accuracy(y_true, y_pred, boundary_width=2):
        """Calculate accuracy at class boundaries."""
        # Split predictions
        y_pred_classes = y_pred[..., :-1]
        y_true_classes = y_true[..., :-1]
        
        # Get predicted and true classes
        pred_classes = tf.argmax(y_pred_classes, axis=-1)
        true_classes = tf.argmax(y_true_classes, axis=-1)
        
        # Detect boundaries (class transitions)
        true_boundaries = tf.cast(
            tf.reduce_any([
                true_classes != tf.roll(true_classes, shift, axis=axis)
                for shift in range(-boundary_width, boundary_width + 1)
                for axis in [1, 2]
            ], axis=0),
            tf.float32
        )
        
        # Calculate accuracy only at boundary pixels
        correct_predictions = tf.cast(pred_classes == true_classes, tf.float32)
        boundary_accuracy = tf.reduce_sum(correct_predictions * true_boundaries) / (
            tf.reduce_sum(true_boundaries) + keras.backend.epsilon()
        )
        
        return boundary_accuracy
    
    def compile_model(self, learning_rate=0.001, class_weights=None):
        """
        Compile the model with multi-class loss functions and metrics.
        
        Args:
            learning_rate: Learning rate for the Adam optimizer
            class_weights: Optional dictionary of class weights for imbalanced datasets
        """
        # Compile model with custom loss and metrics
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self.weighted_categorical_crossentropy,
            metrics=[
                self.mean_iou,
                self.confidence_accuracy,
                self.boundary_accuracy,
                'categorical_accuracy'
            ]
        )
    
    def save_model(self, model_path, metadata_path=None):
        """
        Save the model and its metadata.
        
        Args:
            model_path: Path to save the model
            metadata_path: Optional path to save model metadata
        """
        # Save model
        self.model.save(model_path)
        
        # Save metadata if path provided
        if metadata_path:
            metadata = {
                'input_shape': self.input_shape,
                'filters_base': self.filters_base,
                'model_parameters': self.model.count_params(),
                'model_layers': len(self.model.layers),
                'architecture': 'U-Net',
                'task': 'Deforestation Detection',
                'output_activation': 'sigmoid',
                'metrics': ['IoU', 'F1-Score']
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
    
    @classmethod
    def load_model(cls, model_path, metadata_path=None):
        """
        Load a saved model and its metadata.
        
        Args:
            model_path: Path to the saved model
            metadata_path: Optional path to the model metadata
        """
        # Load metadata if available
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            input_shape = metadata['input_shape']
            filters_base = metadata['filters_base']
        else:
            # Use default values if metadata not available
            input_shape = (256, 256, 6)  # Default shape
            filters_base = 64
        
        # Create instance
        instance = cls(input_shape=input_shape, filters_base=filters_base)
        
        # Load model with custom objects
        instance.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'weighted_categorical_crossentropy': UNetModel.weighted_categorical_crossentropy,
                'mean_iou': instance.mean_iou,
                'confidence_accuracy': UNetModel.confidence_accuracy,
                'boundary_accuracy': UNetModel.boundary_accuracy
            }
        )
        return instance

def main():
    """Example usage of the UNetModel class."""
    try:
        # Example input shape (height, width, channels)
        # Channels = 6 (3 bands/indices from current + 3 from previous period)
        input_shape = (256, 256, 6)
        
        # Initialize model
        unet = UNetModel(input_shape=input_shape)
        
        # Compile model
        unet.compile_model(learning_rate=0.001)
        
        # Print model summary
        unet.model.summary()
        
        # Save model architecture visualization
        tf.keras.utils.plot_model(
            unet.model,
            to_file='model_architecture.png',
            show_shapes=True,
            show_layer_names=True
        )
        
        # Create models directory
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model and metadata
        unet.save_model(
            model_path=os.path.join(model_dir, 'unet_deforestation.h5'),
            metadata_path=os.path.join(model_dir, 'model_metadata.json')
        )
        
        print("\nModel architecture saved as 'model_architecture.png'")
        print(f"Model and metadata saved in: {model_dir}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()