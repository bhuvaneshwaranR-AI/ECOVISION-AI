import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
import json
import datetime
from unet_model import UNetModel
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, dataset_dir, model_dir, batch_size=8):
        """
        Initialize the model trainer.
        
        Args:
            dataset_dir: Directory containing the prepared dataset
            model_dir: Directory to save model checkpoints and logs
            batch_size: Batch size for training
        """
        self.dataset_dir = dataset_dir
        self.model_dir = model_dir
        self.batch_size = batch_size
        
        # Create necessary directories
        self.checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        self.logs_dir = os.path.join(model_dir, 'logs')
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Load dataset metadata
        with open(os.path.join(dataset_dir, 'dataset_metadata.json'), 'r') as f:
            self.dataset_metadata = json.load(f)
    
    def load_data(self):
        """Load training, validation, and test datasets."""
        print("Loading datasets...")
        
        # Load training data
        X_train = np.load(os.path.join(self.dataset_dir, 'train', 'X_train.npy'))
        y_train = np.load(os.path.join(self.dataset_dir, 'train', 'y_train.npy'))
        
        # Load validation data
        X_val = np.load(os.path.join(self.dataset_dir, 'val', 'X_val.npy'))
        y_val = np.load(os.path.join(self.dataset_dir, 'val', 'y_val.npy'))
        
        # Load test data
        X_test = np.load(os.path.join(self.dataset_dir, 'test', 'X_test.npy'))
        y_test = np.load(os.path.join(self.dataset_dir, 'test', 'y_test.npy'))
        
        print(f"Dataset loaded:")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def create_data_generators(self, X_train, y_train, X_val, y_val):
        """Create data generators with augmentation for training."""
        # Data augmentation for training
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='reflect'
        )
        
        # No augmentation for validation
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        
        # Create generators
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def create_callbacks(self):
        """Create training callbacks for monitoring and checkpointing."""
        # Model checkpoint callback
        checkpoint_path = os.path.join(
            self.checkpoints_dir,
            'model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5'
        )
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # TensorBoard callback
        log_dir = os.path.join(
            self.logs_dir,
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
        
        # Learning rate reduction callback
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        return [checkpoint, early_stopping, tensorboard, reduce_lr]
    
    def train_model(self, epochs=100):
        """Train the model with the prepared dataset."""
        try:
            # Load data
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.load_data()
            
            # Create data generators
            train_generator, val_generator = self.create_data_generators(
                X_train, y_train, X_val, y_val
            )
            
            # Initialize model
            model = UNetModel(input_shape=X_train.shape[1:])
            model.compile_model(learning_rate=0.001)
            
            # Create callbacks
            callbacks = self.create_callbacks()
            
            # Train model
            print("\nStarting model training...")
            history = model.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save training history
            history_path = os.path.join(self.model_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(history.history, f, indent=4)
            
            # Save final model
            final_model_path = os.path.join(self.model_dir, 'final_model.h5')
            final_metadata_path = os.path.join(self.model_dir, 'final_model_metadata.json')
            model.save_model(final_model_path, final_metadata_path)
            
            # Evaluate on test set
            print("\nEvaluating model on test set...")
            test_results = model.model.evaluate(X_test, y_test, verbose=1)
            test_metrics = dict(zip(model.model.metrics_names, test_results))
            
            # Save test results
            test_results_path = os.path.join(self.model_dir, 'test_results.json')
            with open(test_results_path, 'w') as f:
                json.dump(test_metrics, f, indent=4)
            
            print("\nTraining completed successfully!")
            print(f"Final model saved to: {final_model_path}")
            print(f"Test results saved to: {test_results_path}")
            
            return history, test_metrics
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
    
    def plot_training_history(self, history):
        """Plot training history metrics."""
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot metrics
        ax2.plot(history.history['iou_metric'], label='Training IoU')
        ax2.plot(history.history['val_iou_metric'], label='Validation IoU')
        ax2.plot(history.history['f1_metric'], label='Training F1')
        ax2.plot(history.history['val_f1_metric'], label='Validation F1')
        ax2.set_title('Model Metrics')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.legend()
        
        # Save plot
        plt.tight_layout()
        plot_path = os.path.join(self.model_dir, 'training_history.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f"\nTraining history plot saved to: {plot_path}")

def main():
    """Example usage of the ModelTrainer class."""
    try:
        # Set up directories
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(base_dir, 'dataset')
        model_dir = os.path.join(base_dir, 'models')
        
        # Initialize trainer
        trainer = ModelTrainer(dataset_dir, model_dir)
        
        # Train model
        history, test_metrics = trainer.train_model(epochs=100)
        
        # Plot training history
        trainer.plot_training_history(history)
        
        print("\nTest metrics:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()