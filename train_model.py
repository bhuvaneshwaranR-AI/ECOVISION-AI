import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from unet_model import UNetModel

class ModelTrainer:
    def __init__(self, data_dir="data/processed"):
        """
        Initialize model trainer.
        
        Args:
            data_dir: Directory containing processed datasets
        """
        self.data_dir = data_dir
        self.model = None
        
    def load_data(self):
        """Load prepared datasets."""
        try:
            train_data = np.load(os.path.join(self.data_dir, "train_data.npy"))
            val_data = np.load(os.path.join(self.data_dir, "val_data.npy"))
            test_data = np.load(os.path.join(self.data_dir, "test_data.npy"))
            
            return train_data, val_data, test_data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
            
    def train_model(self, train_data, val_data, epochs=50, batch_size=32):
        """Train the U-Net model."""
        try:
            # Create and compile model
            unet = UNetModel()
            self.model = unet.build_model()
            self.model = unet.compile_model(self.model)
            
            # Create checkpoint callback
            checkpoint_path = "checkpoints/model_{epoch:02d}.h5"
            os.makedirs("checkpoints", exist_ok=True)
            
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    checkpoint_path,
                    save_best_only=True,
                    monitor='val_loss'
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    histogram_freq=1
                )
            ]
            
            # Train model
            history = self.model.fit(
                train_data,
                validation_data=val_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )
            
            return history
            
    def evaluate_model(self, test_data):
        """Evaluate the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        # Evaluate model
        results = self.model.evaluate(test_data)
        
        # Print evaluation metrics
        metrics = self.model.metrics_names
        print("\nModel Evaluation Results:")
        for metric, value in zip(metrics, results):
            print(f"{metric}: {value:.4f}")
            
    def save_model(self, model_path="model/deforestation_detector.h5"):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        print(f"\nModel saved to {model_path}")

def main():
    try:
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Load data
        print("Loading datasets...")
        train_data, val_data, test_data = trainer.load_data()
        
        # Train model
        print("\nStarting model training...")
        history = trainer.train_model(train_data, val_data)
        
        # Evaluate model
        print("\nEvaluating model...")
        trainer.evaluate_model(test_data)
        
        # Save model
        trainer.save_model()
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()