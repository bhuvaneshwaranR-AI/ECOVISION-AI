import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

class DatasetPreparator:
    def __init__(self, data_dir, tile_size=256):
        """
        Initialize dataset preparation for deforestation detection.
        
        Args:
            data_dir: Directory containing the satellite imagery
            tile_size: Size of image tiles for training (default: 256x256)
        """
        self.data_dir = data_dir
        self.tile_size = tile_size
        
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a satellite image."""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to numpy array and normalize
            image_array = np.array(image, dtype=np.float32)
            image_array = image_array / 255.0  # Normalize to [0,1]
            
            return image_array
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None
            
    def prepare_dataset(self, train_ratio=0.7, val_ratio=0.15):
        """
        Prepare the dataset for training, validation, and testing.
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            (remaining will be used for testing)
        """
        try:
            # Get all image files
            image_files = [f for f in os.listdir(self.data_dir) 
                         if f.startswith('sample_image_') and f.endswith('.png')]
            
            images = []
            masks = []
            
            for image_file in image_files:
                # Load image
                image_path = os.path.join(self.data_dir, image_file)
                image_array = self.load_and_preprocess_image(image_path)
                
                # Load corresponding mask
                mask_file = image_file.replace('image', 'mask')
                mask_path = os.path.join(self.data_dir, mask_file)
                mask_array = self.load_and_preprocess_image(mask_path)
                
                if image_array is not None and mask_array is not None:
                    images.append(image_array)
                    masks.append(mask_array)
            
            # Convert to numpy arrays
            images = np.array(images)
            masks = np.array(masks)
            
            # Split dataset
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                images, masks,
                test_size=(1-train_ratio-val_ratio),
                random_state=42
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=val_ratio/(train_ratio+val_ratio),
                random_state=42
            )
            
            print(f"\nDataset preparation complete:")
            print(f"Training samples: {len(X_train)}")
            print(f"Validation samples: {len(X_val)}")
            print(f"Test samples: {len(X_test)}")
            
            return (X_train, y_train), (X_val, y_val), (X_test, y_test)
            
        except Exception as e:
            print(f"Error preparing dataset: {str(e)}")
            raise

def main():
    try:
        # Initialize dataset preparator
        data_dir = "data/sentinel_images"
        preparator = DatasetPreparator(data_dir)
        
        # Prepare dataset
        train_data, val_data, test_data = preparator.prepare_dataset()
        
        # Save prepared datasets
        os.makedirs("data/processed", exist_ok=True)
        
        print("\nSaving processed datasets...")
        np.save("data/processed/train_images.npy", train_data[0])
        np.save("data/processed/train_masks.npy", train_data[1])
        np.save("data/processed/val_images.npy", val_data[0])
        np.save("data/processed/val_masks.npy", val_data[1])
        np.save("data/processed/test_images.npy", test_data[0])
        np.save("data/processed/test_masks.npy", test_data[1])
        
        print("Dataset saved successfully!")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()