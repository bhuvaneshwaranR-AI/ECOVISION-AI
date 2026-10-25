import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class DatasetManager:
    def __init__(self, preprocessed_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
        """
        Initialize the dataset manager for deforestation detection.
        
        Args:
            preprocessed_dir: Directory containing preprocessed image tiles
            output_dir: Directory to store the final dataset
            train_ratio: Proportion of data for training (default: 0.7)
            val_ratio: Proportion of data for validation (default: 0.15)
            (remaining will be used for testing)
        """
        self.preprocessed_dir = preprocessed_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    def load_tile_pairs(self):
        """Load and pair tiles from different time periods."""
        try:
            tiles_dir = os.path.join(self.preprocessed_dir, 'tiles')
            metadata_dir = os.path.join(self.preprocessed_dir, 'metadata')
            
            # Load metadata for both time periods
            pre_monsoon_metadata = None
            previous_season_metadata = None
            
            for file in os.listdir(metadata_dir):
                if file.endswith('_tiles_metadata.json'):
                    with open(os.path.join(metadata_dir, file), 'r') as f:
                        metadata = json.load(f)
                        if 'Pre_Monsoon' in file:
                            pre_monsoon_metadata = metadata
                        elif 'Previous_Season' in file:
                            previous_season_metadata = metadata
            
            if not (pre_monsoon_metadata and previous_season_metadata):
                raise ValueError("Could not find metadata for both time periods")
            
            # Match tiles between time periods
            tile_pairs = []
            for pre_tile in pre_monsoon_metadata['tiles']:
                for prev_tile in previous_season_metadata['tiles']:
                    if (pre_tile['x'] == prev_tile['x'] and 
                        pre_tile['y'] == prev_tile['y']):
                        tile_pairs.append({
                            'current': pre_tile,
                            'previous': prev_tile
                        })
            
            return tile_pairs
            
        except Exception as e:
            print(f"Error loading tile pairs: {str(e)}")
            raise
    
    def create_deforestation_mask(self, current_ndvi, previous_ndvi, threshold=0.2):
        """
        Create deforestation mask by comparing NDVI values.
        
        Args:
            current_ndvi: NDVI values for current period
            previous_ndvi: NDVI values for previous period
            threshold: NDVI difference threshold for deforestation
        """
        # Detect significant NDVI decrease
        ndvi_decrease = previous_ndvi - current_ndvi
        deforestation_mask = (ndvi_decrease > threshold).astype(np.float32)
        
        return deforestation_mask
    
    def prepare_dataset(self):
        """Prepare the complete dataset for training."""
        try:
            # Load tile pairs
            print("Loading tile pairs...")
            tile_pairs = self.load_tile_pairs()
            
            # Prepare data arrays
            X = []  # Input features
            y = []  # Ground truth masks
            
            print("\nCreating training samples...")
            for pair in tqdm(tile_pairs):
                # Load current period data
                current_path = os.path.join(
                    self.preprocessed_dir,
                    'tiles',
                    f"Pre_Monsoon_indices_tile_{pair['current']['x']}_{pair['current']['y']}.npy"
                )
                current_data = np.load(current_path)
                
                # Load previous period data
                previous_path = os.path.join(
                    self.preprocessed_dir,
                    'tiles',
                    f"Previous_Season_indices_tile_{pair['previous']['x']}_{pair['previous']['y']}.npy"
                )
                previous_data = np.load(previous_path)
                
                # Create input features (concatenate relevant bands and indices)
                features = np.concatenate([
                    current_data,   # Current period indices
                    previous_data,  # Previous period indices
                ], axis=0)
                
                # Create deforestation mask
                mask = self.create_deforestation_mask(
                    current_data[0],  # Current NDVI
                    previous_data[0]  # Previous NDVI
                )
                
                X.append(features)
                y.append(mask)
            
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Split dataset
            print("\nSplitting dataset...")
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y,
                test_size=(1-self.train_ratio-self.val_ratio),
                random_state=42
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=self.val_ratio/(self.train_ratio+self.val_ratio),
                random_state=42
            )
            
            # Save datasets
            print("\nSaving datasets...")
            np.save(os.path.join(self.output_dir, 'train', 'X_train.npy'), X_train)
            np.save(os.path.join(self.output_dir, 'train', 'y_train.npy'), y_train)
            np.save(os.path.join(self.output_dir, 'val', 'X_val.npy'), X_val)
            np.save(os.path.join(self.output_dir, 'val', 'y_val.npy'), y_val)
            np.save(os.path.join(self.output_dir, 'test', 'X_test.npy'), X_test)
            np.save(os.path.join(self.output_dir, 'test', 'y_test.npy'), y_test)
            
            # Save dataset metadata
            metadata = {
                'dataset_size': len(X),
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test),
                'feature_shape': X.shape[1:],
                'mask_shape': y.shape[1:],
                'deforestation_threshold': 0.2,
                'train_ratio': self.train_ratio,
                'val_ratio': self.val_ratio,
                'test_ratio': 1-self.train_ratio-self.val_ratio,
                'random_seed': 42
            }
            
            with open(os.path.join(self.output_dir, 'dataset_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=4)
            
            print("\nDataset preparation complete!")
            print(f"Total samples: {len(X)}")
            print(f"Training samples: {len(X_train)}")
            print(f"Validation samples: {len(X_val)}")
            print(f"Test samples: {len(X_test)}")
            print(f"Feature shape: {X.shape[1:]}")
            print(f"Mask shape: {y.shape[1:]}")
            print(f"\nDataset saved in: {self.output_dir}")
            
            return metadata
            
        except Exception as e:
            print(f"Error preparing dataset: {str(e)}")
            raise

def main():
    try:
        # Set up directories
        base_dir = os.path.dirname(os.path.abspath(__file__))
        preprocessed_dir = os.path.join(base_dir, 'preprocessed_data')
        dataset_dir = os.path.join(base_dir, 'dataset')
        
        # Initialize dataset manager
        manager = DatasetManager(preprocessed_dir, dataset_dir)
        
        # Prepare dataset
        manager.prepare_dataset()
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()