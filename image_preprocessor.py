import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import json
from tqdm import tqdm

class ImagePreprocessor:
    def __init__(self, input_dir, output_dir, tile_size=256, overlap=32):
        """
        Initialize the image preprocessor for satellite imagery.
        
        Args:
            input_dir: Directory containing organized satellite data
            output_dir: Directory to store preprocessed data
            tile_size: Size of image tiles (default: 256x256)
            overlap: Overlap between adjacent tiles (default: 32 pixels)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.tile_size = tile_size
        self.overlap = overlap
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'tiles'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)
    
    def normalize_band(self, data, min_val=None, max_val=None):
        """
        Normalize band data to [0,1] range using min-max scaling.
        
        Args:
            data: Input band data
            min_val: Optional minimum value for normalization
            max_val: Optional maximum value for normalization
        """
        if min_val is None:
            min_val = np.percentile(data, 1)  # 1st percentile to handle outliers
        if max_val is None:
            max_val = np.percentile(data, 99)  # 99th percentile to handle outliers
        
        normalized = np.clip((data - min_val) / (max_val - min_val), 0, 1)
        return normalized, min_val, max_val
    
    def create_tiles(self, image_path, time_period, image_type):
        """
        Create overlapping tiles from satellite imagery.
        
        Args:
            image_path: Path to the input GeoTIFF file
            time_period: Time period identifier (e.g., 'Pre_Monsoon')
            image_type: Type of image ('spectral', 'indices', or 'qa')
        """
        try:
            with rasterio.open(image_path) as src:
                # Get image dimensions
                height = src.height
                width = src.width
                num_bands = src.count
                
                # Calculate tile positions with overlap
                x_tiles = range(0, width - self.overlap, self.tile_size - self.overlap)
                y_tiles = range(0, height - self.overlap, self.tile_size - self.overlap)
                
                tiles_data = []
                normalization_params = {}
                
                # Process each tile
                for y in tqdm(y_tiles, desc=f"Processing {time_period} {image_type} tiles"):
                    for x in x_tiles:
                        # Read tile data
                        window = Window(x, y, self.tile_size, self.tile_size)
                        tile = src.read(window=window)
                        
                        # Skip tiles with no data or excessive clouds (for QA images)
                        if image_type == 'qa' and np.mean(tile[0] > 0) > 0.5:  # More than 50% clouds
                            continue
                        
                        # Normalize each band
                        normalized_tile = np.zeros_like(tile, dtype=np.float32)
                        for band in range(num_bands):
                            normalized_tile[band], min_val, max_val = self.normalize_band(tile[band])
                            normalization_params[f'band_{band+1}'] = {'min': float(min_val), 'max': float(max_val)}
                        
                        # Save tile metadata
                        tile_info = {
                            'x': int(x),
                            'y': int(y),
                            'width': self.tile_size,
                            'height': self.tile_size,
                            'time_period': time_period,
                            'image_type': image_type,
                            'original_file': os.path.basename(image_path),
                            'normalization': normalization_params
                        }
                        tiles_data.append(tile_info)
                        
                        # Save normalized tile
                        tile_filename = f"{time_period}_{image_type}_tile_{x}_{y}.npy"
                        tile_path = os.path.join(self.output_dir, 'tiles', tile_filename)
                        np.save(tile_path, normalized_tile)
                
                # Save tiles metadata
                metadata_filename = f"{time_period}_{image_type}_tiles_metadata.json"
                metadata_path = os.path.join(self.output_dir, 'metadata', metadata_filename)
                with open(metadata_path, 'w') as f:
                    json.dump({
                        'tiles': tiles_data,
                        'preprocessing_params': {
                            'tile_size': self.tile_size,
                            'overlap': self.overlap,
                            'num_bands': num_bands
                        }
                    }, f, indent=4)
                
                return len(tiles_data)
        
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            raise
    
    def preprocess_all(self):
        """Process all satellite images in the input directory."""
        try:
            total_tiles = 0
            
            # Process each time period
            for time_period in ['Pre_Monsoon', 'Previous_Season']:
                period_dir = os.path.join(self.input_dir, time_period)
                if not os.path.exists(period_dir):
                    continue
                
                print(f"\nProcessing {time_period} data...")
                
                # Process each image type
                for image_type in ['spectral', 'indices', 'qa']:
                    type_dir = os.path.join(period_dir, image_type)
                    if not os.path.exists(type_dir):
                        continue
                    
                    # Process each GeoTIFF file
                    for file in os.listdir(type_dir):
                        if file.endswith('.tif'):
                            image_path = os.path.join(type_dir, file)
                            num_tiles = self.create_tiles(image_path, time_period, image_type)
                            total_tiles += num_tiles
                            print(f"Created {num_tiles} tiles from {file}")
            
            print(f"\nPreprocessing complete! Total tiles created: {total_tiles}")
            print(f"Preprocessed data saved in: {self.output_dir}")
            
            return total_tiles
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise

def main():
    try:
        # Set up directories
        base_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(base_dir, 'processed_data')
        output_dir = os.path.join(base_dir, 'preprocessed_data')
        
        # Initialize preprocessor
        preprocessor = ImagePreprocessor(input_dir, output_dir)
        
        # Run preprocessing
        preprocessor.preprocess_all()
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()