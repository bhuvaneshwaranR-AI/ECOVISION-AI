import os
import numpy as np
from PIL import Image

def generate_sample_data(num_samples=10, image_size=256):
    """Generate sample satellite imagery and masks for testing."""
    try:
        # Create directories if they don't exist
        os.makedirs("data/sentinel_images", exist_ok=True)
        
        for i in range(num_samples):
            # Generate sample satellite image (RGB)
            image = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
            
            # Generate sample mask (binary)
            mask = np.zeros((image_size, image_size), dtype=np.uint8)
            # Add some random "deforested" regions
            num_regions = np.random.randint(1, 5)
            for _ in range(num_regions):
                x = np.random.randint(0, image_size)
                y = np.random.randint(0, image_size)
                radius = np.random.randint(10, 50)
                for dx in range(-radius, radius):
                    for dy in range(-radius, radius):
                        if dx*dx + dy*dy <= radius*radius:
                            px = min(max(x + dx, 0), image_size-1)
                            py = min(max(y + dy, 0), image_size-1)
                            mask[px, py] = 255
            
            # Save image and mask
            image_pil = Image.fromarray(image)
            mask_pil = Image.fromarray(mask)
            
            image_pil.save(f"data/sentinel_images/sample_image_{i}.png")
            mask_pil.save(f"data/sentinel_images/sample_mask_{i}.png")
            
        print(f"Generated {num_samples} sample images and masks")
        print("Sample data saved in data/sentinel_images/")
        
    except Exception as e:
        print(f"Error generating sample data: {str(e)}")
        raise

def main():
    try:
        # Generate sample data
        generate_sample_data()
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()