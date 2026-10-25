import ee
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import logging

from logging_config import setup_logging

class EarthEngineProcessor:
    """
    Handles Earth Engine data preprocessing for land cover classification.
    Includes cloud masking, atmospheric correction, and edge enhancement.
    """
    
    def __init__(self, credentials_path: str):
        """
        Initialize Earth Engine with credentials and setup logging.
        
        Args:
            credentials_path: Path to Earth Engine credentials file
        """
        # Initialize Earth Engine
        ee.Initialize(credentials=ee.ServiceAccountCredentials(credentials_path))
        
        # Setup logging
        self.logger = setup_logging('satellite_processing')
        
        # Define spectral bands for Landsat and Sentinel
        self.landsat_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
        self.sentinel_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
        
        # Define scaling factors
        self.landsat_scale = 30
        self.sentinel_scale = 10
    
    def validate_date_range(self, start_date: str, end_date: str) -> bool:
        """
        Validate date range format and logic.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            True if dates are valid
            
        Raises:
            ValueError: If dates are invalid
        """
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            if start > end:
                raise ValueError("Start date must be before end date")
            
            if end > datetime.now():
                raise ValueError("End date cannot be in the future")
            
            return True
        except ValueError as e:
            raise ValueError(f"Invalid date format: {str(e)}")
    
    def validate_region(self, region: ee.Geometry) -> bool:
        """
        Validate region geometry.
        
        Args:
            region: Region of interest
            
        Returns:
            True if region is valid
            
        Raises:
            ValueError: If region is invalid
        """
        try:
            # Check if region is too large
            area = region.area().getInfo()
            max_area = 1e6  # 1 million square kilometers
            if area > max_area:
                raise ValueError(f"Region area ({area} km²) exceeds maximum allowed ({max_area} km²)")
            
            # Check if region crosses antimeridian
            bounds = region.bounds().getInfo()['coordinates'][0]
            lons = [coord[0] for coord in bounds]
            if max(lons) - min(lons) > 180:
                raise ValueError("Region cannot cross the antimeridian (180° longitude)")
            
            return True
        except Exception as e:
            raise ValueError(f"Invalid region: {str(e)}")
    
    def get_landsat_collection(
        self,
        region: ee.Geometry,
        start_date: str,
        end_date: str,
        cloud_cover: float = 20
    ) -> ee.ImageCollection:
        """
        Get Landsat 8/9 collection with filtering and preprocessing.
        
        Args:
            region: Earth Engine geometry defining the area of interest
            start_date: Start date for collection (YYYY-MM-DD)
            end_date: End date for collection (YYYY-MM-DD)
            cloud_cover: Maximum cloud cover percentage
            
        Returns:
            Filtered and preprocessed Landsat collection
        """
        # Validate inputs
        self.validate_date_range(start_date, end_date)
        self.validate_region(region)
        
        if not 0 <= cloud_cover <= 100:
            raise ValueError("Cloud cover must be between 0 and 100")
            
        # Get both Landsat 8 and 9 collections
        l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
        l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
        
        # Merge collections
        collection = l8.merge(l9)
        
        # Filter collection
        filtered = collection.filterBounds(region) \
                           .filterDate(start_date, end_date) \
                           .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover))
        
        # Validate collection size
        size = filtered.size().getInfo()
        if size == 0:
            raise ValueError("No images found matching the specified criteria")
            
        # Apply preprocessing to each image
        return filtered.map(self._preprocess_landsat)
    
    def get_sentinel_collection(
        self,
        region: ee.Geometry,
        start_date: str,
        end_date: str,
        cloud_cover: float = 20
    ) -> ee.ImageCollection:
        """
        Get Sentinel-2 collection with filtering and preprocessing.
        
        Args:
            region: Earth Engine geometry defining the area of interest
            start_date: Start date for collection (YYYY-MM-DD)
            end_date: End date for collection (YYYY-MM-DD)
            cloud_cover: Maximum cloud cover percentage
            
        Returns:
            Filtered and preprocessed Sentinel collection
            
        Raises:
            ValueError: If input parameters are invalid or no suitable data is found
        """
        # Validate inputs
        self.validate_date_range(start_date, end_date)
        self.validate_region(region)
        
        if not 0 <= cloud_cover <= 100:
            raise ValueError("Cloud cover must be between 0 and 100")
            
        # Get Sentinel-2 Level-2A collection
        collection = ee.ImageCollection('COPERNICUS/S2_SR')
        
        # Filter collection
        filtered = collection.filterBounds(region) \
                           .filterDate(start_date, end_date) \
                           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))
        
        # Validate collection size
        size = filtered.size().getInfo()
        if size == 0:
            raise ValueError("No images found matching the specified criteria")
            
        # Apply preprocessing to each image
        return filtered.map(self._preprocess_sentinel)
    
    def validate_preprocessing_result(self, image: ee.Image) -> Dict[str, Any]:
        """
        Validate preprocessing results.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Dictionary with validation metrics
        """
        # Calculate basic statistics
        stats = image.reduceRegion(
            reducer=ee.Reducer.percentile([0, 2, 50, 98, 100]),
            geometry=image.geometry(),
            scale=30,
            maxPixels=1e13
        ).getInfo()
        
        # Check for invalid values
        has_invalid = any(
            any(v < 0 or v > 1 for v in band_stats.values())
            for band_stats in stats.values()
        )
        
        # Check for sufficient valid pixels
        valid_pixels = image.mask().reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=image.geometry(),
            scale=30,
            maxPixels=1e13
        ).getInfo()
        
        coverage_threshold = 0.7  # 70% minimum coverage
        has_sufficient_coverage = all(
            v >= coverage_threshold for v in valid_pixels.values()
        )
        
        # Check edge enhancement
        edge_metrics = self._validate_edge_enhancement(image)
        
        return {
            'statistics': stats,
            'validation': {
                'has_invalid_values': has_invalid,
                'has_sufficient_coverage': has_sufficient_coverage,
                'edge_quality': edge_metrics
            }
        }
    
    def _validate_edge_enhancement(self, image: ee.Image) -> Dict[str, float]:
        """
        Validate edge enhancement results.
        
        Args:
            image: Edge-enhanced image
            
        Returns:
            Dictionary with edge quality metrics
        """
        # Calculate gradient magnitude
        gradient = image.gradient()
        gradient_mag = gradient.select('x').pow(2).add(gradient.select('y').pow(2)).sqrt()
        
        # Calculate edge metrics
        edge_metrics = gradient_mag.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.stdDev(), None, True
            ),
            geometry=image.geometry(),
            scale=30,
            maxPixels=1e13
        ).getInfo()
        
        # Normalize metrics
        edge_strength = edge_metrics.get('mean', 0)
        edge_consistency = edge_metrics.get('stdDev', 0)
        
        return {
            'edge_strength': edge_strength,
            'edge_consistency': edge_consistency,
            'quality_score': edge_strength / (edge_consistency + 1e-6)
        }

    def _preprocess_landsat(self, image: ee.Image) -> ee.Image:
        """
        Preprocess Landsat image with validation.
        
        Args:
            image: Landsat image
            
        Returns:
            Preprocessed image
            
        Raises:
            ValueError: If preprocessing results are invalid
        """
        try:
            # Get QA band
            qa = image.select('QA_PIXEL')
            
            # Create cloud mask
            cloud_mask = qa.bitwiseAnd(1 << 3).eq(0) \
                          .And(qa.bitwiseAnd(1 << 4).eq(0))  # Clear and no cloud shadow
            
            # Apply cloud mask and select bands
            masked = image.updateMask(cloud_mask).select(self.landsat_bands)
            
            # Apply scaling factors for surface reflectance
            scaled = masked.multiply(0.0000275).add(-0.2)
            
            # Apply atmospheric correction (DOS)
            corrected = self._apply_dos_correction(scaled)
            
            # Apply edge enhancement
            enhanced = self._enhance_edges(corrected)
            
            # Validate results
            validation = self.validate_preprocessing_result(enhanced)
            
            if not validation['validation']['has_sufficient_coverage']:
                raise ValueError("Insufficient valid pixel coverage after preprocessing")
            
            if validation['validation']['has_invalid_values']:
                raise ValueError("Invalid pixel values detected after preprocessing")
            
            # Add validation metrics to image properties
            enhanced = enhanced.set('preprocessing_validation', validation)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error preprocessing Landsat image: {str(e)}")
            raise
    
    def _preprocess_sentinel(self, image: ee.Image) -> ee.Image:
        """
        Preprocess Sentinel image with validation.
        
        Args:
            image: Sentinel image
            
        Returns:
            Preprocessed image
            
        Raises:
            ValueError: If preprocessing results are invalid
        """
        try:
            # Get SCL band for cloud masking
            scl = image.select('SCL')
            
            # Create cloud mask (4=vegetation, 5=bare soil, 6=water)
            cloud_mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
            
            # Apply cloud mask and select bands
            masked = image.updateMask(cloud_mask).select(self.sentinel_bands)
            
            # Apply scaling factors
            scaled = masked.divide(10000)
            
            # Apply atmospheric correction
            corrected = self._apply_dos_correction(scaled)
            
            # Apply edge enhancement
            enhanced = self._enhance_edges(corrected)
            
            # Validate results
            validation = self.validate_preprocessing_result(enhanced)
            
            if not validation['validation']['has_sufficient_coverage']:
                raise ValueError("Insufficient valid pixel coverage after preprocessing")
            
            if validation['validation']['has_invalid_values']:
                raise ValueError("Invalid pixel values detected after preprocessing")
            
            # Add validation metrics to image properties
            enhanced = enhanced.set('preprocessing_validation', validation)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error preprocessing Sentinel image: {str(e)}")
            raise
    
    def _apply_dos_correction(self, image: ee.Image) -> ee.Image:
        """
        Apply Dark Object Subtraction (DOS) atmospheric correction.
        
        Args:
            image: Input image
            
        Returns:
            Atmospherically corrected image
        """
        # Calculate dark objects (1st percentile) for each band
        dark_stats = image.reduceRegion(
            reducer=ee.Reducer.percentile([1]),
            geometry=image.geometry(),
            scale=30,
            maxPixels=1e9
        )
        
        # Create an image of dark object values
        dark_vals = ee.Image.constant(dark_stats.values())
        
        # Subtract dark object values
        return image.subtract(dark_vals)
    
    def _enhance_edges(self, image: ee.Image) -> ee.Image:
        """
        Apply edge enhancement using unsharp masking.
        
        Args:
            image: Input image
            
        Returns:
            Edge-enhanced image
        """
        # Create Gaussian blur
        kernel = ee.Kernel.gaussian(radius=1.5)
        blurred = image.convolve(kernel)
        
        # Calculate unsharp mask
        mask = image.subtract(blurred)
        
        # Add mask back to original image with weighting
        return image.add(mask.multiply(0.7))
    
    def export_image(
        self,
        image: ee.Image,
        region: ee.Geometry,
        scale: int,
        filename: str,
        max_tiles: int = 100
    ) -> List[np.ndarray]:
        """
        Export image tiles for model input.
        
        Args:
            image: Image to export
            region: Region to export
            scale: Export resolution in meters
            filename: Base filename for exported tiles
            max_tiles: Maximum number of tiles to export
            
        Returns:
            List of image tiles as numpy arrays
        """
        try:
            # Get image bounds
            bounds = region.bounds().getInfo()['coordinates'][0]
            
            # Calculate tile size in degrees
            tile_size_m = 256 * scale  # 256 pixels at given scale
            approx_degree = tile_size_m / 111320  # rough conversion to degrees
            
            # Generate tile grid
            tiles = []
            for x in np.arange(bounds[0][0], bounds[2][0], approx_degree):
                for y in np.arange(bounds[0][1], bounds[2][1], approx_degree):
                    if len(tiles) >= max_tiles:
                        break
                    
                    # Create tile geometry
                    tile_geometry = ee.Geometry.Rectangle(
                        [x, y, x + approx_degree, y + approx_degree]
                    )
                    
                    # Get tile data
                    tile_data = image.clip(tile_geometry).getInfo()
                    
                    if tile_data:
                        tiles.append(np.array(tile_data['bands']))
            
            self.logger.info(f"Successfully exported {len(tiles)} tiles")
            return tiles
            
        except Exception as e:
            self.logger.error(f"Error exporting image: {str(e)}")
            raise
    
    def get_ground_truth_data(self, region: ee.Geometry, year: int) -> ee.Image:
        """
        Retrieve and process ground truth data for land cover classification.
        
        Args:
            region: Region of interest
            year: Year for ground truth data
            
        Returns:
            Earth Engine image with ground truth labels
        """
        # Get ESA WorldCover data (10m resolution)
        worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
        
        # Get MODIS Land Cover Type (500m resolution)
        modis = ee.ImageCollection("MODIS/006/MCD12Q1") \
            .filter(ee.Filter.calendarRange(year, year, 'year')) \
            .first() \
            .select('LC_Type1')
        
        # Get Global Forest Change data
        gfc = ee.Image("UMD/hansen/global_forest_change_2021_v1_9")
        
        # Combine and harmonize datasets
        ground_truth = ee.Image.cat([
            worldcover.rename('worldcover'),
            modis.rename('modis'),
            gfc.select(['treecover2000', 'loss', 'gain']).rename(['forest2000', 'forest_loss', 'forest_gain'])
        ])
        
        return ground_truth.clip(region)
    
    def create_label_masks(self, ground_truth: ee.Image, num_classes: int) -> ee.Image:
        """
        Create binary masks for each land cover class.
        
        Args:
            ground_truth: Ground truth image
            num_classes: Number of land cover classes
            
        Returns:
            Image with binary masks for each class
        """
        # Define class mapping based on WorldCover and MODIS
        class_conditions = {
            0: ground_truth.select('worldcover').eq(10),  # Tree cover
            1: ground_truth.select('worldcover').eq(20),  # Shrubland
            2: ground_truth.select('worldcover').eq(30),  # Grassland
            3: ground_truth.select('worldcover').eq(40),  # Cropland
            4: ground_truth.select('worldcover').eq(50),  # Built-up
            5: ground_truth.select('worldcover').eq(60).Or(  # Water bodies
               ground_truth.select('worldcover').eq(80)),   # Permanent water bodies
            6: ground_truth.select('worldcover').eq(70).Or(  # Other
               ground_truth.select('worldcover').eq(90).Or(
               ground_truth.select('worldcover').eq(95).Or(
               ground_truth.select('worldcover').eq(100))))
        }
        
        # Create binary masks
        masks = []
        for i in range(num_classes):
            mask = class_conditions[i].rename(f'class_{i}')
            masks.append(mask)
        
        return ee.Image.cat(masks)
    
    def validate_labels(self, labels: ee.Image, region: ee.Geometry) -> Dict[str, float]:
        """
        Validate label quality and compute class distribution.
        
        Args:
            labels: Image with class labels
            region: Region of interest
            
        Returns:
            Dictionary with validation metrics
        """
        # Calculate class distribution
        class_stats = labels.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=10,
            maxPixels=1e13
        ).getInfo()
        
        # Calculate coverage percentage
        total_pixels = ee.Number(region.area()).divide(100)  # 10m resolution
        coverage = labels.mask().reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=10,
            maxPixels=1e13
        ).getInfo()
        
        return {
            'class_distribution': class_stats,
            'coverage_percentage': coverage
        }

    def create_training_dataset(
        self,
        images: List[np.ndarray],
        labels: List[np.ndarray],
        batch_size: int = 32
    ) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from image tiles and labels.
        
        Args:
            images: List of image tiles
            labels: List of corresponding label tiles
            batch_size: Batch size for training
            
        Returns:
            TensorFlow dataset
        """
        # Validate inputs
        if not images or not labels:
            raise ValueError("Empty images or labels provided")
        if len(images) != len(labels):
            raise ValueError(f"Mismatched number of images ({len(images)}) and labels ({len(labels)})")
        
        # Convert to arrays
        x = np.stack(images)
        y = np.stack(labels)
        
        # Validate shapes
        if x.ndim != 4:
            raise ValueError(f"Expected 4D image array (N, H, W, C), got shape {x.shape}")
        if y.ndim != 4:
            raise ValueError(f"Expected 4D label array (N, H, W, num_classes), got shape {y.shape}")
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        
        # Apply augmentation and batching
        dataset = dataset.map(self._augment_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _augment_data(
        self,
        image: tf.Tensor,
        label: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply data augmentation to image and label pairs.
        
        Args:
            image: Input image
            label: Input label
            
        Returns:
            Augmented image and label
        """
        # Random flip
        if tf.random.uniform([]) > 0.5:
            image = tf.image.flip_left_right(image)
            label = tf.image.flip_left_right(label)
        
        if tf.random.uniform([]) > 0.5:
            image = tf.image.flip_up_down(image)
            label = tf.image.flip_up_down(label)
        
        # Random rotation
        angle = tf.random.uniform([], -0.2, 0.2)
        image = tf.image.rotate(image, angle)
        label = tf.image.rotate(label, angle)
        
        # Random brightness and contrast (image only)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        
        return image, label