"""
Real-time data processing and model updating module.
"""

import os
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
import tensorflow as tf
import ee
from scipy.stats import wasserstein_distance
from config import Config
from earth_engine_processor import EarthEngineProcessor
from unet_model import UNetModel

logger = logging.getLogger(__name__)

class RealTimeProcessor:
    """
    Handles real-time satellite data processing and model updates.
    """
    
    def __init__(
        self,
        model: UNetModel,
        ee_processor: EarthEngineProcessor,
        region: ee.Geometry,
        update_interval: int = 24,  # hours
        drift_threshold: float = 0.15,
        batch_size: int = 32
    ):
        """
        Initialize real-time processor.
        
        Args:
            model: Trained U-Net model
            ee_processor: Earth Engine processor instance
            region: Region of interest
            update_interval: Hours between updates
            drift_threshold: Threshold for model drift detection
            batch_size: Batch size for processing
        """
        self.model = model
        self.ee_processor = ee_processor
        self.region = region
        self.update_interval = update_interval
        self.drift_threshold = drift_threshold
        self.batch_size = batch_size
        
        self.reference_distribution = None
        self.is_running = False
        self.processing_thread = None
        
        # Initialize monitoring metrics
        self.metrics_history = {
            'confidence_scores': [],
            'class_distributions': [],
            'drift_scores': [],
            'update_timestamps': []
        }
    
    def start_monitoring(self):
        """Start real-time monitoring thread."""
        if self.is_running:
            logger.warning("Real-time monitoring is already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._monitoring_loop)
        self.processing_thread.start()
        logger.info("Started real-time monitoring")
    
    def stop_monitoring(self):
        """Stop real-time monitoring thread."""
        if not self.is_running:
            logger.warning("Real-time monitoring is not running")
            return
        
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        logger.info("Stopped real-time monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Get latest satellite data
                latest_data = self._get_latest_data()
                
                # Process and validate data
                processed_data = self._process_data(latest_data)
                
                if processed_data is not None:
                    # Check for model drift
                    drift_detected = self._check_model_drift(processed_data)
                    
                    if drift_detected:
                        # Update model if drift detected
                        self._update_model(processed_data)
                    
                    # Update monitoring metrics
                    self._update_metrics(processed_data, drift_detected)
                
                # Wait for next update interval
                time.sleep(self.update_interval * 3600)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def _get_latest_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get latest satellite imagery.
        
        Returns:
            Tuple of (images, labels)
        """
        # Calculate date range for latest data
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=self.update_interval)
        
        # Get Landsat data
        landsat = self.ee_processor.get_landsat_collection(
            region=self.region,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            cloud_cover=Config.EARTH_ENGINE['cloud_cover_threshold']
        )
        
        # Get Sentinel data
        sentinel = self.ee_processor.get_sentinel_collection(
            region=self.region,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            cloud_cover=Config.EARTH_ENGINE['cloud_cover_threshold']
        )
        
        # Get ground truth data
        ground_truth = self.ee_processor.get_ground_truth_data(
            region=self.region,
            year=end_date.year
        )
        
        # Create label masks
        labels = self.ee_processor.create_label_masks(
            ground_truth=ground_truth,
            num_classes=Config.MODEL['num_classes']
        )
        
        # Export data
        landsat_tiles = self.ee_processor.export_image(
            image=landsat.mosaic(),
            region=self.region,
            scale=Config.EARTH_ENGINE['export_scale'],
            filename='latest_landsat',
            max_tiles=Config.EARTH_ENGINE['max_tiles']
        )
        
        sentinel_tiles = self.ee_processor.export_image(
            image=sentinel.mosaic(),
            region=self.region,
            scale=Config.EARTH_ENGINE['export_scale'],
            filename='latest_sentinel',
            max_tiles=Config.EARTH_ENGINE['max_tiles']
        )
        
        label_tiles = self.ee_processor.export_image(
            image=labels,
            region=self.region,
            scale=Config.EARTH_ENGINE['export_scale'],
            filename='latest_labels',
            max_tiles=Config.EARTH_ENGINE['max_tiles']
        )
        
        return (
            np.concatenate([landsat_tiles, sentinel_tiles], axis=-1),
            label_tiles
        )
    
    def _process_data(
        self,
        data: Tuple[np.ndarray, np.ndarray]
    ) -> Optional[tf.data.Dataset]:
        """
        Process and validate new data.
        
        Args:
            data: Tuple of (images, labels)
            
        Returns:
            TensorFlow dataset if valid, None otherwise
        """
        try:
            images, labels = data
            
            # Validate data shapes
            if images.shape[0] == 0 or labels.shape[0] == 0:
                logger.warning("Empty data received")
                return None
            
            if images.shape[:-1] != labels.shape[:-1]:
                logger.error("Mismatched image and label shapes")
                return None
            
            # Create dataset
            dataset = self.ee_processor.create_training_dataset(
                images=[images],
                labels=[labels],
                batch_size=self.batch_size
            )
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return None
    
    def _check_model_drift(self, dataset: tf.data.Dataset) -> bool:
        """
        Check for model drift using new data.
        
        Args:
            dataset: New data for drift detection
            
        Returns:
            True if drift detected, False otherwise
        """
        # Get predictions on new data
        predictions = []
        for images, _ in dataset:
            batch_pred = self.model.predict(images)
            predictions.append(batch_pred)
        predictions = np.concatenate(predictions, axis=0)
        
        # Calculate prediction distribution
        current_distribution = np.mean(predictions, axis=(1, 2))
        
        # Initialize reference distribution if needed
        if self.reference_distribution is None:
            self.reference_distribution = current_distribution
            return False
        
        # Calculate Wasserstein distance for each class
        drift_scores = [
            wasserstein_distance(self.reference_distribution[:, i],
                               current_distribution[:, i])
            for i in range(current_distribution.shape[1])
        ]
        
        # Check if drift exceeds threshold
        drift_detected = any(score > self.drift_threshold for score in drift_scores)
        
        if drift_detected:
            logger.warning(f"Model drift detected. Scores: {drift_scores}")
        
        return drift_detected
    
    def _update_model(self, dataset: tf.data.Dataset):
        """
        Update model with new data.
        
        Args:
            dataset: New training data
        """
        try:
            # Fine-tune model
            self.model.fine_tune(
                train_data=dataset,
                epochs=Config.TRAINING['fine_tune_epochs'],
                batch_size=self.batch_size,
                class_weights=Config.CLASS_WEIGHTS
            )
            
            # Update reference distribution
            self.reference_distribution = None  # Will be updated on next check
            
            logger.info("Model successfully updated")
            
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
    
    def _update_metrics(self, dataset: tf.data.Dataset, drift_detected: bool):
        """
        Update monitoring metrics.
        
        Args:
            dataset: Latest processed data
            drift_detected: Whether drift was detected
        """
        try:
            # Get predictions
            predictions = []
            for images, _ in dataset:
                batch_pred = self.model.predict(images)
                predictions.append(batch_pred)
            predictions = np.concatenate(predictions, axis=0)
            
            # Update metrics
            self.metrics_history['confidence_scores'].append(
                np.mean(np.max(predictions, axis=-1))
            )
            self.metrics_history['class_distributions'].append(
                np.mean(predictions, axis=(0, 1, 2))
            )
            self.metrics_history['drift_scores'].append(drift_detected)
            self.metrics_history['update_timestamps'].append(
                datetime.now().isoformat()
            )
            
            # Keep only recent history
            max_history = 100
            if len(self.metrics_history['update_timestamps']) > max_history:
                for key in self.metrics_history:
                    self.metrics_history[key] = self.metrics_history[key][-max_history:]
            
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        Get current monitoring status and metrics.
        
        Returns:
            Dictionary with monitoring status and metrics
        """
        return {
            'is_running': self.is_running,
            'last_update': self.metrics_history['update_timestamps'][-1]
                if self.metrics_history['update_timestamps'] else None,
            'drift_detected_count': sum(self.metrics_history['drift_scores']),
            'average_confidence': np.mean(self.metrics_history['confidence_scores'])
                if self.metrics_history['confidence_scores'] else None,
            'latest_class_distribution': self.metrics_history['class_distributions'][-1]
                if self.metrics_history['class_distributions'] else None
        }