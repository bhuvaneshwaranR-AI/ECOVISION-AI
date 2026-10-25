from typing import Dict, Any
import os

class Config:
    """
    Configuration settings for the land cover classification project.
    """
    
    # Earth Engine settings
    EARTH_ENGINE = {
        'credentials_path': os.path.join('credentials', 'earth-engine-key.json'),
        'cloud_cover_threshold': 20,
        'export_scale': 30,  # meters per pixel
        'max_tiles': 1000,
        'tile_size': 256  # pixels
    }
    
    # Land cover classes
    LAND_COVER_CLASSES = {
        0: 'Water',
        1: 'Urban',
        2: 'Forest',
        3: 'Agriculture',
        4: 'Grassland',
        5: 'Barren',
        6: 'Wetland'
    }
    
    # Land cover class colors (in hex format)
    LAND_COVER_COLORS = [
        '#0077BE',  # Water (Blue)
        '#C0C0C0',  # Urban (Gray)
        '#228B22',  # Forest (Forest Green)
        '#FFD700',  # Agriculture (Golden)
        '#90EE90',  # Grassland (Light Green)
        '#DEB887',  # Barren (Burlywood)
        '#6B8E23'   # Wetland (Olive Drab)
    ]
    
    # Class weights for handling imbalanced data
    CLASS_WEIGHTS = {
        0: 1.0,  # Water
        1: 2.0,  # Urban (less common)
        2: 1.0,  # Forest
        3: 1.0,  # Agriculture
        4: 1.5,  # Grassland
        5: 2.0,  # Barren (less common)
        6: 2.5   # Wetland (rare)
    }
    
    # Model architecture
    MODEL = {
        'input_shape': (256, 256, 6),  # (height, width, channels)
        'num_classes': 7,
        'filters_base': 64,
        'num_blocks': 5,
        'dropout_rate': 0.3,
        'batch_norm': True,
        'confidence_threshold': 0.8,
        'ensemble': {
            'enabled': True,  # Enable model ensemble
            'model_types': ['standard', 'resnet', 'efficientnet'],  # Types of models to include
            'model_weights': [1.0, 1.0, 1.0],  # Weights for each model
            'cross_validation': {
                'enabled': True,  # Enable cross-validation
                'folds': 5,  # Number of folds
                'shuffle': True  # Shuffle data before splitting
            }
        }
    }
    
    # Training settings
    TRAINING = {
        'batch_size': 32,
        'epochs': 100,
        'initial_learning_rate': 0.001,
        'min_learning_rate': 1e-6,
        'validation_split': 0.2,
        'early_stopping_patience': 15
    }
    
    # Fine-tuning settings
    FINE_TUNING = {
        'batch_size': 16,
        'epochs': 20,
        'learning_rate': 0.0001,
        'early_stopping_patience': 5
    }
    
    # Data augmentation
    AUGMENTATION = {
        'rotation_range': 20,
        'flip_horizontal': True,
        'flip_vertical': True,
        'brightness_range': 0.2,
        'contrast_range': (0.8, 1.2)
    }
    
    # Model evaluation
    EVALUATION = {
        'min_accuracy': 0.85,
        'iou_threshold': 0.5,
        'confidence_threshold': 0.8,
        'boundary_thickness': 2
    }
    
    # Drift monitoring
    DRIFT_MONITORING = {
        'distribution_threshold': 0.1,
        'confidence_threshold': 0.1,
        'check_frequency': 1000,  # samples
        'min_samples_for_drift': 500,  # Minimum samples needed for drift detection
        'max_drift_ratio': 0.2  # Maximum allowed drift before retraining
    }
    
    # Real-time processing
    REAL_TIME = {
        'update_interval_hours': 24,  # How often to check for new data
        'min_update_interval': 6,  # Minimum hours between updates
        'max_update_interval': 168,  # Maximum hours (1 week)
        'batch_size': 32,  # Batch size for real-time processing
        'max_queue_size': 1000,  # Maximum samples to queue for processing
        'monitoring_interval': 300,  # Seconds between monitoring status updates
        'edge_enhancement': {
            'kernel_size': 3,
            'sigma': 1.0,
            'alpha': 1.5
        },
        'quality_thresholds': {
            'min_valid_pixels': 0.8,  # Minimum fraction of valid pixels
            'max_cloud_cover': 0.2,  # Maximum cloud cover
            'min_edge_strength': 0.1  # Minimum edge strength for boundary detection
        }
    }
    
    # File paths
    PATHS = {
        'model_dir': os.path.join('models', 'land_cover'),
        'logs_dir': os.path.join('logs', 'land_cover'),
        'data_dir': os.path.join('data', 'satellite'),
        'output_dir': os.path.join('output', 'predictions')
    }
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        for path in cls.PATHS.values():
            os.makedirs(path, exist_ok=True)
    
    @classmethod
    def get_model_path(cls, model_name: str) -> str:
        """Get full path for model file."""
        return os.path.join(cls.PATHS['model_dir'], f'{model_name}.h5')
    
    @classmethod
    def get_log_path(cls, component: str) -> str:
        """Get full path for log file."""
        return os.path.join(cls.PATHS['logs_dir'], f'{component}.log')
    
    @classmethod
    def update_config(cls, updates: Dict[str, Any]) -> None:
        """
        Update configuration settings.
        
        Args:
            updates: Dictionary of settings to update
        """
        for key, value in updates.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise ValueError(f"Invalid configuration key: {key}")
    
    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            True if configuration is valid
        """
        # Check required paths
        if not os.path.exists(cls.EARTH_ENGINE['credentials_path']):
            raise ValueError("Earth Engine credentials not found")
        
        # Validate model settings
        if cls.MODEL['num_classes'] != len(cls.LAND_COVER_CLASSES):
            raise ValueError("Number of classes mismatch")
        
        # Validate class weights
        if set(cls.CLASS_WEIGHTS.keys()) != set(cls.LAND_COVER_CLASSES.keys()):
            raise ValueError("Class weights do not match land cover classes")
        
        # Validate evaluation thresholds
        if not 0 <= cls.EVALUATION['confidence_threshold'] <= 1:
            raise ValueError("Invalid confidence threshold")
        
        return True