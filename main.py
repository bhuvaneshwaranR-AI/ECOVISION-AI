import os
import argparse
import ee
import json
import datetime
import time
import numpy as np
import logging
from typing import Dict, Optional
import tensorflow as tf

from deforestation_detection import DeforestationDetector
from image_preprocessor import ImagePreprocessor
from dataset_manager import DatasetManager
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from earth_engine_processor import EarthEngineProcessor
from unet_model import UNetModel
from config import Config
from logging_config import setup_logging
from visualization import VisualizationTools
from export_manager import ExportManager
from model_ensemble import EnsembleModel
from real_time_processor import RealTimeProcessor

def setup_directories(base_dir: str) -> Dict[str, str]:
    """
    Create necessary directories for both deforestation detection and land cover classification.
    
    Args:
        base_dir: Base directory path
        
    Returns:
        Dictionary of directory paths
    """
    # Original deforestation detection directories
    dirs = {
        'downloads': os.path.join(base_dir, 'downloads'),
        'processed_data': os.path.join(base_dir, 'processed_data'),
        'dataset': os.path.join(base_dir, 'dataset'),
        'models': os.path.join(base_dir, 'models'),
        'evaluation': os.path.join(base_dir, 'evaluation_results')
    }
    
    # Add land cover classification directories
    dirs.update({
        'land_cover_data': os.path.join(base_dir, 'data', 'land_cover'),
        'land_cover_models': os.path.join(base_dir, 'models', 'land_cover'),
        'land_cover_predictions': os.path.join(base_dir, 'predictions', 'land_cover'),
        'logs': os.path.join(base_dir, 'logs')
    })
    
    # Create all directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def save_pipeline_metadata(base_dir: str, args: argparse.Namespace) -> Dict:
    """
    Save pipeline configuration and metadata for both tasks.
    
    Args:
        base_dir: Base directory path
        args: Command line arguments
        
    Returns:
        Dictionary containing pipeline metadata
    """
    metadata = {
        'pipeline_timestamp': datetime.datetime.now().isoformat(),
        'configuration': vars(args),
        'aoi': {
            'west': 78.0,
            'south': 20.0,
            'east': 82.0,
            'north': 24.0,
            'region': 'Central India',
            'states': ['Madhya Pradesh', 'Chhattisgarh']
        },
        'deforestation_detection': {
            'satellite_data': {
                'source': 'Sentinel-2',
                'cloud_cover_threshold': 20,
                'time_periods': ['Pre_Monsoon', 'Previous_Season']
            },
            'model': {
                'architecture': 'U-Net',
                'input_shape': [256, 256, 6],
                'task': 'Deforestation Detection'
            }
        },
        'land_cover_classification': {
            'satellite_data': {
                'sources': ['Landsat-8/9', 'Sentinel-2'],
                'cloud_cover_threshold': Config.EARTH_ENGINE['cloud_cover_threshold'],
                'export_scale': Config.EARTH_ENGINE['export_scale'],
                'tile_size': Config.EARTH_ENGINE['tile_size']
            },
            'model': {
                'architecture': 'Enhanced U-Net',
                'input_shape': Config.MODEL['input_shape'],
                'num_classes': Config.MODEL['num_classes'],
                'class_names': list(Config.LAND_COVER_CLASSES.values()),
                'confidence_threshold': Config.MODEL['confidence_threshold']
            },
            'training': {
                'batch_size': Config.TRAINING['batch_size'],
                'epochs': Config.TRAINING['epochs'],
                'learning_rate': Config.TRAINING['initial_learning_rate'],
                'class_weights': Config.CLASS_WEIGHTS
            },
            'evaluation': {
                'min_accuracy': Config.EVALUATION['min_accuracy'],
                'iou_threshold': Config.EVALUATION['iou_threshold'],
                'boundary_thickness': Config.EVALUATION['boundary_thickness']
            }
        }
    }
    
    # Save metadata
    metadata_path = os.path.join(base_dir, 'pipeline_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return metadata

def run_deforestation_pipeline(args: argparse.Namespace, dirs: Dict[str, str]) -> None:
    """
    Run the deforestation detection pipeline.
    
    Args:
        args: Command line arguments
        dirs: Directory paths
    """
    # Setup logging
    loggers = setup_logging()
    main_logger = loggers['satellite']  # Use satellite logger for deforestation detection
    
    # Step 1: Earth Engine Processing
    if not args.skip_earth_engine:
        main_logger.info("=== Step 1: Earth Engine Processing (Deforestation) ===")
        detector = DeforestationDetector()
        detector.main()
        main_logger.info("Earth Engine processing complete!")
        main_logger.info("Please download the exported files from Google Drive")
        main_logger.info(f"and place them in: {dirs['downloads']}")
        input("\nPress Enter when the files are downloaded...")
    
    # Step 2: Image Preprocessing
    if not args.skip_preprocessing:
        print("\n=== Step 2: Image Preprocessing (Deforestation) ===")
        # Organize downloaded data
        detector = DeforestationDetector()
        detector.organize_local_data(
            downloads_dir=dirs['downloads'],
            output_dir=dirs['processed_data']
        )
        
        # Preprocess images
        preprocessor = ImagePreprocessor(
            input_dir=dirs['processed_data'],
            output_dir=dirs['processed_data']
        )
        preprocessor.preprocess_all()
    
    # Step 3: Dataset Preparation
    print("\n=== Step 3: Dataset Preparation (Deforestation) ===")
    dataset_manager = DatasetManager(
        preprocessed_dir=dirs['processed_data'],
        output_dir=dirs['dataset']
    )
    dataset_manager.prepare_dataset()
    
    # Step 4: Model Training
    if not args.skip_training:
        print("\n=== Step 4: Model Training (Deforestation) ===")
        trainer = ModelTrainer(
            dataset_dir=dirs['dataset'],
            model_dir=dirs['models'],
            batch_size=args.batch_size
        )
        history, test_metrics = trainer.train_model(epochs=args.epochs)
        trainer.plot_training_history(history)
    
    # Step 5: Model Evaluation
    print("\n=== Step 5: Model Evaluation (Deforestation) ===")
    evaluator = ModelEvaluator(
        model_path=os.path.join(dirs['models'], 'final_model.h5'),
        dataset_dir=dirs['dataset'],
        output_dir=dirs['evaluation']
    )
    evaluator.generate_evaluation_report()

def run_land_cover_pipeline(args: argparse.Namespace, dirs: Dict[str, str]) -> None:
    """
    Run the land cover classification pipeline.
    
    Args:
        args: Command line arguments
        dirs: Directory paths
    """
    # Initialize logger
    logger = setup_logging('land_cover_pipeline')
    
    try:
        # Initialize Earth Engine processor
        ee_processor = EarthEngineProcessor(Config.EARTH_ENGINE['credentials_path'])
        
        # Define region of interest
        region = ee.Geometry.Rectangle([
            78.0, 20.0,  # Central India coordinates
            82.0, 24.0
        ])
        
        # Step 1: Data Collection and Preprocessing
        print("\n=== Step 1: Earth Engine Processing (Land Cover) ===")
        if not args.skip_earth_engine:
            # Get satellite data
            landsat_collection = ee_processor.get_landsat_collection(
                region=region,
                start_date='2023-01-01',
                end_date='2023-12-31',
                cloud_cover=Config.EARTH_ENGINE['cloud_cover_threshold']
            )
            
            sentinel_collection = ee_processor.get_sentinel_collection(
                region=region,
                start_date='2023-01-01',
                end_date='2023-12-31',
                cloud_cover=Config.EARTH_ENGINE['cloud_cover_threshold']
            )
            
            # Get ground truth data
            ground_truth = ee_processor.get_ground_truth_data(
                region=region,
                year=2023
            )
            
            # Create label masks
            labels = ee_processor.create_label_masks(
                ground_truth=ground_truth,
                num_classes=Config.MODEL['num_classes']
            )
            
            # Export preprocessed data
            landsat_tiles = ee_processor.export_image(
                image=landsat_collection.mosaic(),
                region=region,
                scale=Config.EARTH_ENGINE['export_scale'],
                filename='landsat_tiles',
                max_tiles=Config.EARTH_ENGINE['max_tiles']
            )
            
            sentinel_tiles = ee_processor.export_image(
                image=sentinel_collection.mosaic(),
                region=region,
                scale=Config.EARTH_ENGINE['export_scale'],
                filename='sentinel_tiles',
                max_tiles=Config.EARTH_ENGINE['max_tiles']
            )
            
            label_tiles = ee_processor.export_image(
                image=labels,
                region=region,
                scale=Config.EARTH_ENGINE['export_scale'],
                filename='label_tiles',
                max_tiles=Config.EARTH_ENGINE['max_tiles']
            )
            
            # Save tiles
            np.save(os.path.join(dirs['land_cover_data'], 'landsat_tiles.npy'), landsat_tiles)
            np.save(os.path.join(dirs['land_cover_data'], 'sentinel_tiles.npy'), sentinel_tiles)
            np.save(os.path.join(dirs['land_cover_data'], 'label_tiles.npy'), label_tiles)
        
        # Step 2: Model Setup
        print("\n=== Step 2: Model Setup (Land Cover) ===")
        ensemble = EnsembleModel(
            input_shape=Config.MODEL['input_shape'],
            num_classes=Config.MODEL['num_classes']
        )
        
        # Add different model variants to the ensemble
        print("Adding models to ensemble...")
        ensemble.add_model('standard', filters_base=Config.MODEL['filters_base'], weight=1.0)
        ensemble.add_model('resnet', filters_base=Config.MODEL['filters_base'], weight=1.0)
        ensemble.add_model('efficientnet', filters_base=Config.MODEL['filters_base'], weight=1.0)
        
        # Compile all models
        ensemble.compile_models(
            learning_rate=Config.TRAINING['initial_learning_rate'],
            class_weights=Config.CLASS_WEIGHTS
        )
        
        # Step 3: Training
        if not args.skip_training:
            print("\n=== Step 3: Model Training (Land Cover) ===")
            # Load training data
            landsat_tiles = np.load(os.path.join(dirs['land_cover_data'], 'landsat_tiles.npy'))
            sentinel_tiles = np.load(os.path.join(dirs['land_cover_data'], 'sentinel_tiles.npy'))
            label_tiles = np.load(os.path.join(dirs['land_cover_data'], 'label_tiles.npy'))
            
            # Create training dataset
            train_dataset = ee_processor.create_training_dataset(
                images=[np.concatenate([l, s], axis=-1) for l, s in zip(landsat_tiles, sentinel_tiles)],
                labels=label_tiles,
                batch_size=Config.TRAINING['batch_size']
            )
            
            # Create validation dataset from last 100 tiles
            val_size = 100
            val_dataset = ee_processor.create_training_dataset(
                images=[np.concatenate([l, s], axis=-1) for l, s in zip(
                    landsat_tiles[-val_size:], sentinel_tiles[-val_size:]
                )],
                labels=label_tiles[-val_size:],
                batch_size=Config.TRAINING['batch_size']
            )
            
            # Create callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(dirs['logs'], 'ensemble'),
                    histogram_freq=1
                )
            ]
            
            # Train ensemble
            print("\nTraining ensemble models...")
            histories = ensemble.train(
                train_data=train_dataset,
                validation_data=val_dataset,
                epochs=Config.TRAINING['epochs'],
                callbacks=callbacks
            )
            
            # Save ensemble models
            ensemble_dir = os.path.join(dirs['land_cover_models'], 'ensemble')
            ensemble.save_models(ensemble_dir)
            print(f"\nEnsemble models saved to: {ensemble_dir}")
        
        # Step 4: Evaluation
        print("\n=== Step 4: Model Evaluation (Land Cover) ===")
        # Load ensemble models if not training
        if args.skip_training:
            ensemble_dir = os.path.join(dirs['land_cover_models'], 'ensemble')
            ensemble = EnsembleModel.load_models(ensemble_dir)
        
        # Create test dataset from last 100 tiles
        test_images = [np.concatenate([l, s], axis=-1) for l, s in zip(
            landsat_tiles[-100:], sentinel_tiles[-100:]
        )]
        test_images = np.array(test_images)
        test_labels = label_tiles[-100:]
        
        # Evaluate ensemble
        print("\nEvaluating ensemble performance...")
        metrics = ensemble.evaluate(
            test_images,
            test_labels,
            confidence_threshold=Config.EVALUATION['confidence_threshold']
        )
        
        # Generate and save sample predictions
        class_probs, confidence = ensemble.predict(test_images[:5])
        
        # Initialize visualization tools
        vis_tools = VisualizationTools(
            model=ensemble.models[0],  # Use first model for visualization
            class_names=Config.LAND_COVER_CLASSES.values(),
            class_colors=Config.LAND_COVER_COLORS
        )
        
        # Create visualization directory
        vis_dir = os.path.join(dirs['evaluation'], 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate advanced visualizations for sample predictions
        print("\nGenerating advanced visualizations...")
        for i, (image, pred) in enumerate(zip(test_images[:5], class_probs)):
            # Basic visualization with edge detection and uncertainty
            vis_tools.plot_advanced_visualization(
                image=image,
                prediction=np.concatenate([pred, confidence[i:i+1]], axis=-1),
                output_path=os.path.join(vis_dir, f'advanced_vis_{i+1}.png')
            )
            
            # Temporal changes (if applicable)
            if i < len(test_images) - 1:
                vis_tools.plot_temporal_changes(
                    image1=test_images[i],
                    image2=test_images[i+1],
                    output_path=os.path.join(vis_dir, f'temporal_changes_{i+1}.png')
                )
        
        # Create comprehensive analysis report
        print("Generating analysis report...")
        vis_tools.create_analysis_report(
            images=test_images[:5],
            predictions=[np.concatenate([p, c[None, ...]], axis=-1) for p, c in zip(class_probs, confidence)],
            output_dir=vis_dir
        )
        
        # Initialize export manager
        export_manager = ExportManager(os.path.join(dirs['evaluation'], 'exports'))
        
        # Export predictions and visualizations
        print("\nExporting analysis results...")
        export_metadata = {
            'model_type': 'ensemble',
            'ensemble_config': {
                'num_models': len(ensemble.models),
                'model_types': ['standard', 'resnet', 'efficientnet']
            },
            'metrics': metrics,
            'timestamp': datetime.datetime.now().isoformat(),
            'bounds': {
                'west': 78.0,
                'south': 20.0,
                'east': 82.0,
                'north': 24.0
            }
        }
        
        # Export predictions
        export_dir = export_manager.export_predictions(
            predictions=class_probs,
            confidence=confidence,
            metadata=export_metadata,
            region_id='central_india'
        )
        
        # Export visualizations
        export_manager.export_visualizations(vis_dir, export_dir)
        
        # Generate PDF report
        print("Generating PDF report...")
        report_path = export_manager.generate_report(
            export_dir=export_dir,
            class_names=list(Config.LAND_COVER_CLASSES.values()),
            include_visualizations=True
        )
        
        # Export GeoTIFF
        print("Exporting GeoTIFF...")
        geotiff_path = os.path.join(export_dir, 'predictions.tif')
        try:
            export_manager.export_to_geotiff(
                predictions=class_probs[0],
                confidence=confidence[0],
                metadata=export_metadata,
                output_path=geotiff_path
            )
        except Exception as e:
            print(f"Warning: GeoTIFF export failed: {str(e)}")
        
        # Export statistics
        print("Exporting statistics...")
        stats_path = os.path.join(export_dir, 'statistics.xlsx')
        export_manager.export_statistics(
            predictions=class_probs[0],
            confidence=confidence[0],
            class_names=list(Config.LAND_COVER_CLASSES.values()),
            output_path=stats_path
        )
        
        # Save evaluation results
        results = {
            'metrics': metrics,
            'model_type': 'ensemble',
            'ensemble_config': {
                'num_models': len(ensemble.models),
                'model_types': ['standard', 'resnet', 'efficientnet']
            }
        }
        
        with open(os.path.join(dirs['land_cover_predictions'], 'ensemble_evaluation.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Step 5: Real-time Processing Setup
        print("\n=== Step 5: Real-time Processing Setup (Land Cover) ===")
        real_time_processor = RealTimeProcessor(
            model=ensemble.models[0],
            ee_processor=ee_processor,
            region=region,
            update_interval=Config.REAL_TIME['update_interval_hours'],
            drift_threshold=Config.DRIFT_MONITORING['distribution_threshold'],
            batch_size=Config.TRAINING['batch_size']
        )
        
        # Start real-time monitoring if requested
        if args.enable_monitoring:
            print("\nStarting real-time monitoring...")
            real_time_processor.start_monitoring()
            
            try:
                while True:
                    # Get current status
                    status = real_time_processor.get_monitoring_status()
                    
                    # Print monitoring metrics
                    print("\nMonitoring Status:")
                    print(f"- Last Update: {status['last_update']}")
                    print(f"- Drift Detections: {status['drift_detected_count']}")
                    print(f"- Average Confidence: {status['average_confidence']:.4f}")
                    
                    # Wait before next status check
                    time.sleep(300)  # 5 minutes
                    
            except KeyboardInterrupt:
                print("\nStopping real-time monitoring...")
                real_time_processor.stop_monitoring()
        
    except Exception as e:
        logger.error(f"Error in land cover pipeline: {str(e)}")
        raise

def parse_arguments():
    parser = argparse.ArgumentParser(description='ECOVISION: Deforestation Detection Pipeline')
    parser.add_argument('--skip-earth-engine', action='store_true',
                      help='Skip Earth Engine data collection and processing')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs for model training')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for model training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Learning rate for model training')
    parser.add_argument('--validation-split', type=float, default=0.2,
                      help='Validation split ratio')
    return parser.parse_args()

def run_deforestation_pipeline(args):
    """
    Run the deforestation detection pipeline.
    
    Args:
        args: Command line arguments
    """
    # Set up logging
    loggers = setup_logging()
    logger = loggers['satellite']  # Use satellite logger for deforestation detection
    
    try:
        logger.info("Starting ECOVISION pipeline...")
        
        # Set up directories
        dirs = setup_directories(os.getcwd())
        
        # Initialize components
        detector = DeforestationDetector()
        trainer = ModelTrainer(
            dataset_dir=dirs['dataset'],
            model_dir=dirs['models']
        )
        evaluator = ModelEvaluator()
        processor = RealTimeProcessor()
        
        # Earth Engine Processing
        if not args.skip_earth_engine:
            logger.info("Starting Earth Engine processing...")
            detector.main()  # Now correctly calling the instance method
        else:
            logger.info("Skipping Earth Engine processing...")
        
        # Model Training
        logger.info("Starting model training...")
        trainer.train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_split=args.validation_split
        )
        
        # Model Evaluation
        logger.info("Starting model evaluation...")
        evaluator.evaluate_model()
        
        # Real-time Processing Setup
        logger.info("Setting up real-time processing...")
        processor.setup_processing()
        
        logger.info("ECOVISION pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    args = parse_arguments()
    run_deforestation_pipeline(args)