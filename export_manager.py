"""
Export manager for land cover classification results and visualizations.
"""

import os
import json
import shutil
import datetime
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import seaborn as sns
from PIL import Image

class ExportManager:
    def __init__(self, output_dir: str):
        """
        Initialize export manager.
        
        Args:
            output_dir: Directory to store exported files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        self.dirs = {
            'images': os.path.join(output_dir, 'images'),
            'data': os.path.join(output_dir, 'data'),
            'reports': os.path.join(output_dir, 'reports'),
            'visualizations': os.path.join(output_dir, 'visualizations')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def export_predictions(self, predictions: np.ndarray, confidence: np.ndarray,
                         metadata: Dict[str, Any], region_id: str) -> str:
        """
        Export prediction results to various formats.
        
        Args:
            predictions: Model predictions (class probabilities)
            confidence: Confidence scores
            metadata: Prediction metadata
            region_id: Identifier for the region
            
        Returns:
            Path to exported data directory
        """
        # Create export directory for this prediction
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        export_id = f"{region_id}_{timestamp}"
        export_dir = os.path.join(self.dirs['data'], export_id)
        os.makedirs(export_dir, exist_ok=True)
        
        # Save predictions as numpy arrays
        np.save(os.path.join(export_dir, 'predictions.npy'), predictions)
        np.save(os.path.join(export_dir, 'confidence.npy'), confidence)
        
        # Save metadata
        metadata.update({
            'export_id': export_id,
            'timestamp': timestamp,
            'region_id': region_id
        })
        with open(os.path.join(export_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return export_dir
    
    def export_visualizations(self, vis_dir: str, export_dir: str) -> None:
        """
        Export visualization images.
        
        Args:
            vis_dir: Directory containing visualization images
            export_dir: Directory to export to
        """
        # Copy visualization images
        vis_export_dir = os.path.join(export_dir, 'visualizations')
        os.makedirs(vis_export_dir, exist_ok=True)
        
        for filename in os.listdir(vis_dir):
            if filename.endswith(('.png', '.jpg')):
                shutil.copy2(
                    os.path.join(vis_dir, filename),
                    os.path.join(vis_export_dir, filename)
                )
    
    def generate_report(self, export_dir: str, class_names: List[str],
                       include_visualizations: bool = True) -> str:
        """
        Generate PDF report of analysis results.
        
        Args:
            export_dir: Directory containing exported data
            class_names: Names of land cover classes
            include_visualizations: Whether to include visualization images
            
        Returns:
            Path to generated report
        """
        # Load data
        predictions = np.load(os.path.join(export_dir, 'predictions.npy'))
        confidence = np.load(os.path.join(export_dir, 'confidence.npy'))
        with open(os.path.join(export_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Land Cover Classification Report', ln=True, align='C')
        pdf.ln(10)
        
        # Metadata
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Analysis Information', ln=True)
        pdf.set_font('Arial', '', 10)
        for key, value in metadata.items():
            if key not in ['export_id', 'predictions', 'confidence']:
                pdf.cell(0, 8, f"{key}: {value}", ln=True)
        pdf.ln(5)
        
        # Class Distribution
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Land Cover Distribution', ln=True)
        
        # Create and save distribution plot
        plt.figure(figsize=(10, 6))
        class_counts = np.sum(predictions > 0.5, axis=(0, 1))
        plt.bar(class_names, class_counts)
        plt.xticks(rotation=45)
        plt.title('Land Cover Class Distribution')
        plt.tight_layout()
        dist_plot_path = os.path.join(export_dir, 'class_distribution.png')
        plt.savefig(dist_plot_path)
        plt.close()
        
        # Add distribution plot to PDF
        pdf.image(dist_plot_path, x=10, w=190)
        pdf.ln(5)
        
        # Confidence Analysis
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Confidence Analysis', ln=True)
        
        # Create and save confidence histogram
        plt.figure(figsize=(10, 6))
        plt.hist(confidence.flatten(), bins=50)
        plt.title('Confidence Score Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.tight_layout()
        conf_plot_path = os.path.join(export_dir, 'confidence_distribution.png')
        plt.savefig(conf_plot_path)
        plt.close()
        
        # Add confidence plot to PDF
        pdf.image(conf_plot_path, x=10, w=190)
        pdf.ln(5)
        
        # Include visualizations if available
        if include_visualizations:
            vis_dir = os.path.join(export_dir, 'visualizations')
            if os.path.exists(vis_dir):
                pdf.add_page()
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, 'Advanced Visualizations', ln=True)
                
                for filename in sorted(os.listdir(vis_dir)):
                    if filename.endswith(('.png', '.jpg')):
                        pdf.add_page()
                        pdf.set_font('Arial', 'B', 10)
                        title = ' '.join(filename.split('_')).replace('.png', '').replace('.jpg', '')
                        pdf.cell(0, 10, title, ln=True)
                        pdf.image(os.path.join(vis_dir, filename), x=10, w=190)
        
        # Save report
        report_path = os.path.join(self.dirs['reports'], f"report_{metadata['export_id']}.pdf")
        pdf.output(report_path)
        
        return report_path
    
    def export_to_geotiff(self, predictions: np.ndarray, confidence: np.ndarray,
                         metadata: Dict[str, Any], output_path: str) -> None:
        """
        Export predictions to GeoTIFF format.
        
        Args:
            predictions: Model predictions
            confidence: Confidence scores
            metadata: Prediction metadata including geospatial information
            output_path: Path to save the GeoTIFF file
        """
        try:
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError:
            print("Warning: GDAL/rasterio not available. GeoTIFF export functionality will be disabled.")
            return
            
        try:
            # Get geospatial information from metadata
            bounds = metadata.get('bounds', None)
            if not bounds:
                raise ValueError("Geospatial bounds not found in metadata")
            
            # Create transform
            transform = from_bounds(
                bounds['west'], bounds['south'],
                bounds['east'], bounds['north'],
                predictions.shape[1], predictions.shape[0]
            )
            
            # Stack predictions and confidence
            data = np.concatenate([predictions, confidence[..., np.newaxis]], axis=-1)
            
            # Write GeoTIFF
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=data.shape[2],
                dtype=data.dtype,
                crs='+proj=longlat +datum=WGS84 +no_defs',
                transform=transform
            ) as dst:
                for i in range(data.shape[2]):
                    dst.write(data[..., i], i + 1)
                
                # Add metadata
                dst.update_tags(**{
                    k: str(v) for k, v in metadata.items()
                    if isinstance(v, (str, int, float, bool))
                })
                
        except Exception as e:
            print(f"Error exporting to GeoTIFF: {str(e)}")
            raise
    
    def create_time_series_animation(self, predictions_list: List[np.ndarray],
                                   timestamps: List[str], output_path: str) -> None:
        """
        Create animated GIF of time series predictions.
        
        Args:
            predictions_list: List of prediction arrays
            timestamps: List of timestamp strings
            output_path: Path to save the animation
        """
        # Create frames
        frames = []
        for pred, timestamp in zip(predictions_list, timestamps):
            # Create figure
            plt.figure(figsize=(10, 8))
            plt.imshow(np.argmax(pred, axis=-1))
            plt.title(f'Land Cover Classification - {timestamp}')
            plt.colorbar(label='Class')
            plt.axis('off')
            
            # Convert to image
            plt.savefig('temp.png', bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # Open and append frame
            frames.append(Image.open('temp.png'))
        
        # Save animation
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000,  # 1 second per frame
            loop=0
        )
        
        # Clean up
        os.remove('temp.png')
        
    def export_statistics(self, predictions: np.ndarray, confidence: np.ndarray,
                         class_names: List[str], output_path: str) -> None:
        """
        Export detailed statistics to Excel format.
        
        Args:
            predictions: Model predictions
            confidence: Confidence scores
            class_names: Names of land cover classes
            output_path: Path to save the Excel file
        """
        # Calculate statistics
        stats = {
            'Class': class_names,
            'Area (pixels)': np.sum(predictions > 0.5, axis=(0, 1)),
            'Area (%)': np.sum(predictions > 0.5, axis=(0, 1)) / np.prod(predictions.shape[:2]) * 100,
            'Mean Confidence': [np.mean(confidence[predictions[..., i] > 0.5]) for i in range(len(class_names))],
            'Min Confidence': [np.min(confidence[predictions[..., i] > 0.5]) for i in range(len(class_names))],
            'Max Confidence': [np.max(confidence[predictions[..., i] > 0.5]) for i in range(len(class_names))]
        }
        
        # Create DataFrame and export
        df = pd.DataFrame(stats)
        df.to_excel(output_path, index=False)