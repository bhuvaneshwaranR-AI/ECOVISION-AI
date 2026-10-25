"""
Advanced visualization tools for land cover classification analysis.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import feature
import cv2
from typing import List, Tuple, Dict, Optional, Union
import seaborn as sns

class VisualizationTools:
    def __init__(self, model: tf.keras.Model, class_names: List[str], class_colors: List[str]):
        """
        Initialize visualization tools.
        
        Args:
            model: Trained TensorFlow model
            class_names: List of class names
            class_colors: List of colors for each class
        """
        self.model = model
        self.class_names = class_names
        self.class_colors = class_colors
        self.grad_model = self._create_grad_model()
    
    def _create_grad_model(self) -> tf.keras.Model:
        """Create a model that returns both predictions and gradients."""
        last_conv_layer = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
        
        grad_model = tf.keras.Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer('class_probabilities').output,
                last_conv_layer.output
            ]
        )
        return grad_model
    
    def generate_edge_map(self, image: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """
        Generate edge map using Canny edge detection.
        
        Args:
            image: Input image
            threshold: Edge detection threshold
            
        Returns:
            Edge map
        """
        # Convert to grayscale if needed
        if image.shape[-1] > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.squeeze()
        
        # Apply Canny edge detection
        edges = feature.canny(
            gray,
            sigma=1,
            low_threshold=threshold,
            high_threshold=threshold * 2
        )
        
        return edges.astype(np.float32)
    
    def generate_uncertainty_map(self, predictions: np.ndarray) -> np.ndarray:
        """
        Generate uncertainty map from model predictions.
        
        Args:
            predictions: Model predictions (class probabilities)
            
        Returns:
            Uncertainty map
        """
        # Calculate entropy for each pixel
        epsilon = 1e-7
        entropy = -np.sum(predictions * np.log(predictions + epsilon), axis=-1)
        
        # Normalize to [0, 1]
        entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + epsilon)
        
        return entropy
    
    def generate_class_activation_map(self, image: np.ndarray, class_idx: int) -> np.ndarray:
        """
        Generate class activation map for a specific class.
        
        Args:
            image: Input image
            class_idx: Index of the target class
            
        Returns:
            Class activation map
        """
        # Get gradients and predictions
        with tf.GradientTape() as tape:
            probs, conv_outputs = self.grad_model(image[np.newaxis])
            class_output = probs[:, :, :, class_idx]
        
        # Calculate gradients
        grads = tape.gradient(class_output, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps with gradients
        conv_outputs = conv_outputs[0]
        cam = tf.zeros(conv_outputs.shape[:-1])
        
        for i, w in enumerate(pooled_grads):
            cam += w * conv_outputs[:, :, i]
        
        # Post-process CAM
        cam = cam.numpy()
        cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
        
        return cam
    
    def generate_prediction_difference(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction difference analysis between two time points.
        
        Args:
            image1: First time point image
            image2: Second time point image
            
        Returns:
            Tuple of (difference map, change magnitude)
        """
        # Get predictions for both images
        pred1 = self.model.predict(image1[np.newaxis])
        pred2 = self.model.predict(image2[np.newaxis])
        
        # Calculate class differences
        diff = np.abs(pred1[0, :, :, :-1] - pred2[0, :, :, :-1])
        
        # Calculate overall change magnitude
        magnitude = np.mean(diff, axis=-1)
        
        return diff, magnitude
    
    def plot_advanced_visualization(self, image: np.ndarray, prediction: np.ndarray,
                                 output_path: str, dpi: int = 300) -> None:
        """
        Create and save comprehensive visualization plot.
        
        Args:
            image: Input image
            prediction: Model prediction
            output_path: Path to save the visualization
            dpi: DPI for saved image
        """
        plt.figure(figsize=(20, 10))
        
        # Original image
        plt.subplot(231)
        plt.imshow(image[:, :, :3])  # Show RGB channels
        plt.title('Original Image')
        plt.axis('off')
        
        # Predicted classes
        plt.subplot(232)
        pred_mask = np.argmax(prediction[..., :-1], axis=-1)
        plt.imshow(pred_mask, cmap='tab20')
        plt.title('Predicted Classes')
        plt.colorbar(ticks=range(len(self.class_names)))
        plt.axis('off')
        
        # Edge detection
        plt.subplot(233)
        edges = self.generate_edge_map(image)
        plt.imshow(edges, cmap='gray')
        plt.title('Edge Detection')
        plt.axis('off')
        
        # Uncertainty map
        plt.subplot(234)
        uncertainty = self.generate_uncertainty_map(prediction[..., :-1])
        plt.imshow(uncertainty, cmap='RdYlBu_r')
        plt.title('Prediction Uncertainty')
        plt.colorbar()
        plt.axis('off')
        
        # Class activation map for dominant class
        plt.subplot(235)
        dominant_class = np.argmax(np.bincount(pred_mask.flatten()))
        cam = self.generate_class_activation_map(image, dominant_class)
        plt.imshow(image[:, :, :3])
        plt.imshow(cam, cmap='jet', alpha=0.5)
        plt.title(f'Class Activation Map\n({self.class_names[dominant_class]})')
        plt.axis('off')
        
        # Confidence scores
        plt.subplot(236)
        confidence = prediction[..., -1]
        plt.imshow(confidence, cmap='viridis')
        plt.title('Confidence Scores')
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def plot_temporal_changes(self, image1: np.ndarray, image2: np.ndarray,
                           output_path: str, dpi: int = 300) -> None:
        """
        Create and save temporal change visualization.
        
        Args:
            image1: First time point image
            image2: Second time point image
            output_path: Path to save the visualization
            dpi: DPI for saved image
        """
        # Get predictions and differences
        diff, magnitude = self.generate_prediction_difference(image1, image2)
        
        plt.figure(figsize=(20, 10))
        
        # First time point
        plt.subplot(231)
        plt.imshow(image1[:, :, :3])
        plt.title('Time Point 1')
        plt.axis('off')
        
        # Second time point
        plt.subplot(232)
        plt.imshow(image2[:, :, :3])
        plt.title('Time Point 2')
        plt.axis('off')
        
        # Change magnitude
        plt.subplot(233)
        plt.imshow(magnitude, cmap='RdYlBu_r')
        plt.title('Change Magnitude')
        plt.colorbar()
        plt.axis('off')
        
        # Class-specific changes
        for i, (class_name, color) in enumerate(zip(self.class_names[:3], self.class_colors[:3])):
            plt.subplot(234 + i)
            plt.imshow(diff[:, :, i], cmap='RdYlBu_r')
            plt.title(f'Changes in {class_name}')
            plt.colorbar()
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def create_analysis_report(self, images: List[np.ndarray], predictions: List[np.ndarray],
                            output_dir: str) -> None:
        """
        Create comprehensive analysis report with all visualizations.
        
        Args:
            images: List of input images
            predictions: List of model predictions
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations for each image
        for i, (image, pred) in enumerate(zip(images, predictions)):
            # Basic visualization
            vis_path = os.path.join(output_dir, f'visualization_{i+1}.png')
            self.plot_advanced_visualization(image, pred, vis_path)
            
            # Temporal changes (if more than one image)
            if i < len(images) - 1:
                temporal_path = os.path.join(output_dir, f'temporal_changes_{i+1}.png')
                self.plot_temporal_changes(image, images[i+1], temporal_path)
        
        # Create summary plots
        self._create_summary_plots(predictions, output_dir)
    
    def _create_summary_plots(self, predictions: List[np.ndarray], output_dir: str) -> None:
        """Create summary plots for the entire dataset."""
        # Class distribution
        plt.figure(figsize=(10, 6))
        class_counts = []
        for pred in predictions:
            pred_mask = np.argmax(pred[..., :-1], axis=-1)
            counts = np.bincount(pred_mask.flatten(), minlength=len(self.class_names))
            class_counts.append(counts)
        
        class_counts = np.array(class_counts).mean(axis=0)
        plt.bar(self.class_names, class_counts)
        plt.title('Average Class Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
        plt.close()
        
        # Confidence distribution
        plt.figure(figsize=(10, 6))
        confidences = [pred[..., -1].flatten() for pred in predictions]
        plt.hist(confidences, bins=50, alpha=0.5)
        plt.title('Confidence Score Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
        plt.close()
        
        # Uncertainty heatmap
        plt.figure(figsize=(10, 6))
        uncertainties = [self.generate_uncertainty_map(pred[..., :-1]) for pred in predictions]
        mean_uncertainty = np.mean(uncertainties, axis=0)
        sns.heatmap(mean_uncertainty, cmap='RdYlBu_r')
        plt.title('Average Uncertainty Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'uncertainty_heatmap.png'))
        plt.close()