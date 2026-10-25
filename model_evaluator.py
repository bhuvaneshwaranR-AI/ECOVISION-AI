import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import json
import datetime
from unet_model import UNetModel

class ModelEvaluator:
    def __init__(self, model_path, dataset_dir, output_dir):
        """
        Initialize the model evaluator.
        
        Args:
            model_path: Path to the trained model
            dataset_dir: Directory containing the test dataset
            output_dir: Directory to save evaluation results
        """
        self.model_path = model_path
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model
        self.model = UNetModel.load_model(model_path)
        
        # Load test data
        self.X_test = np.load(os.path.join(dataset_dir, 'test', 'X_test.npy'))
        self.y_test = np.load(os.path.join(dataset_dir, 'test', 'y_test.npy'))
    
    def evaluate_model(self):
        """Evaluate model performance on test set."""
        print("Evaluating model performance...")
        
        # Get predictions
        y_pred = self.model.model.predict(self.X_test)
        y_pred_binary = (y_pred > 0.5).astype(np.float32)
        
        # Calculate metrics
        test_results = self.model.model.evaluate(self.X_test, self.y_test, verbose=1)
        test_metrics = dict(zip(self.model.model.metrics_names, test_results))
        
        # Calculate confusion matrix
        y_true_flat = self.y_test.reshape(-1)
        y_pred_flat = y_pred_binary.reshape(-1)
        cm = confusion_matrix(y_true_flat, y_pred_flat)
        
        # Generate classification report
        report = classification_report(y_true_flat, y_pred_flat, 
                                    target_names=['Background', 'Deforestation'],
                                    output_dict=True)
        
        # Save results
        results = {
            'test_metrics': test_metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'evaluation_timestamp': datetime.datetime.now().isoformat()
        }
        
        results_path = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nEvaluation results saved to: {results_path}")
        return results
    
    def plot_confusion_matrix(self, results):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(8, 6))
        cm = np.array(results['confusion_matrix'])
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create heatmap
        sns.heatmap(cm_percent, annot=True, fmt='.1f',
                   xticklabels=['Background', 'Deforestation'],
                   yticklabels=['Background', 'Deforestation'],
                   cmap='Blues')
        
        plt.title('Confusion Matrix (%)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix plot saved to: {plot_path}")
    
    def visualize_predictions(self, num_samples=5):
        """Visualize and save model predictions on test samples."""
        # Randomly select samples
        indices = np.random.choice(len(self.X_test), num_samples, replace=False)
        
        for idx, sample_idx in enumerate(indices):
            # Get sample data
            X_sample = self.X_test[sample_idx]
            y_true = self.y_test[sample_idx]
            y_pred = self.model.model.predict(X_sample[np.newaxis, ...])[0]
            y_pred_binary = (y_pred > 0.5).astype(np.float32)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Plot current period image (using first 3 channels)
            axes[0, 0].imshow(X_sample[..., :3])
            axes[0, 0].set_title('Current Period')
            axes[0, 0].axis('off')
            
            # Plot previous period image (using last 3 channels)
            axes[0, 1].imshow(X_sample[..., 3:])
            axes[0, 1].set_title('Previous Period')
            axes[0, 1].axis('off')
            
            # Plot true mask
            axes[0, 2].imshow(y_true, cmap='RdYlGn_r')
            axes[0, 2].set_title('True Mask')
            axes[0, 2].axis('off')
            
            # Plot predicted probability mask
            axes[1, 0].imshow(y_pred, cmap='RdYlGn_r')
            axes[1, 0].set_title('Predicted Probability')
            axes[1, 0].axis('off')
            
            # Plot binary prediction
            axes[1, 1].imshow(y_pred_binary, cmap='RdYlGn_r')
            axes[1, 1].set_title('Binary Prediction')
            axes[1, 1].axis('off')
            
            # Plot difference map
            difference = np.abs(y_true - y_pred_binary)
            axes[1, 2].imshow(difference, cmap='RdYlGn_r')
            axes[1, 2].set_title('Prediction Error')
            axes[1, 2].axis('off')
            
            # Add colorbar
            fig.colorbar(axes[0, 2].imshow(y_true, cmap='RdYlGn_r'),
                        ax=axes.ravel().tolist(),
                        label='Deforestation Probability')
            
            # Save plot
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, f'prediction_sample_{idx+1}.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            
            print(f"Prediction visualization {idx+1} saved to: {plot_path}")
    
    def generate_evaluation_report(self):
        """Generate a comprehensive evaluation report."""
        # Evaluate model
        results = self.evaluate_model()
        
        # Plot confusion matrix
        self.plot_confusion_matrix(results)
        
        # Visualize predictions
        self.visualize_predictions()
        
        # Generate HTML report
        report_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Deforestation Detection Model Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric { margin: 10px 0; }
                .section { margin: 20px 0; }
                img { max-width: 100%; margin: 10px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Deforestation Detection Model Evaluation Report</h1>
            <div class="section">
                <h2>Model Performance Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
        """
        
        # Add metrics
        for metric, value in results['test_metrics'].items():
            report_template += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>"
        
        report_template += """
                </table>
            </div>
            
            <div class="section">
                <h2>Classification Report</h2>
                <table>
                    <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Support</th>
                    </tr>
        """
        
        # Add classification report
        for class_name, metrics in results['classification_report'].items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                report_template += f"""
                    <tr>
                        <td>{class_name}</td>
                        <td>{metrics['precision']:.4f}</td>
                        <td>{metrics['recall']:.4f}</td>
                        <td>{metrics['f1-score']:.4f}</td>
                        <td>{metrics['support']}</td>
                    </tr>
                """
        
        report_template += """
                </table>
            </div>
            
            <div class="section">
                <h2>Confusion Matrix</h2>
                <img src="confusion_matrix.png" alt="Confusion Matrix">
            </div>
            
            <div class="section">
                <h2>Sample Predictions</h2>
        """
        
        # Add sample predictions
        for i in range(5):
            report_template += f'<img src="prediction_sample_{i+1}.png" alt="Prediction Sample {i+1}">'
        
        report_template += """
            </div>
        </body>
        </html>
        """
        
        # Save report
        report_path = os.path.join(self.output_dir, 'evaluation_report.html')
        with open(report_path, 'w') as f:
            f.write(report_template)
        
        print(f"\nEvaluation report saved to: {report_path}")

def main():
    """Example usage of the ModelEvaluator class."""
    try:
        # Set up directories
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'models', 'final_model.h5')
        dataset_dir = os.path.join(base_dir, 'dataset')
        output_dir = os.path.join(base_dir, 'evaluation_results')
        
        # Initialize evaluator
        evaluator = ModelEvaluator(model_path, dataset_dir, output_dir)
        
        # Generate evaluation report
        evaluator.generate_evaluation_report()
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()