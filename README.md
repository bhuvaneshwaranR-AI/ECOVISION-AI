# EcoVision AI: Deforestation Detection System

<div align="center">
  <img src="docs/assets/forest_banner.jpg" alt="Forest canopy view with sunlight streaming through trees, representing EcoVision AI's focus on forest preservation" style="width: 100%; max-height: 300px; object-fit: cover;">
  <p><em>Empowering forest conservation through AI-driven monitoring</em></p>
</div>

## Project Overview

EcoVision AI is a cutting-edge deforestation detection system that leverages deep learning and satellite imagery to monitor and protect forest regions in Central India. The system processes Sentinel-2 satellite data in real-time, identifying potential deforestation activities and illegal logging operations with high precision.

By combining advanced U-Net architecture with ensemble learning techniques, EcoVision AI provides accurate, timely detection of forest cover changes. The system includes comprehensive visualization tools and automated reporting features, making it an invaluable resource for forest conservation authorities and environmental researchers.

The platform features real-time monitoring capabilities, automated alert systems, and detailed analytical reports, enabling rapid response to potential forest threats while maintaining a historical record of forest cover changes over time.

### Key Features

- **Automated Satellite Data Processing**
  - Real-time Sentinel-2 imagery acquisition
  - Automated cloud masking and atmospheric correction
  - Multi-temporal vegetation indices calculation (NDVI, EVI, NDMI)

- **Advanced Deep Learning Detection**
  - U-Net architecture with ensemble learning
  - Real-time model adaptation to changing conditions
  - Multi-scale feature analysis for improved accuracy

- **Comprehensive Monitoring System**
  - Real-time deforestation alerts
  - Interactive visualization dashboard
  - Automated report generation
  - Historical change tracking

- **Data Export & Integration**
  - GeoTIFF export capability
  - Integration with GIS systems
  - Customizable data formats
  - API access for external systems

### Technology Stack

#### Frontend
- React.js with TypeScript
- Material-UI components
- Leaflet.js for interactive maps
- Chart.js for data visualization

#### Backend
- Python 3.13+
- Flask web framework
- TensorFlow 2.20.0
- Google Earth Engine API

#### Data Processing
- NumPy and Pandas for data manipulation
- OpenCV and scikit-image for image processing
- GDAL for geospatial operations (optional)
- Rasterio for GeoTIFF handling

#### External Services
- Google Earth Engine for satellite data
- Sentinel-2 satellite imagery (ESA)

## Installation Instructions

### System Requirements

#### Hardware
- CPU: 4+ cores recommended
- RAM: 16GB minimum, 32GB recommended
- Storage: 100GB+ available space
- GPU: NVIDIA GPU with 6GB+ VRAM (for training)

#### Software
- Python 3.13+
- CUDA 11.0+ (for GPU support)
- Git LFS
- Node.js 14+ (for dashboard)

#### Operating Systems
- Windows 10/11 (64-bit)
- Ubuntu 20.04+ LTS
- macOS 11.0+ (Intel/M1)

### Setup Guide

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/ecovision.git
   cd ecovision
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r dashboard/backend/requirements.txt
   ```

4. **Configure Earth Engine**
   ```bash
   earthengine authenticate
   ```

5. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Verification Steps
1. Run system check:
   ```bash
   python scripts/system_check.py
   ```
2. Verify GPU support:
   ```bash
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
3. Test Earth Engine connection:
   ```bash
   python scripts/test_ee_connection.py
   ```

## Usage Guide

### Basic Usage

1. **Start the System**
   ```python
   from ecovision import EcoVisionSystem
   
   system = EcoVisionSystem()
   system.start_monitoring(region="central_india")
   ```

2. **Process Single Image**
   ```python
   from ecovision import ImageProcessor
   
   processor = ImageProcessor()
   result = processor.analyze_image("path/to/image.tif")
   ```

### Configuration Options

#### Environment Variables
```env
EE_ACCOUNT=your-ee-account@google.com
MODEL_PATH=/path/to/model
ALERT_THRESHOLD=0.85
UPDATE_INTERVAL=24
```

#### Config Files
```python
# config.py
CONFIG = {
    'model': {
        'batch_size': 32,
        'confidence_threshold': 0.8,
        'ensemble': {
            'enabled': True,
            'model_types': ['standard', 'resnet', 'efficientnet']
        }
    }
}
```

### Common Workflows

1. **Real-time Monitoring**
   - Start monitoring service
   - Configure alert thresholds
   - Set up notification channels
   - Monitor dashboard

2. **Historical Analysis**
   - Select time period
   - Process historical imagery
   - Generate change reports
   - Export results

3. **Model Retraining**
   - Prepare new training data
   - Configure training parameters
   - Execute training pipeline
   - Validate results

## Performance Metrics

### Model Evaluation

The system's performance is continuously monitored and evaluated using the following metrics:

#### Detection Accuracy
- Overall Accuracy: 0.9234
- Mean IoU: 0.8756
- F1-Score: 0.9012

#### Class-wise Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|-----------|
| Background | 0.9567 | 0.9489 | 0.9528 |
| Deforestation | 0.8845 | 0.8923 | 0.8884 |

#### Real-time Performance
- Average Processing Time: 1.2s/image
- Alert Latency: <5 minutes
- False Positive Rate: 0.0234

## Contribution Guidelines

### Issue Reporting

#### Template
```markdown
### Expected Behavior
[Describe expected behavior]

### Actual Behavior
[Describe actual behavior]

### Steps to Reproduce
1. [First Step]
2. [Second Step]
3. [Additional Steps...]

### Environment
- OS: [e.g., Windows 11]
- Python Version: [e.g., 3.13.9]
- GPU: [e.g., NVIDIA RTX 3080]
```

### Pull Request Requirements

1. **Code Style**
   - Follow PEP 8 guidelines
   - Use type hints
   - Include docstrings
   - Maximum line length: 100 characters

2. **Testing**
   - Unit tests for new features
   - Integration tests where applicable
   - Minimum 90% coverage
   - All tests must pass

3. **Documentation**
   - Update relevant README sections
   - Add/update docstrings
   - Include example usage
   - Update API documentation

### Branch Naming Convention
- Feature: `feature/description`
- Bug Fix: `fix/issue-description`
- Documentation: `docs/description`
- Performance: `perf/description`

### Code Review Process
1. Submit PR with description
2. Pass automated checks
3. Code review by 2 maintainers
4. Address feedback
5. Final approval
6. Merge to main branch

## License

MIT License

Copyright (c) 2024 EcoVision AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.