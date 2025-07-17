# Random Forest Classification for Land Cover Analysis

A comprehensive machine learning project for land cover classification using Random Forest algorithms with advanced feature engineering, including RGB statistics, Wavelet Scattering Transform (WST), and hybrid approaches.

## 🎯 Project Overview

This project focuses on classifying land cover types from aerial imagery using multiple feature extraction techniques:

- **RGB Statistical Features**: Mean and standard deviation of color channels
- **Wavelet Scattering Transform (WST)**: Advanced texture and pattern analysis
- **Hybrid Approaches**: Combination of RGB and WST features
- **Noise Robustness**: Evaluation across different noise conditions

### Land Cover Classes
- **Assatigue**: Agricultural/farmland areas
- **Popolar**: Forest/woodland areas  
- **Sunset**: Urban/built-up areas

## 🏗️ Project Structure

```
random_forest/
├── 📁 experiments_organized/     # Organized experiment results
│   ├── rgb_clean_kbest/         # Clean dataset experiments
│   ├── rgb_gaussian30_kbest/    # Gaussian noise experiments
│   ├── rgb_gaussian50_kbest/    # Higher Gaussian noise
│   ├── rgb_poisson60_kbest/     # Poisson noise experiments
│   └── rgb_salt_pepper25_kbest/ # Salt & pepper noise
├── 📁 inference_k5_popolar_results/ # Inference results
├── 📁 scripts/                  # Python scripts and utilities
│   ├── 🐍 train_and_save_model.py   # Model training script
│   ├── 🐍 inference_on_dataset.py   # Inference script
│   ├── 🐍 add_noise.py              # Noise generation utilities
│   ├── 🐍 analyze_experiments.py    # Results analysis
│   ├── 🐍 create_visualizations.py  # Visualization generation
│   ├── 🐍 fixed_analysis.py         # Fixed analysis pipeline
│   ├── 🐍 simple_analysis.py        # Simplified analysis
│   └── 🔧 generate_noise_datasets.sh # Dataset generation script
├── 📓 Poplar.ipynb              # Main analysis notebook
└── 📋 requirements.txt          # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster WST computation)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/mattiabestiaccia/random_forest.git
cd random_forest
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Train a Model
```bash
python scripts/train_and_save_model.py
```
This script will:
- Load the dataset
- Extract features (RGB + WST)
- Train a Random Forest classifier
- Save the trained model

#### 2. Run Inference
```bash
python scripts/inference_on_dataset.py
```
Performs classification on new datasets and generates:
- Confusion matrices
- Classification reports
- Performance metrics

#### 3. Analyze Results
```bash
python scripts/analyze_experiments.py
```
Comprehensive analysis of experimental results across different conditions.

#### 4. Generate Visualizations
```bash
python scripts/create_visualizations.py
```
Creates plots and charts for result interpretation.

## 📊 Features

### Feature Extraction Methods

1. **RGB Statistical Features**
   - Mean and standard deviation per color channel
   - Simple but effective for color-based classification

2. **Wavelet Scattering Transform (WST)**
   - Multi-scale texture analysis
   - Translation-invariant features
   - Robust to deformations

3. **Hybrid Features**
   - Combination of RGB and WST
   - Leverages both color and texture information
   - Best performance across most scenarios

### Experimental Design

- **K-Best Feature Selection**: Tests with k=2, 5, 10, 20 features
- **Dataset Sizes**: Mini, small, original, augmented
- **Noise Conditions**: Clean, Gaussian, Poisson, Salt & Pepper
- **Cross-validation**: Stratified K-fold validation

## 📈 Results

The project includes comprehensive experimental results organized by:

```
experiments_organized/
└── {noise_type}/
    └── {land_cover_class}/
        └── {dataset_size}/
            └── {k_features}/
                └── {method}/
                    └── results.json
```

### Key Findings

- **Hybrid approach** (RGB + WST) consistently outperforms individual methods
- **WST features** provide excellent noise robustness
- **Feature selection** with k=10-20 offers optimal performance
- **Original dataset size** balances accuracy and computational efficiency

## 🔬 Technical Details

### Machine Learning Pipeline

1. **Data Preprocessing**
   - Image normalization
   - Noise injection (if applicable)
   - Patch extraction and augmentation

2. **Feature Engineering**
   - RGB statistical moments
   - WST coefficient extraction
   - Feature standardization

3. **Model Training**
   - Random Forest with 100 estimators
   - Stratified train/test split (80/20)
   - Cross-validation for hyperparameter tuning

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrices
   - Feature importance analysis

### Dependencies

Key libraries used:
- `scikit-learn`: Machine learning algorithms
- `kymatio`: Wavelet Scattering Transform
- `opencv-python`: Image processing
- `numpy`, `pandas`: Data manipulation
- `matplotlib`, `seaborn`: Visualization
- `torch`: Deep learning framework (for WST)

## 📝 Experiments

The project includes extensive experiments across different conditions:

### Noise Types
- **Clean**: Original images without noise
- **Gaussian**: Various noise levels (10, 30, 50)
- **Poisson**: Poisson noise (40, 60)
- **Salt & Pepper**: Salt and pepper noise (5, 15, 25)
- **Speckle**: Speckle noise (15, 35, 55)
- **Uniform**: Uniform noise (10, 25, 40)

### Feature Selection
- **k=2**: Minimal feature set
- **k=5**: Small feature set
- **k=10**: Medium feature set
- **k=20**: Large feature set

### Dataset Sizes
- **Mini**: Small subset for rapid prototyping
- **Small**: Reduced dataset for faster training
- **Original**: Full original dataset
- **Augmented**: Enhanced with data augmentation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Kymatio**: For the excellent Wavelet Scattering Transform implementation
- **scikit-learn**: For robust machine learning tools
- **OpenCV**: For comprehensive image processing capabilities

## 📧 Contact

For questions or collaborations, please contact:
- **Author**: Mattia Bestiaccia
- **Email**: mattiabestiaccia@gmail.com
- **GitHub**: [@mattiabestiaccia](https://github.com/mattiabestiaccia)

---

*This project demonstrates advanced machine learning techniques for remote sensing applications, with emphasis on robustness and feature engineering.*