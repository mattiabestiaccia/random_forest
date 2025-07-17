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

#### 5. Run Batch Experiments
```bash
cd experiments_organized
./run_all_experiments.sh [dataset_path]
```
Executes all possible experiment combinations systematically:
- All noise conditions (clean, gaussian, poisson, salt&pepper, etc.)
- All land cover areas (assatigue, popolar, sunset)
- All feature methods (RGB stats, WST, hybrid)
- All k-best values (2, 5, 10, 20)

The script generates organized JSON reports and logs for comprehensive analysis.

## 📊 Features

### Experimental Infrastructure

The project includes a comprehensive experimental infrastructure located in `experiments_organized/`:

- **`run_all_experiments.sh`**: Automated batch script that executes all possible experiment combinations
- **`structure.md`**: Defines the hierarchical organization of experiments
- **758 JSON result files**: Organized across multiple noise conditions and parameters

The experiments were generated using the `run_all_experiments.sh` script, which systematically runs all combinations of:
- Dataset types (mini, small, original, augmented)
- Land cover areas (assatigue, popolar, sunset)
- Feature methods (RGB stats, WST, hybrid)
- K-best feature selection values (2, 5, 10, 20)

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
└── {noise_type}/                    # e.g., rgb_clean_kbest, rgb_gaussian30_kbest
    └── {land_cover_class}/          # assatigue, popolar, sunset
        └── {dataset_size}/          # mini, small, original, augmented
            └── {k_features}/        # k2, k5, k10, k20
                └── {method}/        # advanced_stats, wst, hybrid
                    └── {method}_k{k}_{size}_report.json
```

### Experiment Generation Script

The comprehensive experiment results were generated using `experiments_organized/run_all_experiments.sh`, which:

- Systematically runs all parameter combinations
- Generates detailed logs for each experiment
- Creates organized JSON reports with performance metrics
- Provides batch processing capabilities for large-scale analysis

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

The project includes extensive experiments across different conditions, with results stored in `experiments_organized/`:

### Noise Types
- **Clean**: Original images without noise (`rgb_clean_kbest/`)
- **Gaussian**: Gaussian noise level 30 and 50 (`rgb_gaussian30_kbest/`, `rgb_gaussian50_kbest/`)
- **Poisson**: Poisson noise level 60 (`rgb_poisson60_kbest/`)
- **Salt & Pepper**: Salt and pepper noise level 25 (`rgb_salt_pepper25_kbest/`)
- **Speckle**: Speckle noise level 55 (`rgb_salt_spekle55_kbest/`)
- **Uniform**: Uniform noise level 40 (`rgb_salt_uniform40_kbest/`)

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