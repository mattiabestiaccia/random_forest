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
├── 📁 analysis/     # Noise robustness analysis
│   ├── comparison/         # Camparative analysis of different scenarios
│   ├── detailed/         # Single graphs of different scenarios
│   ├── analysis_summary.md         # Executive summary with key findings from the analysis
│   ├── comprehensive_report.md         # Complete statistical analysis report
│   ├── experiments_summary.csv        # Complete export of analysis results
│   ├── qualitative_analysis.md         # In-depth qualitative interpretation of the experimental results
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
python scripts/train_and_save_model.py dataset_path area_name feature_method k_features output_dir [options]
```

**Required Arguments:**
- `dataset_path`: Path to the dataset directory
- `area_name`: Land cover area (`assatigue`, `popolar`, `sunset`)
- `feature_method`: Feature extraction method (`advanced_stats`, `wst`, `hybrid`)
- `k_features`: Number of features to select (`2`, `5`, `10`, `20`)
- `output_dir`: Output directory for model and results

**Optional Arguments:**
- `--n_estimators`: Number of trees in the forest (default: 50)
- `--test_size`: Test set size fraction (default: 0.2)
- `--random_state`: Random seed for reproducibility (default: 42)
- `--cv_folds`: Number of cross-validation folds (default: 5)

**Examples:**
```bash
# Basic usage with advanced statistics
python scripts/train_and_save_model.py /path/to/dataset popolar advanced_stats 5 ./output

# Using WST features with custom parameters
python scripts/train_and_save_model.py /path/to/dataset assatigue wst 10 ./output --n_estimators 100 --test_size 0.3

# Hybrid features with all parameters
python scripts/train_and_save_model.py /path/to/dataset sunset hybrid 20 ./output --n_estimators 200 --test_size 0.25 --random_state 123 --cv_folds 10
```

This script will:
- Load the dataset from the specified path
- Extract features using the chosen method
- Train a Random Forest classifier
- Save the trained model and all artifacts

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
./run_all_experiments.sh [base_dataset_dir]
```

**Arguments:**
- `base_dataset_dir`: Base directory containing different dataset versions (optional, defaults to `/home/brusc/Projects/random_forest/dataset_rgb`)

**What it does:**
Executes all possible experiment combinations systematically:
- **Dataset types**: mini, small, original, augmented
- **Noise conditions**: clean, gaussian30, gaussian50, poisson60, salt_pepper25, speckle55, uniform40
- **Land cover areas**: assatigue, popolar, sunset
- **Feature methods**: advanced_stats, wst, hybrid
- **K-best values**: 2, 5, 10, 20

**Example:**
```bash
# Use default dataset directory
cd experiments_organized
./run_all_experiments.sh

# Use custom dataset directory
cd experiments_organized
./run_all_experiments.sh /custom/path/to/datasets
```

The script generates:
- Organized JSON reports in structured directories
- Detailed experiment logs
- Summary reports with success/failure statistics
- CSV files with all results for analysis

**Note:** This script requires the `rgb_rf_generalized.py` script to be present in the experiments_organized directory.

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

## 📈 Results & Analysis

### Complete Analysis Suite

The project now includes a comprehensive analysis of 324 experiments with professional reports and visualizations. Use the complete analysis script:

```bash
# Activate virtual environment
source venv_rf/bin/activate

# Run complete analysis
python create_complete_analysis.py
```

This generates:
- **Statistical reports**: Comprehensive analysis with performance metrics
- **Qualitative analysis**: In-depth interpretation of robustness findings  
- **Executive summary**: Key findings and recommendations
- **Data export**: Complete CSV dataset (186KB)
- **Visualizations**: 37 high-quality plots (6.6MB)

### Key Findings from Analysis

- **🏆 Best Method**: WST achieves highest accuracy (0.913 average)
- **🔊 Noise Impact**: 11.4% performance loss from clean to Gaussian σ=50
- **📊 Dataset Effect**: 7.4% improvement from mini to original size
- **🛡️ Robustness**: WST shows lowest degradation under noise
- **⚖️ Feature Selection**: k=10-20 provides optimal performance

### Experimental Results Structure

```
experiments_organized/
└── {noise_type}/                    # e.g., rgb_clean_kbest, rgb_gaussian30_kbest
    └── {land_cover_class}/          # assatigue, popolar, sunset
        └── {dataset_size}/          # mini, small, original, augmented
            └── {k_features}/        # k2, k5, k10, k20
                └── {method}/        # advanced_stats, wst, hybrid
                    └── {method}_k{k}_{size}_report.json
```

### Analysis Outputs

```
analysis/
├── comprehensive_report.md         # Complete statistical analysis
├── qualitative_analysis.md         # In-depth qualitative interpretation
├── analysis_summary.md            # Executive summary with key findings
├── experiments_summary.csv # Complete dataset export
├── comparisons/                    # High-level comparison plots (4 plots)
└── detailed/                      # Detailed analysis plots (33+ plots)
```

### Professional Visualizations

The analysis includes publication-ready plots:
- **Comparison plots**: Accuracy vs method/noise/dataset (averaged across dimensions)
- **Detailed plots**: Individual condition analysis (geographic-only averaging)
- **Statistical plots**: Performance distributions and confidence intervals
- **Clean formatting**: Professional styling suitable for research publications

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