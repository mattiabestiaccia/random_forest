# 📊 Random Forest Analysis - Salt & Pepper Noise Visualization Gallery

This document presents a comprehensive gallery of all visualizations generated from the analysis of Salt & Pepper noise Random Forest experiments. The plots are organized by category for easy navigation and comparison.

## 📋 Table of Contents

1. [High-Level Comparisons](#high-level-comparisons)
2. [Dataset Size Analysis](#dataset-size-analysis)
3. [Feature Selection Analysis](#feature-selection-analysis)
4. [Noise Robustness Analysis](#noise-robustness-analysis)

---

## 🔍 High-Level Comparisons

### Overall Performance Summary
![Accuracy Heatmap Summary](experiments/s%26p/saltpepper_analysis/comparisons/accuracy_heatmap_saltpepper_summary.png)
*Comprehensive heatmap showing accuracy across all Salt & Pepper noise experimental conditions*

### Method Comparison
![Accuracy vs Method Boxplot](experiments/s%26p/saltpepper_analysis/comparisons/accuracy_vs_method_boxplot_saltpepper.png)
*Statistical comparison of feature extraction methods (WST, Advanced Stats, Hybrid) under Salt & Pepper noise*

### Dataset Size Impact
![Accuracy vs Dataset Size](experiments/s%26p/saltpepper_analysis/comparisons/accuracy_vs_dataset_size_overall.png)
*Overall performance improvement with dataset size under Salt & Pepper noise conditions*

### Noise Robustness Overview
![Accuracy vs Noise Level](experiments/s%26p/saltpepper_analysis/comparisons/accuracy_vs_saltpepper_noise_overall.png)
*Performance degradation under different Salt & Pepper noise levels (5%, 15%, 25%)*

---

## 📈 Dataset Size Analysis

### Clean Dataset Conditions

#### K=2 Features
![Clean K2](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_dataset_clean_k2.png)
*Performance vs dataset size with minimal feature selection (k=2)*

#### K=5 Features
![Clean K5](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_dataset_clean_k5.png)
*Performance vs dataset size with small feature selection (k=5)*

#### K=10 Features
![Clean K10](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_dataset_clean_k10.png)
*Performance vs dataset size with medium feature selection (k=10)*

#### K=20 Features
![Clean K20](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_dataset_clean_k20.png)
*Performance vs dataset size with large feature selection (k=20)*

### Salt & Pepper Noise Conditions

#### K=2 Features
![SaltPepper5 K2](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_dataset_saltpepper5_k2.png)
*Performance vs dataset size under Salt & Pepper noise (5%) with k=2 features*

![SaltPepper15 K2](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_dataset_saltpepper15_k2.png)
*Performance vs dataset size under Salt & Pepper noise (15%) with k=2 features*

![SaltPepper25 K2](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_dataset_saltpepper25_k2.png)
*Performance vs dataset size under Salt & Pepper noise (25%) with k=2 features*

#### K=5 Features
![SaltPepper5 K5](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_dataset_saltpepper5_k5.png)
*Performance vs dataset size under Salt & Pepper noise (5%) with k=5 features*

![SaltPepper15 K5](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_dataset_saltpepper15_k5.png)
*Performance vs dataset size under Salt & Pepper noise (15%) with k=5 features*

![SaltPepper25 K5](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_dataset_saltpepper25_k5.png)
*Performance vs dataset size under Salt & Pepper noise (25%) with k=5 features*

#### K=10 Features
![SaltPepper5 K10](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_dataset_saltpepper5_k10.png)
*Performance vs dataset size under Salt & Pepper noise (5%) with k=10 features*

![SaltPepper15 K10](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_dataset_saltpepper15_k10.png)
*Performance vs dataset size under Salt & Pepper noise (15%) with k=10 features*

![SaltPepper25 K10](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_dataset_saltpepper25_k10.png)
*Performance vs dataset size under Salt & Pepper noise (25%) with k=10 features*

#### K=20 Features
![SaltPepper5 K20](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_dataset_saltpepper5_k20.png)
*Performance vs dataset size under Salt & Pepper noise (5%) with k=20 features*

![SaltPepper15 K20](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_dataset_saltpepper15_k20.png)
*Performance vs dataset size under Salt & Pepper noise (15%) with k=20 features*

![SaltPepper25 K20](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_dataset_saltpepper25_k20.png)
*Performance vs dataset size under Salt & Pepper noise (25%) with k=20 features*

---

## 🎯 Feature Selection Analysis

### Clean Dataset Conditions

#### Mini Dataset
![Clean Mini](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_k_clean_mini.png)
*Feature selection impact on mini dataset (clean conditions)*

#### Small Dataset
![Clean Small](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_k_clean_small.png)
*Feature selection impact on small dataset (clean conditions)*

#### Original Dataset
![Clean Original](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_k_clean_original.png)
*Feature selection impact on original dataset (clean conditions)*

### Salt & Pepper Noise Conditions

#### Mini Dataset
![SaltPepper5 Mini](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_k_saltpepper5_mini.png)
*Feature selection impact on mini dataset under Salt & Pepper noise (5%)*

![SaltPepper15 Mini](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_k_saltpepper15_mini.png)
*Feature selection impact on mini dataset under Salt & Pepper noise (15%)*

![SaltPepper25 Mini](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_k_saltpepper25_mini.png)
*Feature selection impact on mini dataset under Salt & Pepper noise (25%)*

#### Small Dataset
![SaltPepper5 Small](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_k_saltpepper5_small.png)
*Feature selection impact on small dataset under Salt & Pepper noise (5%)*

![SaltPepper15 Small](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_k_saltpepper15_small.png)
*Feature selection impact on small dataset under Salt & Pepper noise (15%)*

![SaltPepper25 Small](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_k_saltpepper25_small.png)
*Feature selection impact on small dataset under Salt & Pepper noise (25%)*

#### Original Dataset
![SaltPepper5 Original](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_k_saltpepper5_original.png)
*Feature selection impact on original dataset under Salt & Pepper noise (5%)*

![SaltPepper15 Original](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_k_saltpepper15_original.png)
*Feature selection impact on original dataset under Salt & Pepper noise (15%)*

![SaltPepper25 Original](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_k_saltpepper25_original.png)
*Feature selection impact on original dataset under Salt & Pepper noise (25%)*

---

## 🛡️ Noise Robustness Analysis

### Mini Dataset

#### K=2 Features
![Noise Mini K2](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_saltpepper_mini_k2.png)
*Salt & Pepper noise robustness on mini dataset with k=2 features*

#### K=5 Features
![Noise Mini K5](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_saltpepper_mini_k5.png)
*Salt & Pepper noise robustness on mini dataset with k=5 features*

#### K=10 Features
![Noise Mini K10](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_saltpepper_mini_k10.png)
*Salt & Pepper noise robustness on mini dataset with k=10 features*

#### K=20 Features
![Noise Mini K20](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_saltpepper_mini_k20.png)
*Salt & Pepper noise robustness on mini dataset with k=20 features*

### Small Dataset

#### K=2 Features
![Noise Small K2](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_saltpepper_small_k2.png)
*Salt & Pepper noise robustness on small dataset with k=2 features*

#### K=5 Features
![Noise Small K5](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_saltpepper_small_k5.png)
*Salt & Pepper noise robustness on small dataset with k=5 features*

#### K=10 Features
![Noise Small K10](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_saltpepper_small_k10.png)
*Salt & Pepper noise robustness on small dataset with k=10 features*

#### K=20 Features
![Noise Small K20](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_saltpepper_small_k20.png)
*Salt & Pepper noise robustness on small dataset with k=20 features*

### Original Dataset

#### K=2 Features
![Noise Original K2](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_saltpepper_original_k2.png)
*Salt & Pepper noise robustness on original dataset with k=2 features*

#### K=5 Features
![Noise Original K5](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_saltpepper_original_k5.png)
*Salt & Pepper noise robustness on original dataset with k=5 features*

#### K=10 Features
![Noise Original K10](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_saltpepper_original_k10.png)
*Salt & Pepper noise robustness on original dataset with k=10 features*

#### K=20 Features
![Noise Original K20](experiments/s%26p/saltpepper_analysis/detailed/accuracy_vs_saltpepper_original_k20.png)
*Salt & Pepper noise robustness on original dataset with k=20 features*

---

## 📊 Summary Statistics

- **Noise Type**: Salt & Pepper (impulse) noise (5%, 15%, 25%)
- **Total Visualizations**: 37 plots
- **High-Level Comparisons**: 4 plots
- **Detailed Analysis**: 33 plots
- **Format**: PNG (publication-ready quality)

## 🎨 Visualization Features

- **Professional styling**: Clean, publication-ready aesthetics
- **Error bars**: Standard deviation indicators for statistical significance
- **Color coding**: Consistent color scheme across all plots
- **Clear labeling**: Comprehensive titles and axis labels
- **Statistical rigor**: Averaged over geographic areas for robust analysis
- **Noise-specific analysis**: Focus on impulse noise characteristics and robustness

---

*Generated from comprehensive analysis of Salt & Pepper noise Random Forest experiments comparing WST, Advanced Stats, and Hybrid feature extraction methods across different noise intensities (5%, 15%, 25%) and dataset sizes.*