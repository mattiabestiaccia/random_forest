# 📊 Random Forest Analysis - Visualization Gallery

This document presents a comprehensive gallery of all visualizations generated from the analysis of 324 Random Forest experiments. The plots are organized by category for easy navigation and comparison.

## 📋 Table of Contents

1. [High-Level Comparisons](#high-level-comparisons)
2. [Dataset Size Analysis](#dataset-size-analysis)
3. [Feature Selection Analysis](#feature-selection-analysis)
4. [Noise Robustness Analysis](#noise-robustness-analysis)

---

## 🔍 High-Level Comparisons

### Overall Performance Summary
![Accuracy Heatmap Summary](./gaussian_analysis/comparisons/accuracy_heatmap_summary.png)
*Comprehensive heatmap showing accuracy across all experimental conditions*

### Method Comparison
![Accuracy vs Method Boxplot](./gaussian_analysis/comparisons/accuracy_vs_method_boxplot.png)
*Statistical comparison of feature extraction methods (WST, Advanced Stats, Hybrid)*

### Dataset Size Impact
![Accuracy vs Dataset Size](./gaussian_analysis/comparisons/accuracy_vs_dataset_size_overall.png)
*Overall performance improvement with dataset size (Mini → Small → Original)*

### Noise Robustness Overview
![Accuracy vs Noise Level](./gaussian_analysis/comparisons/accuracy_vs_noise_overall.png)
*Performance degradation under different noise conditions (Clean → Gaussian σ=30 → Gaussian σ=50)*

---

## 📈 Dataset Size Analysis

### Clean Dataset Conditions

#### K=2 Features
![Clean K2](./gaussian_analysis/detailed/accuracy_vs_dataset_clean_k2.png)
*Performance vs dataset size with minimal feature selection (k=2)*

#### K=5 Features
![Clean K5](./gaussian_analysis/detailed/accuracy_vs_dataset_clean_k5.png)
*Performance vs dataset size with small feature selection (k=5)*

#### K=10 Features
![Clean K10](./gaussian_analysis/detailed/accuracy_vs_dataset_clean_k10.png)
*Performance vs dataset size with medium feature selection (k=10)*

#### K=20 Features
![Clean K20](./gaussian_analysis/detailed/accuracy_vs_dataset_clean_k20.png)
*Performance vs dataset size with large feature selection (k=20)*

### Gaussian Noise σ=30 Conditions

#### K=2 Features
![Gaussian30 K2](./gaussian_analysis/detailed/accuracy_vs_dataset_gaussian30_k2.png)
*Performance vs dataset size under moderate noise with k=2 features*

#### K=5 Features
![Gaussian30 K5](./gaussian_analysis/detailed/accuracy_vs_dataset_gaussian30_k5.png)
*Performance vs dataset size under moderate noise with k=5 features*

#### K=10 Features
![Gaussian30 K10](./gaussian_analysis/detailed/accuracy_vs_dataset_gaussian30_k10.png)
*Performance vs dataset size under moderate noise with k=10 features*

#### K=20 Features
![Gaussian30 K20](./gaussian_analysis/detailed/accuracy_vs_dataset_gaussian30_k20.png)
*Performance vs dataset size under moderate noise with k=20 features*

### Gaussian Noise σ=50 Conditions

#### K=2 Features
![Gaussian50 K2](./gaussian_analysis/detailed/accuracy_vs_dataset_gaussian50_k2.png)
*Performance vs dataset size under high noise with k=2 features*

#### K=5 Features
![Gaussian50 K5](./gaussian_analysis/detailed/accuracy_vs_dataset_gaussian50_k5.png)
*Performance vs dataset size under high noise with k=5 features*

#### K=10 Features
![Gaussian50 K10](./gaussian_analysis/detailed/accuracy_vs_dataset_gaussian50_k10.png)
*Performance vs dataset size under high noise with k=10 features*

#### K=20 Features
![Gaussian50 K20](./gaussian_analysis/detailed/accuracy_vs_dataset_gaussian50_k20.png)
*Performance vs dataset size under high noise with k=20 features*

---

## 🎯 Feature Selection Analysis

### Clean Dataset Conditions

#### Mini Dataset
![Clean Mini](./gaussian_analysis/detailed/accuracy_vs_k_clean_mini.png)
*Feature selection impact on mini dataset (clean conditions)*

#### Small Dataset
![Clean Small](./gaussian_analysis/detailed/accuracy_vs_k_clean_small.png)
*Feature selection impact on small dataset (clean conditions)*

#### Original Dataset
![Clean Original](./gaussian_analysis/detailed/accuracy_vs_k_clean_original.png)
*Feature selection impact on original dataset (clean conditions)*

### Gaussian Noise σ=30 Conditions

#### Mini Dataset
![Gaussian30 Mini](./gaussian_analysis/detailed/accuracy_vs_k_gaussian30_mini.png)
*Feature selection impact on mini dataset under moderate noise*

#### Small Dataset
![Gaussian30 Small](./gaussian_analysis/detailed/accuracy_vs_k_gaussian30_small.png)
*Feature selection impact on small dataset under moderate noise*

#### Original Dataset
![Gaussian30 Original](./gaussian_analysis/detailed/accuracy_vs_k_gaussian30_original.png)
*Feature selection impact on original dataset under moderate noise*

### Gaussian Noise σ=50 Conditions

#### Mini Dataset
![Gaussian50 Mini](./gaussian_analysis/detailed/accuracy_vs_k_gaussian50_mini.png)
*Feature selection impact on mini dataset under high noise*

#### Small Dataset
![Gaussian50 Small](./gaussian_analysis/detailed/accuracy_vs_k_gaussian50_small.png)
*Feature selection impact on small dataset under high noise*

#### Original Dataset
![Gaussian50 Original](./gaussian_analysis/detailed/accuracy_vs_k_gaussian50_original.png)
*Feature selection impact on original dataset under high noise*

---

## 🛡️ Noise Robustness Analysis

### Mini Dataset

#### K=2 Features
![Noise Mini K2](./gaussian_analysis/detailed/accuracy_vs_noise_mini_k2.png)
*Noise robustness on mini dataset with k=2 features*

#### K=5 Features
![Noise Mini K5](./gaussian_analysis/detailed/accuracy_vs_noise_mini_k5.png)
*Noise robustness on mini dataset with k=5 features*

#### K=10 Features
![Noise Mini K10](./gaussian_analysis/detailed/accuracy_vs_noise_mini_k10.png)
*Noise robustness on mini dataset with k=10 features*

#### K=20 Features
![Noise Mini K20](./gaussian_analysis/detailed/accuracy_vs_noise_mini_k20.png)
*Noise robustness on mini dataset with k=20 features*

### Small Dataset

#### K=2 Features
![Noise Small K2](./gaussian_analysis/detailed/accuracy_vs_noise_small_k2.png)
*Noise robustness on small dataset with k=2 features*

#### K=5 Features
![Noise Small K5](./gaussian_analysis/detailed/accuracy_vs_noise_small_k5.png)
*Noise robustness on small dataset with k=5 features*

#### K=10 Features
![Noise Small K10](./gaussian_analysis/detailed/accuracy_vs_noise_small_k10.png)
*Noise robustness on small dataset with k=10 features*

#### K=20 Features
![Noise Small K20](./gaussian_analysis/detailed/accuracy_vs_noise_small_k20.png)
*Noise robustness on small dataset with k=20 features*

### Original Dataset

#### K=2 Features
![Noise Original K2](./gaussian_analysis/detailed/accuracy_vs_noise_original_k2.png)
*Noise robustness on original dataset with k=2 features*

#### K=5 Features
![Noise Original K5](./gaussian_analysis/detailed/accuracy_vs_noise_original_k5.png)
*Noise robustness on original dataset with k=5 features*

#### K=10 Features
![Noise Original K10](./gaussian_analysis/detailed/accuracy_vs_noise_original_k10.png)
*Noise robustness on original dataset with k=10 features*

#### K=20 Features
![Noise Original K20](./gaussian_analysis/detailed/accuracy_vs_noise_original_k20.png)
*Noise robustness on original dataset with k=20 features*

---

## 📊 Summary Statistics

- **Total Visualizations**: 37 plots
- **High-Level Comparisons**: 4 plots
- **Detailed Analysis**: 33 plots
- **Total Size**: ~6.6MB
- **Format**: PNG (publication-ready quality)

## 🎨 Visualization Features

- **Professional styling**: Clean, publication-ready aesthetics
- **Error bars**: Standard deviation indicators for statistical significance
- **Color coding**: Consistent color scheme across all plots
- **Clear labeling**: Comprehensive titles and axis labels
- **Statistical rigor**: Averaged over geographic areas for robust analysis

---

*Generated from comprehensive analysis of 324 Random Forest experiments comparing WST, Advanced Stats, and Hybrid feature extraction methods across different noise conditions and dataset sizes.*