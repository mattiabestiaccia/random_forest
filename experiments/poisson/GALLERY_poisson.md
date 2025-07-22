# 📊 Random Forest Analysis - Poisson Noise Visualization Gallery

This document presents a comprehensive gallery of all visualizations generated from the analysis of Poisson noise Random Forest experiments. The plots are organized by category for easy navigation and comparison.

## 📋 Table of Contents

1. [High-Level Comparisons](#high-level-comparisons)
2. [Dataset Size Analysis](#dataset-size-analysis)
3. [Feature Selection Analysis](#feature-selection-analysis)
4. [Noise Robustness Analysis](#noise-robustness-analysis)

---

## 🔍 High-Level Comparisons

### Overall Performance Summary
![Accuracy Heatmap Summary](experiments/poisson/poisson_analysis/comparisons/accuracy_heatmap_summary.png)
*Comprehensive heatmap showing accuracy across all Poisson noise experimental conditions*

### Method Comparison
![Accuracy vs Method Boxplot](experiments/poisson/poisson_analysis/comparisons/accuracy_vs_method_boxplot.png)
*Statistical comparison of feature extraction methods (WST, Advanced Stats, Hybrid) under Poisson noise*

### Dataset Size Impact
![Accuracy vs Dataset Size](experiments/poisson/poisson_analysis/comparisons/accuracy_vs_dataset_size_overall.png)
*Overall performance improvement with dataset size under Poisson noise conditions*

### Noise Robustness Overview
![Accuracy vs Noise Level](experiments/poisson/poisson_analysis/comparisons/accuracy_vs_noise_overall.png)
*Performance degradation under different Poisson noise intensities*

---

## 📈 Dataset Size Analysis

### Clean Dataset Conditions

#### K=2 Features
![Clean K2](experiments/poisson/poisson_analysis/detailed/accuracy_vs_dataset_clean_k2.png)
*Performance vs dataset size with minimal feature selection (k=2)*

#### K=5 Features
![Clean K5](experiments/poisson/poisson_analysis/detailed/accuracy_vs_dataset_clean_k5.png)
*Performance vs dataset size with small feature selection (k=5)*

#### K=10 Features
![Clean K10](experiments/poisson/poisson_analysis/detailed/accuracy_vs_dataset_clean_k10.png)
*Performance vs dataset size with medium feature selection (k=10)*

#### K=20 Features
![Clean K20](experiments/poisson/poisson_analysis/detailed/accuracy_vs_dataset_clean_k20.png)
*Performance vs dataset size with large feature selection (k=20)*

### Poisson Noise Conditions

#### K=2 Features
![Poisson K2](experiments/poisson/poisson_analysis/detailed/accuracy_vs_dataset_poisson_k2.png)
*Performance vs dataset size under Poisson noise with k=2 features*

#### K=5 Features
![Poisson K5](experiments/poisson/poisson_analysis/detailed/accuracy_vs_dataset_poisson_k5.png)
*Performance vs dataset size under Poisson noise with k=5 features*

#### K=10 Features
![Poisson K10](experiments/poisson/poisson_analysis/detailed/accuracy_vs_dataset_poisson_k10.png)
*Performance vs dataset size under Poisson noise with k=10 features*

#### K=20 Features
![Poisson K20](experiments/poisson/poisson_analysis/detailed/accuracy_vs_dataset_poisson_k20.png)
*Performance vs dataset size under Poisson noise with k=20 features*

---

## 🎯 Feature Selection Analysis

### Clean Dataset Conditions

#### Mini Dataset
![Clean Mini](experiments/poisson/poisson_analysis/detailed/accuracy_vs_k_clean_mini.png)
*Feature selection impact on mini dataset (clean conditions)*

#### Small Dataset
![Clean Small](experiments/poisson/poisson_analysis/detailed/accuracy_vs_k_clean_small.png)
*Feature selection impact on small dataset (clean conditions)*

#### Original Dataset
![Clean Original](experiments/poisson/poisson_analysis/detailed/accuracy_vs_k_clean_original.png)
*Feature selection impact on original dataset (clean conditions)*

### Poisson Noise Conditions

#### Mini Dataset
![Poisson Mini](experiments/poisson/poisson_analysis/detailed/accuracy_vs_k_poisson_mini.png)
*Feature selection impact on mini dataset under Poisson noise*

#### Small Dataset
![Poisson Small](experiments/poisson/poisson_analysis/detailed/accuracy_vs_k_poisson_small.png)
*Feature selection impact on small dataset under Poisson noise*

#### Original Dataset
![Poisson Original](experiments/poisson/poisson_analysis/detailed/accuracy_vs_k_poisson_original.png)
*Feature selection impact on original dataset under Poisson noise*

---

## 🛡️ Noise Robustness Analysis

### Mini Dataset

#### K=2 Features
![Noise Mini K2](experiments/poisson/poisson_analysis/detailed/accuracy_vs_noise_mini_k2.png)
*Poisson noise robustness on mini dataset with k=2 features*

#### K=5 Features
![Noise Mini K5](experiments/poisson/poisson_analysis/detailed/accuracy_vs_noise_mini_k5.png)
*Poisson noise robustness on mini dataset with k=5 features*

#### K=10 Features
![Noise Mini K10](experiments/poisson/poisson_analysis/detailed/accuracy_vs_noise_mini_k10.png)
*Poisson noise robustness on mini dataset with k=10 features*

#### K=20 Features
![Noise Mini K20](experiments/poisson/poisson_analysis/detailed/accuracy_vs_noise_mini_k20.png)
*Poisson noise robustness on mini dataset with k=20 features*

### Small Dataset

#### K=2 Features
![Noise Small K2](experiments/poisson/poisson_analysis/detailed/accuracy_vs_noise_small_k2.png)
*Poisson noise robustness on small dataset with k=2 features*

#### K=5 Features
![Noise Small K5](experiments/poisson/poisson_analysis/detailed/accuracy_vs_noise_small_k5.png)
*Poisson noise robustness on small dataset with k=5 features*

#### K=10 Features
![Noise Small K10](experiments/poisson/poisson_analysis/detailed/accuracy_vs_noise_small_k10.png)
*Poisson noise robustness on small dataset with k=10 features*

#### K=20 Features
![Noise Small K20](experiments/poisson/poisson_analysis/detailed/accuracy_vs_noise_small_k20.png)
*Poisson noise robustness on small dataset with k=20 features*

### Original Dataset

#### K=2 Features
![Noise Original K2](experiments/poisson/poisson_analysis/detailed/accuracy_vs_noise_original_k2.png)
*Poisson noise robustness on original dataset with k=2 features*

#### K=5 Features
![Noise Original K5](experiments/poisson/poisson_analysis/detailed/accuracy_vs_noise_original_k5.png)
*Poisson noise robustness on original dataset with k=5 features*

#### K=10 Features
![Noise Original K10](experiments/poisson/poisson_analysis/detailed/accuracy_vs_noise_original_k10.png)
*Poisson noise robustness on original dataset with k=10 features*

#### K=20 Features
![Noise Original K20](experiments/poisson/poisson_analysis/detailed/accuracy_vs_noise_original_k20.png)
*Poisson noise robustness on original dataset with k=20 features*

---

## 📊 Summary Statistics

- **Noise Type**: Poisson noise
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
- **Noise-specific analysis**: Focus on Poisson noise characteristics and robustness

---

*Generated from comprehensive analysis of Poisson noise Random Forest experiments comparing WST, Advanced Stats, and Hybrid feature extraction methods across different noise intensities and dataset sizes.*