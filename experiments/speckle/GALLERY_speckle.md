# 📊 Random Forest Analysis - Speckle Noise Visualization Gallery

This document presents a comprehensive gallery of all visualizations generated from the analysis of Speckle noise Random Forest experiments. The plots are organized by category for easy navigation and comparison.

## 📋 Table of Contents

1. [High-Level Comparisons](#high-level-comparisons)
2. [Dataset Size Analysis](#dataset-size-analysis)
3. [Feature Selection Analysis](#feature-selection-analysis)
4. [Noise Robustness Analysis](#noise-robustness-analysis)

---

## 🔍 High-Level Comparisons

### Overall Performance Summary
![Accuracy Heatmap Summary](./speckle_analysis/comparisons/accuracy_heatmap_speckle_summary.png)
*Comprehensive heatmap showing accuracy across all Speckle noise experimental conditions*

### Method Comparison
![Accuracy vs Method Boxplot](./speckle_analysis/comparisons/accuracy_vs_method_boxplot_speckle.png)
*Statistical comparison of feature extraction methods (WST, Advanced Stats, Hybrid) under Speckle noise*

### Dataset Size Impact
![Accuracy vs Dataset Size](./speckle_analysis/comparisons/accuracy_vs_dataset_size_overall.png)
*Overall performance improvement with dataset size under Speckle noise conditions*

### Noise Robustness Overview
![Accuracy vs Noise Level](./speckle_analysis/comparisons/accuracy_vs_speckle_noise_overall.png)
*Performance degradation under different Speckle noise variances (σ²=0.15, σ²=0.35, σ²=0.55)*

---

## 📈 Dataset Size Analysis

### Clean Dataset Conditions

#### K=2 Features
![Clean K2](./speckle_analysis/detailed/accuracy_vs_dataset_clean_k2.png)
*Performance vs dataset size with minimal feature selection (k=2)*

#### K=5 Features
![Clean K5](./speckle_analysis/detailed/accuracy_vs_dataset_clean_k5.png)
*Performance vs dataset size with small feature selection (k=5)*

#### K=10 Features
![Clean K10](./speckle_analysis/detailed/accuracy_vs_dataset_clean_k10.png)
*Performance vs dataset size with medium feature selection (k=10)*

#### K=20 Features
![Clean K20](./speckle_analysis/detailed/accuracy_vs_dataset_clean_k20.png)
*Performance vs dataset size with large feature selection (k=20)*

### Speckle Noise Conditions

#### K=2 Features
![Speckle15 K2](./speckle_analysis/detailed/accuracy_vs_dataset_speckle15_k2.png)
*Performance vs dataset size under Speckle noise (σ²=0.15) with k=2 features*

![Speckle35 K2](./speckle_analysis/detailed/accuracy_vs_dataset_speckle35_k2.png)
*Performance vs dataset size under Speckle noise (σ²=0.35) with k=2 features*

![Speckle55 K2](./speckle_analysis/detailed/accuracy_vs_dataset_speckle55_k2.png)
*Performance vs dataset size under Speckle noise (σ²=0.55) with k=2 features*

#### K=5 Features
![Speckle15 K5](./speckle_analysis/detailed/accuracy_vs_dataset_speckle15_k5.png)
*Performance vs dataset size under Speckle noise (σ²=0.15) with k=5 features*

![Speckle35 K5](./speckle_analysis/detailed/accuracy_vs_dataset_speckle35_k5.png)
*Performance vs dataset size under Speckle noise (σ²=0.35) with k=5 features*

![Speckle55 K5](./speckle_analysis/detailed/accuracy_vs_dataset_speckle55_k5.png)
*Performance vs dataset size under Speckle noise (σ²=0.55) with k=5 features*

#### K=10 Features
![Speckle15 K10](./speckle_analysis/detailed/accuracy_vs_dataset_speckle15_k10.png)
*Performance vs dataset size under Speckle noise (σ²=0.15) with k=10 features*

![Speckle35 K10](./speckle_analysis/detailed/accuracy_vs_dataset_speckle35_k10.png)
*Performance vs dataset size under Speckle noise (σ²=0.35) with k=10 features*

![Speckle55 K10](./speckle_analysis/detailed/accuracy_vs_dataset_speckle55_k10.png)
*Performance vs dataset size under Speckle noise (σ²=0.55) with k=10 features*

#### K=20 Features
![Speckle15 K20](./speckle_analysis/detailed/accuracy_vs_dataset_speckle15_k20.png)
*Performance vs dataset size under Speckle noise (σ²=0.15) with k=20 features*

![Speckle35 K20](./speckle_analysis/detailed/accuracy_vs_dataset_speckle35_k20.png)
*Performance vs dataset size under Speckle noise (σ²=0.35) with k=20 features*

![Speckle55 K20](./speckle_analysis/detailed/accuracy_vs_dataset_speckle55_k20.png)
*Performance vs dataset size under Speckle noise (σ²=0.55) with k=20 features*

---

## 🎯 Feature Selection Analysis

### Clean Dataset Conditions

#### Mini Dataset
![Clean Mini](./speckle_analysis/detailed/accuracy_vs_k_clean_mini.png)
*Feature selection impact on mini dataset (clean conditions)*

#### Small Dataset
![Clean Small](./speckle_analysis/detailed/accuracy_vs_k_clean_small.png)
*Feature selection impact on small dataset (clean conditions)*

#### Original Dataset
![Clean Original](./speckle_analysis/detailed/accuracy_vs_k_clean_original.png)
*Feature selection impact on original dataset (clean conditions)*

### Speckle Noise Conditions

#### Mini Dataset
![Speckle15 Mini](./speckle_analysis/detailed/accuracy_vs_k_speckle15_mini.png)
*Feature selection impact on mini dataset under Speckle noise (σ²=0.15)*

![Speckle35 Mini](./speckle_analysis/detailed/accuracy_vs_k_speckle35_mini.png)
*Feature selection impact on mini dataset under Speckle noise (σ²=0.35)*

![Speckle55 Mini](./speckle_analysis/detailed/accuracy_vs_k_speckle55_mini.png)
*Feature selection impact on mini dataset under Speckle noise (σ²=0.55)*

#### Small Dataset
![Speckle15 Small](./speckle_analysis/detailed/accuracy_vs_k_speckle15_small.png)
*Feature selection impact on small dataset under Speckle noise (σ²=0.15)*

![Speckle35 Small](./speckle_analysis/detailed/accuracy_vs_k_speckle35_small.png)
*Feature selection impact on small dataset under Speckle noise (σ²=0.35)*

![Speckle55 Small](./speckle_analysis/detailed/accuracy_vs_k_speckle55_small.png)
*Feature selection impact on small dataset under Speckle noise (σ²=0.55)*

#### Original Dataset
![Speckle15 Original](./speckle_analysis/detailed/accuracy_vs_k_speckle15_original.png)
*Feature selection impact on original dataset under Speckle noise (σ²=0.15)*

![Speckle35 Original](./speckle_analysis/detailed/accuracy_vs_k_speckle35_original.png)
*Feature selection impact on original dataset under Speckle noise (σ²=0.35)*

![Speckle55 Original](./speckle_analysis/detailed/accuracy_vs_k_speckle55_original.png)
*Feature selection impact on original dataset under Speckle noise (σ²=0.55)*

---

## 🛡️ Noise Robustness Analysis

### Mini Dataset

#### K=2 Features
![Noise Mini K2](./speckle_analysis/detailed/accuracy_vs_speckle_mini_k2.png)
*Speckle noise robustness on mini dataset with k=2 features*

#### K=5 Features
![Noise Mini K5](./speckle_analysis/detailed/accuracy_vs_speckle_mini_k5.png)
*Speckle noise robustness on mini dataset with k=5 features*

#### K=10 Features
![Noise Mini K10](./speckle_analysis/detailed/accuracy_vs_speckle_mini_k10.png)
*Speckle noise robustness on mini dataset with k=10 features*

#### K=20 Features
![Noise Mini K20](./speckle_analysis/detailed/accuracy_vs_speckle_mini_k20.png)
*Speckle noise robustness on mini dataset with k=20 features*

### Small Dataset

#### K=2 Features
![Noise Small K2](./speckle_analysis/detailed/accuracy_vs_speckle_small_k2.png)
*Speckle noise robustness on small dataset with k=2 features*

#### K=5 Features
![Noise Small K5](./speckle_analysis/detailed/accuracy_vs_speckle_small_k5.png)
*Speckle noise robustness on small dataset with k=5 features*

#### K=10 Features
![Noise Small K10](./speckle_analysis/detailed/accuracy_vs_speckle_small_k10.png)
*Speckle noise robustness on small dataset with k=10 features*

#### K=20 Features
![Noise Small K20](./speckle_analysis/detailed/accuracy_vs_speckle_small_k20.png)
*Speckle noise robustness on small dataset with k=20 features*

### Original Dataset

#### K=2 Features
![Noise Original K2](./speckle_analysis/detailed/accuracy_vs_speckle_original_k2.png)
*Speckle noise robustness on original dataset with k=2 features*

#### K=5 Features
![Noise Original K5](./speckle_analysis/detailed/accuracy_vs_speckle_original_k5.png)
*Speckle noise robustness on original dataset with k=5 features*

#### K=10 Features
![Noise Original K10](./speckle_analysis/detailed/accuracy_vs_speckle_original_k10.png)
*Speckle noise robustness on original dataset with k=10 features*

#### K=20 Features
![Noise Original K20](./speckle_analysis/detailed/accuracy_vs_speckle_original_k20.png)
*Speckle noise robustness on original dataset with k=20 features*

---

## 📊 Summary Statistics

- **Noise Type**: Speckle (multiplicative) noise (σ²=0.15, σ²=0.35, σ²=0.55)
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
- **Noise-specific analysis**: Focus on multiplicative noise characteristics and robustness

---

*Generated from comprehensive analysis of Speckle noise Random Forest experiments comparing WST, Advanced Stats, and Hybrid feature extraction methods across different noise variances (σ²=0.15, σ²=0.35, σ²=0.55) and dataset sizes.*