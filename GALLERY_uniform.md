# 📊 Random Forest Analysis - Uniform Noise Visualization Gallery

This document presents a comprehensive gallery of all visualizations generated from the analysis of Uniform noise Random Forest experiments. The plots are organized by category for easy navigation and comparison.

## 📋 Table of Contents

1. [High-Level Comparisons](#high-level-comparisons)
2. [Dataset Size Analysis](#dataset-size-analysis)
3. [Feature Selection Analysis](#feature-selection-analysis)
4. [Noise Robustness Analysis](#noise-robustness-analysis)

---

## 🔍 High-Level Comparisons

### Overall Performance Summary
![Accuracy Heatmap Summary](experiments/uniform/uniform_analysis/comparisons/accuracy_heatmap_uniform_summary.png)
*Comprehensive heatmap showing accuracy across all Uniform noise experimental conditions*

### Method Comparison
![Accuracy vs Method Boxplot](experiments/uniform/uniform_analysis/comparisons/accuracy_vs_method_boxplot_uniform.png)
*Statistical comparison of feature extraction methods (WST, Advanced Stats, Hybrid) under Uniform noise*

### Dataset Size Impact
![Accuracy vs Dataset Size](experiments/uniform/uniform_analysis/comparisons/accuracy_vs_dataset_size_overall.png)
*Overall performance improvement with dataset size under Uniform noise conditions*

### Noise Robustness Overview
![Accuracy vs Noise Level](experiments/uniform/uniform_analysis/comparisons/accuracy_vs_uniform_noise_overall.png)
*Performance degradation under different Uniform noise ranges (±10, ±25, ±40)*

---

## 📈 Dataset Size Analysis

### Clean Dataset Conditions

#### K=2 Features
![Clean K2](experiments/uniform/uniform_analysis/detailed/accuracy_vs_dataset_clean_k2.png)
*Performance vs dataset size with minimal feature selection (k=2)*

#### K=5 Features
![Clean K5](experiments/uniform/uniform_analysis/detailed/accuracy_vs_dataset_clean_k5.png)
*Performance vs dataset size with small feature selection (k=5)*

#### K=10 Features
![Clean K10](experiments/uniform/uniform_analysis/detailed/accuracy_vs_dataset_clean_k10.png)
*Performance vs dataset size with medium feature selection (k=10)*

#### K=20 Features
![Clean K20](experiments/uniform/uniform_analysis/detailed/accuracy_vs_dataset_clean_k20.png)
*Performance vs dataset size with large feature selection (k=20)*

### Uniform Noise Conditions

#### K=2 Features
![Uniform10 K2](experiments/uniform/uniform_analysis/detailed/accuracy_vs_dataset_uniform10_k2.png)
*Performance vs dataset size under Uniform noise (±10) with k=2 features*

![Uniform25 K2](experiments/uniform/uniform_analysis/detailed/accuracy_vs_dataset_uniform25_k2.png)
*Performance vs dataset size under Uniform noise (±25) with k=2 features*

![Uniform40 K2](experiments/uniform/uniform_analysis/detailed/accuracy_vs_dataset_uniform40_k2.png)
*Performance vs dataset size under Uniform noise (±40) with k=2 features*

#### K=5 Features
![Uniform10 K5](experiments/uniform/uniform_analysis/detailed/accuracy_vs_dataset_uniform10_k5.png)
*Performance vs dataset size under Uniform noise (±10) with k=5 features*

![Uniform25 K5](experiments/uniform/uniform_analysis/detailed/accuracy_vs_dataset_uniform25_k5.png)
*Performance vs dataset size under Uniform noise (±25) with k=5 features*

![Uniform40 K5](experiments/uniform/uniform_analysis/detailed/accuracy_vs_dataset_uniform40_k5.png)
*Performance vs dataset size under Uniform noise (±40) with k=5 features*

#### K=10 Features
![Uniform10 K10](experiments/uniform/uniform_analysis/detailed/accuracy_vs_dataset_uniform10_k10.png)
*Performance vs dataset size under Uniform noise (±10) with k=10 features*

![Uniform25 K10](experiments/uniform/uniform_analysis/detailed/accuracy_vs_dataset_uniform25_k10.png)
*Performance vs dataset size under Uniform noise (±25) with k=10 features*

![Uniform40 K10](experiments/uniform/uniform_analysis/detailed/accuracy_vs_dataset_uniform40_k10.png)
*Performance vs dataset size under Uniform noise (±40) with k=10 features*

#### K=20 Features
![Uniform10 K20](experiments/uniform/uniform_analysis/detailed/accuracy_vs_dataset_uniform10_k20.png)
*Performance vs dataset size under Uniform noise (±10) with k=20 features*

![Uniform25 K20](experiments/uniform/uniform_analysis/detailed/accuracy_vs_dataset_uniform25_k20.png)
*Performance vs dataset size under Uniform noise (±25) with k=20 features*

![Uniform40 K20](experiments/uniform/uniform_analysis/detailed/accuracy_vs_dataset_uniform40_k20.png)
*Performance vs dataset size under Uniform noise (±40) with k=20 features*

---

## 🎯 Feature Selection Analysis

### Clean Dataset Conditions

#### Mini Dataset
![Clean Mini](experiments/uniform/uniform_analysis/detailed/accuracy_vs_k_clean_mini.png)
*Feature selection impact on mini dataset (clean conditions)*

#### Small Dataset
![Clean Small](experiments/uniform/uniform_analysis/detailed/accuracy_vs_k_clean_small.png)
*Feature selection impact on small dataset (clean conditions)*

#### Original Dataset
![Clean Original](experiments/uniform/uniform_analysis/detailed/accuracy_vs_k_clean_original.png)
*Feature selection impact on original dataset (clean conditions)*

### Uniform Noise Conditions

#### Mini Dataset
![Uniform10 Mini](experiments/uniform/uniform_analysis/detailed/accuracy_vs_k_uniform10_mini.png)
*Feature selection impact on mini dataset under Uniform noise (±10)*

![Uniform25 Mini](experiments/uniform/uniform_analysis/detailed/accuracy_vs_k_uniform25_mini.png)
*Feature selection impact on mini dataset under Uniform noise (±25)*

![Uniform40 Mini](experiments/uniform/uniform_analysis/detailed/accuracy_vs_k_uniform40_mini.png)
*Feature selection impact on mini dataset under Uniform noise (±40)*

#### Small Dataset
![Uniform10 Small](experiments/uniform/uniform_analysis/detailed/accuracy_vs_k_uniform10_small.png)
*Feature selection impact on small dataset under Uniform noise (±10)*

![Uniform25 Small](experiments/uniform/uniform_analysis/detailed/accuracy_vs_k_uniform25_small.png)
*Feature selection impact on small dataset under Uniform noise (±25)*

![Uniform40 Small](experiments/uniform/uniform_analysis/detailed/accuracy_vs_k_uniform40_small.png)
*Feature selection impact on small dataset under Uniform noise (±40)*

#### Original Dataset
![Uniform10 Original](experiments/uniform/uniform_analysis/detailed/accuracy_vs_k_uniform10_original.png)
*Feature selection impact on original dataset under Uniform noise (±10)*

![Uniform25 Original](experiments/uniform/uniform_analysis/detailed/accuracy_vs_k_uniform25_original.png)
*Feature selection impact on original dataset under Uniform noise (±25)*

![Uniform40 Original](experiments/uniform/uniform_analysis/detailed/accuracy_vs_k_uniform40_original.png)
*Feature selection impact on original dataset under Uniform noise (±40)*

---

## 🛡️ Noise Robustness Analysis

### Mini Dataset

#### K=2 Features
![Noise Mini K2](experiments/uniform/uniform_analysis/detailed/accuracy_vs_uniform_mini_k2.png)
*Uniform noise robustness on mini dataset with k=2 features*

#### K=5 Features
![Noise Mini K5](experiments/uniform/uniform_analysis/detailed/accuracy_vs_uniform_mini_k5.png)
*Uniform noise robustness on mini dataset with k=5 features*

#### K=10 Features
![Noise Mini K10](experiments/uniform/uniform_analysis/detailed/accuracy_vs_uniform_mini_k10.png)
*Uniform noise robustness on mini dataset with k=10 features*

#### K=20 Features
![Noise Mini K20](experiments/uniform/uniform_analysis/detailed/accuracy_vs_uniform_mini_k20.png)
*Uniform noise robustness on mini dataset with k=20 features*

### Small Dataset

#### K=2 Features
![Noise Small K2](experiments/uniform/uniform_analysis/detailed/accuracy_vs_uniform_small_k2.png)
*Uniform noise robustness on small dataset with k=2 features*

#### K=5 Features
![Noise Small K5](experiments/uniform/uniform_analysis/detailed/accuracy_vs_uniform_small_k5.png)
*Uniform noise robustness on small dataset with k=5 features*

#### K=10 Features
![Noise Small K10](experiments/uniform/uniform_analysis/detailed/accuracy_vs_uniform_small_k10.png)
*Uniform noise robustness on small dataset with k=10 features*

#### K=20 Features
![Noise Small K20](experiments/uniform/uniform_analysis/detailed/accuracy_vs_uniform_small_k20.png)
*Uniform noise robustness on small dataset with k=20 features*

### Original Dataset

#### K=2 Features
![Noise Original K2](experiments/uniform/uniform_analysis/detailed/accuracy_vs_uniform_original_k2.png)
*Uniform noise robustness on original dataset with k=2 features*

#### K=5 Features
![Noise Original K5](experiments/uniform/uniform_analysis/detailed/accuracy_vs_uniform_original_k5.png)
*Uniform noise robustness on original dataset with k=5 features*

#### K=10 Features
![Noise Original K10](experiments/uniform/uniform_analysis/detailed/accuracy_vs_uniform_original_k10.png)
*Uniform noise robustness on original dataset with k=10 features*

#### K=20 Features
![Noise Original K20](experiments/uniform/uniform_analysis/detailed/accuracy_vs_uniform_original_k20.png)
*Uniform noise robustness on original dataset with k=20 features*

---

## 📊 Summary Statistics

- **Noise Type**: Uniform (additive) noise (±10, ±25, ±40)
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
- **Noise-specific analysis**: Focus on uniform distribution noise characteristics and robustness

---

*Generated from comprehensive analysis of Uniform noise Random Forest experiments comparing WST, Advanced Stats, and Hybrid feature extraction methods across different noise ranges (±10, ±25, ±40) and dataset sizes.*