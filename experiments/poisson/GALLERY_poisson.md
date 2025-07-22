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
![Accuracy Heatmap Summary](poisson_analysis/comparisons/accuracy_heatmap_poisson_summary.png)
*Comprehensive heatmap showing accuracy across all Poisson noise experimental conditions*

### Method Comparison
![Accuracy vs Method Boxplot](poisson_analysis/comparisons/accuracy_vs_method_boxplot_poisson.png)
*Statistical comparison of feature extraction methods (WST, Advanced Stats, Hybrid) under Poisson noise*

### Dataset Size Impact
![Accuracy vs Dataset Size](poisson_analysis/comparisons/accuracy_vs_dataset_size_overall.png)
*Overall performance improvement with dataset size under Poisson noise conditions*

### Noise Robustness Overview
![Accuracy vs Noise Level](poisson_analysis/comparisons/accuracy_vs_poisson_noise_overall.png)
*Performance degradation under different Poisson noise intensities (λ=40, λ=60)*

---

## 📈 Dataset Size Analysis

### Clean Dataset Conditions

#### K=2 Features
![Clean K2](poisson_analysis/detailed/accuracy_vs_dataset_clean_k2.png)
*Performance vs dataset size with minimal feature selection (k=2)*

#### K=5 Features
![Clean K5](poisson_analysis/detailed/accuracy_vs_dataset_clean_k5.png)
*Performance vs dataset size with small feature selection (k=5)*

#### K=10 Features
![Clean K10](poisson_analysis/detailed/accuracy_vs_dataset_clean_k10.png)
*Performance vs dataset size with medium feature selection (k=10)*

#### K=20 Features
![Clean K20](poisson_analysis/detailed/accuracy_vs_dataset_clean_k20.png)
*Performance vs dataset size with large feature selection (k=20)*

### Poisson Noise Conditions

#### K=2 Features
![Poisson40 K2](poisson_analysis/detailed/accuracy_vs_dataset_poisson40_k2.png)
*Performance vs dataset size under Poisson noise (λ=40) with k=2 features*

![Poisson60 K2](poisson_analysis/detailed/accuracy_vs_dataset_poisson60_k2.png)
*Performance vs dataset size under Poisson noise (λ=60) with k=2 features*

#### K=5 Features
![Poisson40 K5](poisson_analysis/detailed/accuracy_vs_dataset_poisson40_k5.png)
*Performance vs dataset size under Poisson noise (λ=40) with k=5 features*

![Poisson60 K5](poisson_analysis/detailed/accuracy_vs_dataset_poisson60_k5.png)
*Performance vs dataset size under Poisson noise (λ=60) with k=5 features*

#### K=10 Features
![Poisson40 K10](poisson_analysis/detailed/accuracy_vs_dataset_poisson40_k10.png)
*Performance vs dataset size under Poisson noise (λ=40) with k=10 features*

![Poisson60 K10](poisson_analysis/detailed/accuracy_vs_dataset_poisson60_k10.png)
*Performance vs dataset size under Poisson noise (λ=60) with k=10 features*

#### K=20 Features
![Poisson40 K20](poisson_analysis/detailed/accuracy_vs_dataset_poisson40_k20.png)
*Performance vs dataset size under Poisson noise (λ=40) with k=20 features*

![Poisson60 K20](poisson_analysis/detailed/accuracy_vs_dataset_poisson60_k20.png)
*Performance vs dataset size under Poisson noise (λ=60) with k=20 features*

---

## 🎯 Feature Selection Analysis

### Clean Dataset Conditions

#### Mini Dataset
![Clean Mini](poisson_analysis/detailed/accuracy_vs_k_clean_mini.png)
*Feature selection impact on mini dataset (clean conditions)*

#### Small Dataset
![Clean Small](poisson_analysis/detailed/accuracy_vs_k_clean_small.png)
*Feature selection impact on small dataset (clean conditions)*

#### Original Dataset
![Clean Original](poisson_analysis/detailed/accuracy_vs_k_clean_original.png)
*Feature selection impact on original dataset (clean conditions)*

### Poisson Noise Conditions

#### Mini Dataset
![Poisson40 Mini](poisson_analysis/detailed/accuracy_vs_k_poisson40_mini.png)
*Feature selection impact on mini dataset under Poisson noise (λ=40)*

![Poisson60 Mini](poisson_analysis/detailed/accuracy_vs_k_poisson60_mini.png)
*Feature selection impact on mini dataset under Poisson noise (λ=60)*

#### Small Dataset
![Poisson40 Small](poisson_analysis/detailed/accuracy_vs_k_poisson40_small.png)
*Feature selection impact on small dataset under Poisson noise (λ=40)*

![Poisson60 Small](poisson_analysis/detailed/accuracy_vs_k_poisson60_small.png)
*Feature selection impact on small dataset under Poisson noise (λ=60)*

#### Original Dataset
![Poisson40 Original](poisson_analysis/detailed/accuracy_vs_k_poisson40_original.png)
*Feature selection impact on original dataset under Poisson noise (λ=40)*

![Poisson60 Original](poisson_analysis/detailed/accuracy_vs_k_poisson60_original.png)
*Feature selection impact on original dataset under Poisson noise (λ=60)*

---

## 🛡️ Noise Robustness Analysis

### Mini Dataset

#### K=2 Features
![Noise Mini K2](poisson_analysis/detailed/accuracy_vs_poisson_mini_k2.png)
*Poisson noise robustness on mini dataset with k=2 features*

#### K=5 Features
![Noise Mini K5](poisson_analysis/detailed/accuracy_vs_poisson_mini_k5.png)
*Poisson noise robustness on mini dataset with k=5 features*

#### K=10 Features
![Noise Mini K10](poisson_analysis/detailed/accuracy_vs_poisson_mini_k10.png)
*Poisson noise robustness on mini dataset with k=10 features*

#### K=20 Features
![Noise Mini K20](poisson_analysis/detailed/accuracy_vs_poisson_mini_k20.png)
*Poisson noise robustness on mini dataset with k=20 features*

### Small Dataset

#### K=2 Features
![Noise Small K2](poisson_analysis/detailed/accuracy_vs_poisson_small_k2.png)
*Poisson noise robustness on small dataset with k=2 features*

#### K=5 Features
![Noise Small K5](poisson_analysis/detailed/accuracy_vs_poisson_small_k5.png)
*Poisson noise robustness on small dataset with k=5 features*

#### K=10 Features
![Noise Small K10](poisson_analysis/detailed/accuracy_vs_poisson_small_k10.png)
*Poisson noise robustness on small dataset with k=10 features*

#### K=20 Features
![Noise Small K20](poisson_analysis/detailed/accuracy_vs_poisson_small_k20.png)
*Poisson noise robustness on small dataset with k=20 features*

### Original Dataset

#### K=2 Features
![Noise Original K2](poisson_analysis/detailed/accuracy_vs_poisson_original_k2.png)
*Poisson noise robustness on original dataset with k=2 features*

#### K=5 Features
![Noise Original K5](poisson_analysis/detailed/accuracy_vs_poisson_original_k5.png)
*Poisson noise robustness on original dataset with k=5 features*

#### K=10 Features
![Noise Original K10](poisson_analysis/detailed/accuracy_vs_poisson_original_k10.png)
*Poisson noise robustness on original dataset with k=10 features*

#### K=20 Features
![Noise Original K20](poisson_analysis/detailed/accuracy_vs_poisson_original_k20.png)
*Poisson noise robustness on original dataset with k=20 features*

---

## 📊 Summary Statistics

- **Noise Type**: Poisson noise (λ=40, λ=60)
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

*Generated from comprehensive analysis of Poisson noise Random Forest experiments comparing WST, Advanced Stats, and Hybrid feature extraction methods across different noise intensities (λ=40, λ=60) and dataset sizes.*