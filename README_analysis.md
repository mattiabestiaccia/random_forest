# Random Forest Experiments Analysis

## Overview

This repository contains a comprehensive analysis of Random Forest classification experiments comparing different feature extraction methods under various noise conditions and dataset sizes.

## Quick Start

To run the complete analysis:

```bash
# Activate virtual environment
source venv_rf/bin/activate

# Run complete analysis
python create_complete_english_analysis.py
```

This will generate all reports, plots, and data exports in the `analysis_english/` directory.

## Generated Outputs

### Reports
- `analysis_english/comprehensive_report.md`: Complete statistical analysis
- `analysis_english/qualitative_analysis.md`: In-depth qualitative interpretation
- `analysis_english/analysis_summary.md`: Executive summary with key findings

### Data
- `analysis_english/experiments_summary_english.csv`: Complete dataset export

### Visualizations
- `analysis_english/comparisons/`: High-level comparison plots (4 plots)
- `analysis_english/detailed/`: Detailed analysis plots (33+ plots)

## Key Findings

- **Best performing method**: WST (0.913 average accuracy)
- **Noise impact**: 11.4% performance loss (Clean → Gaussian σ=50)  
- **Dataset size effect**: 7.4% improvement (Mini → Original)
- **WST most robust**: Lowest degradation under noise conditions

## Experimental Setup

- **Total experiments**: 324
- **Feature extraction methods**: WST, Advanced RGB Statistics, Hybrid
- **Noise conditions**: Clean, Gaussian σ=30, Gaussian σ=50
- **Dataset sizes**: Mini, Small, Original
- **Geographic areas**: Assatigue, Popolar, Sunset
- **Feature selection**: k ∈ {2, 5, 10, 20}

## Dependencies

```bash
pip install pandas matplotlib seaborn numpy
```

Or use the existing virtual environment:
```bash
source venv_rf/bin/activate
```

## Script Features

The main analysis script (`create_complete_english_analysis.py`) provides:

- **Unified analysis**: Single script for all analysis tasks
- **Professional output**: Publication-ready reports and plots
- **Comprehensive coverage**: Statistical analysis, visualizations, and qualitative interpretation
- **Flexible averaging**: Comparison plots (averaged over multiple dimensions) and detailed plots (averaged only over geographic areas)
- **Clean formatting**: Properly structured markdown reports

## Usage

```python
from create_complete_english_analysis import EnglishExperimentAnalyzer

analyzer = EnglishExperimentAnalyzer()
analyzer.create_complete_analysis(output_dir="my_analysis")
```

## File Structure

```
random_forest/
├── create_complete_english_analysis.py  # Main analysis script
├── analysis_english/                    # Generated analysis outputs
│   ├── comprehensive_report.md
│   ├── qualitative_analysis.md
│   ├── analysis_summary.md
│   ├── experiments_summary_english.csv
│   ├── comparisons/                     # High-level plots
│   └── detailed/                        # Detailed plots
├── experiments_organized/               # Raw experiment data
├── datasets/                           # Image datasets
└── scripts/                           # Training and inference scripts
```