#!/usr/bin/env python3
"""
RGB Random Forest Experiments Analysis
======================================

This script analyzes all experiments in the rgb_gaussian50_kbest directory
and creates a comprehensive performance comparison report with visualizations.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def extract_experiment_data(base_dir):
    """
    Extract performance data from all experiment JSON files
    
    Args:
        base_dir (str): Base directory containing experiment folders
        
    Returns:
        pd.DataFrame: DataFrame with experiment results
    """
    results = []
    base_path = Path(base_dir)
    
    print(f"Analyzing experiments in: {base_dir}")
    
    # Iterate through all experiment directories
    for exp_dir in base_path.iterdir():
        if exp_dir.is_dir():
            # Parse directory name to extract parameters
            dir_name = exp_dir.name
            parts = dir_name.split('_')
            
            if len(parts) >= 4:
                feature_method = parts[0]
                dataset_type = parts[1]
                k_value = parts[2].replace('k', '')
                area = parts[3]
                
                # Find the JSON report file
                json_files = list(exp_dir.glob('*.json'))
                if json_files:
                    json_file = json_files[0]  # Take the first JSON file
                    
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        
                        # Extract performance metrics
                        performance = data.get('performance', {})
                        mean_accuracy = performance.get('mean_accuracy', 0)
                        std_accuracy = performance.get('std_accuracy', 0)
                        
                        # Extract additional info
                        dataset_info = data.get('dataset_info', {})
                        total_images = dataset_info.get('total_images', 0)
                        total_features = dataset_info.get('total_features_available', 0)
                        
                        results.append({
                            'feature_method': feature_method,
                            'dataset_type': dataset_type,
                            'k_value': int(k_value),
                            'area': area,
                            'mean_accuracy': mean_accuracy,
                            'std_accuracy': std_accuracy,
                            'total_images': total_images,
                            'total_features': total_features,
                            'experiment_name': dir_name
                        })
                        
                        print(f"✓ Processed: {dir_name}")
                        
                    except Exception as e:
                        print(f"✗ Error processing {json_file}: {e}")
                else:
                    print(f"✗ No JSON file found in {exp_dir}")
    
    df = pd.DataFrame(results)
    print(f"\nTotal experiments processed: {len(df)}")
    print(f"Feature methods: {df['feature_method'].unique()}")
    print(f"Dataset types: {df['dataset_type'].unique()}")
    print(f"K values: {sorted(df['k_value'].unique())}")
    print(f"Areas: {df['area'].unique()}")
    
    return df

def create_summary_statistics(df):
    """
    Create comprehensive summary statistics
    
    Args:
        df (pd.DataFrame): DataFrame with experiment results
        
    Returns:
        dict: Summary statistics
    """
    summary = {}
    
    # Overall statistics
    summary['overall'] = {
        'total_experiments': len(df),
        'mean_accuracy': df['mean_accuracy'].mean(),
        'std_accuracy': df['mean_accuracy'].std(),
        'min_accuracy': df['mean_accuracy'].min(),
        'max_accuracy': df['mean_accuracy'].max()
    }
    
    # Statistics by feature method
    summary['by_feature_method'] = df.groupby('feature_method')['mean_accuracy'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(4)
    
    # Statistics by dataset type
    summary['by_dataset_type'] = df.groupby('dataset_type')['mean_accuracy'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(4)
    
    # Statistics by K value
    summary['by_k_value'] = df.groupby('k_value')['mean_accuracy'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(4)
    
    # Statistics by area
    summary['by_area'] = df.groupby('area')['mean_accuracy'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(4)
    
    # Cross-tabulation analysis
    summary['feature_method_vs_k'] = df.groupby(['feature_method', 'k_value'])['mean_accuracy'].mean().unstack()
    summary['feature_method_vs_dataset'] = df.groupby(['feature_method', 'dataset_type'])['mean_accuracy'].mean().unstack()
    
    return summary

def create_visualizations(df, output_dir):
    """
    Create comprehensive visualizations
    
    Args:
        df (pd.DataFrame): DataFrame with experiment results
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    
    # 1. Feature Method Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Performance Analysis by Feature Method', fontsize=16, fontweight='bold')
    
    # 1a. Box plot by feature method
    sns.boxplot(data=df, x='feature_method', y='mean_accuracy', ax=axes[0,0])
    axes[0,0].set_title('Accuracy Distribution by Feature Method')
    axes[0,0].set_ylabel('Mean Accuracy')
    axes[0,0].set_xlabel('Feature Method')
    
    # 1b. Performance by K value for each feature method
    for method in df['feature_method'].unique():
        method_data = df[df['feature_method'] == method]
        k_means = method_data.groupby('k_value')['mean_accuracy'].mean()
        k_stds = method_data.groupby('k_value')['mean_accuracy'].std()
        axes[0,1].errorbar(k_means.index, k_means.values, yerr=k_stds.values, 
                          label=method, marker='o', capsize=5)
    
    axes[0,1].set_title('Accuracy vs K Value by Feature Method')
    axes[0,1].set_xlabel('K Value')
    axes[0,1].set_ylabel('Mean Accuracy')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 1c. Performance by dataset type
    sns.boxplot(data=df, x='dataset_type', y='mean_accuracy', hue='feature_method', ax=axes[1,0])
    axes[1,0].set_title('Accuracy by Dataset Type and Feature Method')
    axes[1,0].set_xlabel('Dataset Type')
    axes[1,0].set_ylabel('Mean Accuracy')
    
    # 1d. Performance by area
    sns.boxplot(data=df, x='area', y='mean_accuracy', hue='feature_method', ax=axes[1,1])
    axes[1,1].set_title('Accuracy by Area and Feature Method')
    axes[1,1].set_xlabel('Area')
    axes[1,1].set_ylabel('Mean Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_method_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. K Value Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Performance Analysis by K Value', fontsize=16, fontweight='bold')
    
    # 2a. Overall K value performance
    k_stats = df.groupby('k_value')['mean_accuracy'].agg(['mean', 'std'])
    axes[0,0].errorbar(k_stats.index, k_stats['mean'], yerr=k_stats['std'], 
                      marker='o', capsize=5, linewidth=2)
    axes[0,0].set_title('Overall Accuracy vs K Value')
    axes[0,0].set_xlabel('K Value')
    axes[0,0].set_ylabel('Mean Accuracy')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2b. K value performance by dataset type
    for dataset in df['dataset_type'].unique():
        dataset_data = df[df['dataset_type'] == dataset]
        k_means = dataset_data.groupby('k_value')['mean_accuracy'].mean()
        axes[0,1].plot(k_means.index, k_means.values, marker='o', label=dataset)
    
    axes[0,1].set_title('Accuracy vs K Value by Dataset Type')
    axes[0,1].set_xlabel('K Value')
    axes[0,1].set_ylabel('Mean Accuracy')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 2c. Heatmap of K value vs feature method
    pivot_k_feature = df.groupby(['k_value', 'feature_method'])['mean_accuracy'].mean().unstack()
    sns.heatmap(pivot_k_feature, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1,0])
    axes[1,0].set_title('Accuracy Heatmap: K Value vs Feature Method')
    axes[1,0].set_xlabel('Feature Method')
    axes[1,0].set_ylabel('K Value')
    
    # 2d. Standard deviation by K value
    k_std_stats = df.groupby('k_value')['std_accuracy'].mean()
    axes[1,1].bar(k_std_stats.index, k_std_stats.values, alpha=0.7)
    axes[1,1].set_title('Model Stability (Std Dev) by K Value')
    axes[1,1].set_xlabel('K Value')
    axes[1,1].set_ylabel('Mean Standard Deviation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'k_value_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Dataset Type Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Performance Analysis by Dataset Type', fontsize=16, fontweight='bold')
    
    # 3a. Overall dataset performance
    dataset_stats = df.groupby('dataset_type')['mean_accuracy'].agg(['mean', 'std'])
    axes[0,0].bar(dataset_stats.index, dataset_stats['mean'], yerr=dataset_stats['std'], 
                  capsize=5, alpha=0.7)
    axes[0,0].set_title('Overall Accuracy by Dataset Type')
    axes[0,0].set_xlabel('Dataset Type')
    axes[0,0].set_ylabel('Mean Accuracy')
    
    # 3b. Dataset vs Area performance
    pivot_dataset_area = df.groupby(['dataset_type', 'area'])['mean_accuracy'].mean().unstack()
    sns.heatmap(pivot_dataset_area, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0,1])
    axes[0,1].set_title('Accuracy Heatmap: Dataset Type vs Area')
    axes[0,1].set_xlabel('Area')
    axes[0,1].set_ylabel('Dataset Type')
    
    # 3c. Dataset type performance trends
    for area in df['area'].unique():
        area_data = df[df['area'] == area]
        dataset_means = area_data.groupby('dataset_type')['mean_accuracy'].mean()
        axes[1,0].plot(dataset_means.index, dataset_means.values, marker='o', label=area)
    
    axes[1,0].set_title('Dataset Performance by Area')
    axes[1,0].set_xlabel('Dataset Type')
    axes[1,0].set_ylabel('Mean Accuracy')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 3d. Sample size vs performance
    sample_perf = df.groupby('dataset_type')[['total_images', 'mean_accuracy']].mean()
    axes[1,1].scatter(sample_perf['total_images'], sample_perf['mean_accuracy'], s=100)
    for i, dataset in enumerate(sample_perf.index):
        axes[1,1].annotate(dataset, (sample_perf.iloc[i]['total_images'], 
                                    sample_perf.iloc[i]['mean_accuracy']), 
                          xytext=(5, 5), textcoords='offset points')
    axes[1,1].set_title('Sample Size vs Performance')
    axes[1,1].set_xlabel('Total Images')
    axes[1,1].set_ylabel('Mean Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Comprehensive Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comprehensive Performance Comparison', fontsize=16, fontweight='bold')
    
    # 4a. Overall performance ranking
    overall_ranking = df.groupby('feature_method')['mean_accuracy'].mean().sort_values(ascending=False)
    axes[0,0].bar(overall_ranking.index, overall_ranking.values, alpha=0.7)
    axes[0,0].set_title('Overall Feature Method Ranking')
    axes[0,0].set_xlabel('Feature Method')
    axes[0,0].set_ylabel('Mean Accuracy')
    
    # 4b. Performance improvement with more features
    feature_perf = df.groupby(['feature_method', 'k_value'])['mean_accuracy'].mean().unstack()
    improvement = feature_perf.iloc[:, -1] - feature_perf.iloc[:, 0]  # k=20 - k=2
    axes[0,1].bar(improvement.index, improvement.values, alpha=0.7)
    axes[0,1].set_title('Accuracy Improvement (K=20 vs K=2)')
    axes[0,1].set_xlabel('Feature Method')
    axes[0,1].set_ylabel('Accuracy Improvement')
    
    # 4c. Consistency analysis (lower std is better)
    consistency = df.groupby('feature_method')['std_accuracy'].mean()
    axes[1,0].bar(consistency.index, consistency.values, alpha=0.7)
    axes[1,0].set_title('Model Consistency by Feature Method')
    axes[1,0].set_xlabel('Feature Method')
    axes[1,0].set_ylabel('Mean Standard Deviation')
    
    # 4d. Best configuration for each method
    best_configs = df.loc[df.groupby('feature_method')['mean_accuracy'].idxmax()]
    x_pos = np.arange(len(best_configs))
    bars = axes[1,1].bar(x_pos, best_configs['mean_accuracy'], alpha=0.7)
    axes[1,1].set_title('Best Configuration for Each Feature Method')
    axes[1,1].set_xlabel('Feature Method')
    axes[1,1].set_ylabel('Best Accuracy')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels(best_configs['feature_method'])
    
    # Add configuration details on bars
    for i, (bar, config) in enumerate(zip(bars, best_configs.iterrows())):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                      f'{config[1]["dataset_type"]}\nK={config[1]["k_value"]}\n{config[1]["area"]}',
                      ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All visualizations saved to: {output_dir}")

def generate_report(df, summary, output_dir):
    """
    Generate a comprehensive text report
    
    Args:
        df (pd.DataFrame): DataFrame with experiment results
        summary (dict): Summary statistics
        output_dir (str): Directory to save report
    """
    report_path = os.path.join(output_dir, 'performance_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("RGB RANDOM FOREST EXPERIMENTS - PERFORMANCE ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Experiments Analyzed: {len(df)}\n\n")
        
        # Overall Statistics
        f.write("OVERALL PERFORMANCE STATISTICS\n")
        f.write("-" * 35 + "\n")
        f.write(f"Mean Accuracy: {summary['overall']['mean_accuracy']:.4f}\n")
        f.write(f"Standard Deviation: {summary['overall']['std_accuracy']:.4f}\n")
        f.write(f"Min Accuracy: {summary['overall']['min_accuracy']:.4f}\n")
        f.write(f"Max Accuracy: {summary['overall']['max_accuracy']:.4f}\n\n")
        
        # Feature Method Analysis
        f.write("PERFORMANCE BY FEATURE METHOD\n")
        f.write("-" * 32 + "\n")
        f.write(summary['by_feature_method'].to_string())
        f.write("\n\n")
        
        # Find best performing method
        best_method = summary['by_feature_method']['mean'].idxmax()
        best_accuracy = summary['by_feature_method']['mean'].max()
        f.write(f"BEST PERFORMING FEATURE METHOD: {best_method.upper()}\n")
        f.write(f"Average Accuracy: {best_accuracy:.4f}\n\n")
        
        # Dataset Type Analysis
        f.write("PERFORMANCE BY DATASET TYPE\n")
        f.write("-" * 28 + "\n")
        f.write(summary['by_dataset_type'].to_string())
        f.write("\n\n")
        
        # K Value Analysis
        f.write("PERFORMANCE BY K VALUE\n")
        f.write("-" * 23 + "\n")
        f.write(summary['by_k_value'].to_string())
        f.write("\n\n")
        
        # Area Analysis
        f.write("PERFORMANCE BY AREA\n")
        f.write("-" * 20 + "\n")
        f.write(summary['by_area'].to_string())
        f.write("\n\n")
        
        # Cross-tabulation Analysis
        f.write("FEATURE METHOD vs K VALUE ANALYSIS\n")
        f.write("-" * 36 + "\n")
        f.write(summary['feature_method_vs_k'].to_string())
        f.write("\n\n")
        
        f.write("FEATURE METHOD vs DATASET TYPE ANALYSIS\n")
        f.write("-" * 41 + "\n")
        f.write(summary['feature_method_vs_dataset'].to_string())
        f.write("\n\n")
        
        # Top Performing Configurations
        f.write("TOP 10 PERFORMING CONFIGURATIONS\n")
        f.write("-" * 34 + "\n")
        top_configs = df.nlargest(10, 'mean_accuracy')[['feature_method', 'dataset_type', 'k_value', 'area', 'mean_accuracy', 'std_accuracy']]
        f.write(top_configs.to_string(index=False))
        f.write("\n\n")
        
        # WST vs Hybrid Analysis
        f.write("WST vs HYBRID APPROACH ANALYSIS\n")
        f.write("-" * 33 + "\n")
        
        wst_stats = df[df['feature_method'] == 'wst']['mean_accuracy']
        hybrid_stats = df[df['feature_method'] == 'hybrid']['mean_accuracy']
        advanced_stats = df[df['feature_method'] == 'advanced_stats']['mean_accuracy']
        
        f.write(f"WST Average Accuracy: {wst_stats.mean():.4f} ± {wst_stats.std():.4f}\n")
        f.write(f"Hybrid Average Accuracy: {hybrid_stats.mean():.4f} ± {hybrid_stats.std():.4f}\n")
        f.write(f"Advanced Stats Average Accuracy: {advanced_stats.mean():.4f} ± {advanced_stats.std():.4f}\n\n")
        
        # Performance improvement analysis
        hybrid_improvement = hybrid_stats.mean() - wst_stats.mean()
        f.write(f"Hybrid vs WST Improvement: {hybrid_improvement:.4f} ({hybrid_improvement/wst_stats.mean()*100:.2f}%)\n")
        
        hybrid_vs_advanced = hybrid_stats.mean() - advanced_stats.mean()
        f.write(f"Hybrid vs Advanced Stats Improvement: {hybrid_vs_advanced:.4f} ({hybrid_vs_advanced/advanced_stats.mean()*100:.2f}%)\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 15 + "\n")
        f.write("1. Feature Method: ")
        if best_method == 'hybrid':
            f.write("Use HYBRID approach for best overall performance\n")
        elif best_method == 'wst':
            f.write("Use WST approach for good performance with computational efficiency\n")
        else:
            f.write("Use ADVANCED_STATS approach for interpretable features\n")
        
        # Best K value
        best_k = summary['by_k_value']['mean'].idxmax()
        f.write(f"2. K Value: Use K={best_k} for optimal feature selection\n")
        
        # Best dataset
        best_dataset = summary['by_dataset_type']['mean'].idxmax()
        f.write(f"3. Dataset: {best_dataset.upper()} dataset provides best performance\n")
        
        # Best area
        best_area = summary['by_area']['mean'].idxmax()
        f.write(f"4. Area: {best_area.upper()} area shows highest accuracy\n")
        
        f.write(f"\nReport saved to: {report_path}\n")
    
    print(f"Comprehensive report generated: {report_path}")

def main():
    """
    Main analysis function
    """
    base_dir = "/home/brusc/Projects/random_forest/experiments/rgb_gaussian50_kbest"
    output_dir = "/home/brusc/Projects/random_forest/experiments/analysis_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting RGB Random Forest Experiments Analysis...")
    print("=" * 50)
    
    # Extract experiment data
    df = extract_experiment_data(base_dir)
    
    if len(df) == 0:
        print("No experiment data found!")
        return
    
    # Create summary statistics
    summary = create_summary_statistics(df)
    
    # Generate visualizations
    create_visualizations(df, output_dir)
    
    # Generate comprehensive report
    generate_report(df, summary, output_dir)
    
    # Save processed data
    df.to_csv(os.path.join(output_dir, 'experiment_data.csv'), index=False)
    print(f"Processed data saved to: {os.path.join(output_dir, 'experiment_data.csv')}")
    
    print("\nAnalysis completed successfully!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()