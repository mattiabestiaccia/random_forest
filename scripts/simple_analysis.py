#!/usr/bin/env python3
"""
RGB Random Forest Experiments Analysis - Simple Version
=======================================================

This script analyzes all experiments in the rgb_gaussian50_kbest directory
using only built-in Python modules.
"""

import os
import json
from pathlib import Path
import statistics
from collections import defaultdict, Counter

def extract_experiment_data(base_dir):
    """
    Extract performance data from all experiment JSON files
    
    Args:
        base_dir (str): Base directory containing experiment folders
        
    Returns:
        list: List of experiment result dictionaries
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
    
    print(f"\nTotal experiments processed: {len(results)}")
    return results

def analyze_by_category(results, category_key):
    """
    Analyze results by a specific category
    
    Args:
        results (list): List of experiment results
        category_key (str): Key to group by
        
    Returns:
        dict: Analysis results by category
    """
    category_data = defaultdict(list)
    
    for result in results:
        category_data[result[category_key]].append(result['mean_accuracy'])
    
    analysis = {}
    for category, accuracies in category_data.items():
        analysis[category] = {
            'count': len(accuracies),
            'mean': statistics.mean(accuracies),
            'stdev': statistics.stdev(accuracies) if len(accuracies) > 1 else 0,
            'min': min(accuracies),
            'max': max(accuracies),
            'median': statistics.median(accuracies)
        }
    
    return analysis

def find_best_configurations(results, top_n=10):
    """
    Find the best performing configurations
    
    Args:
        results (list): List of experiment results
        top_n (int): Number of top configurations to return
        
    Returns:
        list: Top N configurations sorted by accuracy
    """
    sorted_results = sorted(results, key=lambda x: x['mean_accuracy'], reverse=True)
    return sorted_results[:top_n]

def compare_feature_methods(results):
    """
    Compare different feature methods
    
    Args:
        results (list): List of experiment results
        
    Returns:
        dict: Comparison results
    """
    method_comparison = {}
    
    # Group by feature method
    methods = defaultdict(list)
    for result in results:
        methods[result['feature_method']].append(result['mean_accuracy'])
    
    # Calculate statistics for each method
    for method, accuracies in methods.items():
        method_comparison[method] = {
            'count': len(accuracies),
            'mean': statistics.mean(accuracies),
            'stdev': statistics.stdev(accuracies) if len(accuracies) > 1 else 0,
            'min': min(accuracies),
            'max': max(accuracies)
        }
    
    # Find improvements
    if 'hybrid' in method_comparison and 'wst' in method_comparison:
        hybrid_mean = method_comparison['hybrid']['mean']
        wst_mean = method_comparison['wst']['mean']
        hybrid_improvement = hybrid_mean - wst_mean
        hybrid_improvement_pct = (hybrid_improvement / wst_mean) * 100
        
        method_comparison['hybrid_vs_wst'] = {
            'improvement': hybrid_improvement,
            'improvement_pct': hybrid_improvement_pct
        }
    
    if 'hybrid' in method_comparison and 'advanced_stats' in method_comparison:
        hybrid_mean = method_comparison['hybrid']['mean']
        advanced_mean = method_comparison['advanced_stats']['mean']
        hybrid_improvement = hybrid_mean - advanced_mean
        hybrid_improvement_pct = (hybrid_improvement / advanced_mean) * 100
        
        method_comparison['hybrid_vs_advanced'] = {
            'improvement': hybrid_improvement,
            'improvement_pct': hybrid_improvement_pct
        }
    
    return method_comparison

def analyze_k_value_trends(results):
    """
    Analyze trends across different K values
    
    Args:
        results (list): List of experiment results
        
    Returns:
        dict: K value analysis
    """
    k_analysis = {}
    
    # Group by K value
    k_data = defaultdict(list)
    for result in results:
        k_data[result['k_value']].append(result['mean_accuracy'])
    
    # Calculate statistics for each K value
    for k_value, accuracies in k_data.items():
        k_analysis[k_value] = {
            'count': len(accuracies),
            'mean': statistics.mean(accuracies),
            'stdev': statistics.stdev(accuracies) if len(accuracies) > 1 else 0,
            'min': min(accuracies),
            'max': max(accuracies)
        }
    
    # Find optimal K
    best_k = max(k_analysis.keys(), key=lambda k: k_analysis[k]['mean'])
    k_analysis['optimal_k'] = best_k
    
    return k_analysis

def generate_detailed_report(results, output_dir):
    """
    Generate a comprehensive text report
    
    Args:
        results (list): List of experiment results
        output_dir (str): Directory to save report
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'detailed_performance_report.txt')
    
    # Perform all analyses
    feature_analysis = analyze_by_category(results, 'feature_method')
    dataset_analysis = analyze_by_category(results, 'dataset_type')
    area_analysis = analyze_by_category(results, 'area')
    k_analysis = analyze_k_value_trends(results)
    feature_comparison = compare_feature_methods(results)
    top_configs = find_best_configurations(results, 10)
    
    # Overall statistics
    all_accuracies = [r['mean_accuracy'] for r in results]
    overall_stats = {
        'count': len(all_accuracies),
        'mean': statistics.mean(all_accuracies),
        'stdev': statistics.stdev(all_accuracies) if len(all_accuracies) > 1 else 0,
        'min': min(all_accuracies),
        'max': max(all_accuracies),
        'median': statistics.median(all_accuracies)
    }
    
    with open(report_path, 'w') as f:
        f.write("RGB RANDOM FOREST EXPERIMENTS - DETAILED PERFORMANCE ANALYSIS\n")
        f.write("=" * 65 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 17 + "\n")
        f.write(f"Total Experiments: {len(results)}\n")
        f.write(f"Feature Methods: {len(feature_analysis)}\n")
        f.write(f"Dataset Types: {len(dataset_analysis)}\n")
        f.write(f"Areas: {len(area_analysis)}\n")
        f.write(f"K Values: {sorted(k_analysis.keys())[:-1]}\n")  # Exclude 'optimal_k' key
        f.write(f"Overall Accuracy Range: {overall_stats['min']:.4f} - {overall_stats['max']:.4f}\n")
        f.write(f"Average Accuracy: {overall_stats['mean']:.4f} ± {overall_stats['stdev']:.4f}\n\n")
        
        # Feature Method Analysis
        f.write("FEATURE METHOD ANALYSIS\n")
        f.write("-" * 24 + "\n")
        for method, stats in sorted(feature_analysis.items()):
            f.write(f"{method.upper()}:\n")
            f.write(f"  Count: {stats['count']}\n")
            f.write(f"  Mean: {stats['mean']:.4f}\n")
            f.write(f"  Std Dev: {stats['stdev']:.4f}\n")
            f.write(f"  Range: {stats['min']:.4f} - {stats['max']:.4f}\n")
            f.write(f"  Median: {stats['median']:.4f}\n\n")
        
        # Feature Method Comparison
        f.write("FEATURE METHOD COMPARISON\n")
        f.write("-" * 26 + "\n")
        
        # Rank methods by performance
        method_ranking = sorted(feature_analysis.items(), key=lambda x: x[1]['mean'], reverse=True)
        f.write("Performance Ranking:\n")
        for i, (method, stats) in enumerate(method_ranking, 1):
            f.write(f"  {i}. {method.upper()}: {stats['mean']:.4f}\n")
        f.write("\n")
        
        # Improvements analysis
        if 'hybrid_vs_wst' in feature_comparison:
            improvement = feature_comparison['hybrid_vs_wst']
            f.write(f"HYBRID vs WST:\n")
            f.write(f"  Improvement: +{improvement['improvement']:.4f} ({improvement['improvement_pct']:+.2f}%)\n")
        
        if 'hybrid_vs_advanced' in feature_comparison:
            improvement = feature_comparison['hybrid_vs_advanced']
            f.write(f"HYBRID vs ADVANCED_STATS:\n")
            f.write(f"  Improvement: +{improvement['improvement']:.4f} ({improvement['improvement_pct']:+.2f}%)\n")
        f.write("\n")
        
        # K Value Analysis
        f.write("K VALUE ANALYSIS\n")
        f.write("-" * 17 + "\n")
        k_sorted = sorted([(k, stats) for k, stats in k_analysis.items() if k != 'optimal_k'])
        for k_value, stats in k_sorted:
            f.write(f"K={k_value}:\n")
            f.write(f"  Count: {stats['count']}\n")
            f.write(f"  Mean: {stats['mean']:.4f}\n")
            f.write(f"  Std Dev: {stats['stdev']:.4f}\n")
            f.write(f"  Range: {stats['min']:.4f} - {stats['max']:.4f}\n\n")
        
        f.write(f"OPTIMAL K VALUE: {k_analysis['optimal_k']}\n")
        f.write(f"Best K Performance: {k_analysis[k_analysis['optimal_k']]['mean']:.4f}\n\n")
        
        # Dataset Type Analysis
        f.write("DATASET TYPE ANALYSIS\n")
        f.write("-" * 22 + "\n")
        dataset_ranking = sorted(dataset_analysis.items(), key=lambda x: x[1]['mean'], reverse=True)
        for dataset, stats in dataset_ranking:
            f.write(f"{dataset.upper()}:\n")
            f.write(f"  Count: {stats['count']}\n")
            f.write(f"  Mean: {stats['mean']:.4f}\n")
            f.write(f"  Std Dev: {stats['stdev']:.4f}\n")
            f.write(f"  Range: {stats['min']:.4f} - {stats['max']:.4f}\n\n")
        
        # Area Analysis
        f.write("AREA ANALYSIS\n")
        f.write("-" * 14 + "\n")
        area_ranking = sorted(area_analysis.items(), key=lambda x: x[1]['mean'], reverse=True)
        for area, stats in area_ranking:
            f.write(f"{area.upper()}:\n")
            f.write(f"  Count: {stats['count']}\n")
            f.write(f"  Mean: {stats['mean']:.4f}\n")
            f.write(f"  Std Dev: {stats['stdev']:.4f}\n")
            f.write(f"  Range: {stats['min']:.4f} - {stats['max']:.4f}\n\n")
        
        # Top Configurations
        f.write("TOP 10 PERFORMING CONFIGURATIONS\n")
        f.write("-" * 34 + "\n")
        for i, config in enumerate(top_configs, 1):
            f.write(f"{i:2d}. {config['feature_method'].upper()}, ")
            f.write(f"{config['dataset_type'].upper()}, ")
            f.write(f"K={config['k_value']}, ")
            f.write(f"{config['area'].upper()}: ")
            f.write(f"{config['mean_accuracy']:.4f} ± {config['std_accuracy']:.4f}\n")
        f.write("\n")
        
        # Key Insights
        f.write("KEY INSIGHTS\n")
        f.write("-" * 12 + "\n")
        
        best_method = method_ranking[0][0]
        best_dataset = dataset_ranking[0][0]
        best_area = area_ranking[0][0]
        best_k = k_analysis['optimal_k']
        
        f.write(f"1. BEST FEATURE METHOD: {best_method.upper()}\n")
        f.write(f"   - Average accuracy: {feature_analysis[best_method]['mean']:.4f}\n")
        f.write(f"   - Consistency: {feature_analysis[best_method]['stdev']:.4f} std dev\n\n")
        
        f.write(f"2. BEST DATASET TYPE: {best_dataset.upper()}\n")
        f.write(f"   - Average accuracy: {dataset_analysis[best_dataset]['mean']:.4f}\n\n")
        
        f.write(f"3. BEST AREA: {best_area.upper()}\n")
        f.write(f"   - Average accuracy: {area_analysis[best_area]['mean']:.4f}\n\n")
        
        f.write(f"4. OPTIMAL K VALUE: {best_k}\n")
        f.write(f"   - Average accuracy: {k_analysis[best_k]['mean']:.4f}\n\n")
        
        f.write("5. FEATURE METHOD ADVANTAGES:\n")
        if best_method == 'hybrid':
            f.write("   - Hybrid approach combines the best of both worlds\n")
            f.write("   - Provides highest overall accuracy\n")
            f.write("   - Suitable for applications requiring maximum performance\n")
        elif best_method == 'wst':
            f.write("   - WST provides good performance with computational efficiency\n")
            f.write("   - Captures spatial patterns effectively\n")
        else:
            f.write("   - Advanced stats provide interpretable features\n")
            f.write("   - Good baseline performance\n")
        
        f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 15 + "\n")
        f.write("1. For MAXIMUM PERFORMANCE:\n")
        f.write(f"   - Use {best_method.upper()} feature method\n")
        f.write(f"   - Use {best_dataset.upper()} dataset\n")
        f.write(f"   - Set K={best_k} for feature selection\n")
        f.write(f"   - Focus on {best_area.upper()} area if domain-specific\n\n")
        
        f.write("2. For BALANCED PERFORMANCE:\n")
        if 'hybrid' in feature_analysis:
            f.write("   - Consider HYBRID approach for comprehensive features\n")
        f.write("   - Use K=10 or K=20 for good feature/performance balance\n")
        f.write("   - Test across multiple areas for generalization\n\n")
        
        f.write("3. For COMPUTATIONAL EFFICIENCY:\n")
        if 'wst' in feature_analysis:
            f.write("   - Use WST for faster processing\n")
        f.write("   - Consider lower K values (K=5) for reduced complexity\n")
        f.write("   - Use MINI or SMALL datasets for faster training\n\n")
        
        # Statistical Significance
        f.write("STATISTICAL ANALYSIS\n")
        f.write("-" * 19 + "\n")
        f.write(f"Total variance in accuracy: {overall_stats['stdev']:.4f}\n")
        f.write(f"Coefficient of variation: {overall_stats['stdev']/overall_stats['mean']*100:.2f}%\n")
        
        # Method variance comparison
        method_vars = [(method, stats['stdev']) for method, stats in feature_analysis.items()]
        method_vars.sort(key=lambda x: x[1])
        f.write(f"Most consistent method: {method_vars[0][0].upper()} (std: {method_vars[0][1]:.4f})\n")
        f.write(f"Most variable method: {method_vars[-1][0].upper()} (std: {method_vars[-1][1]:.4f})\n")
        
        f.write(f"\nReport generated and saved to: {report_path}\n")
    
    print(f"Detailed report generated: {report_path}")
    return report_path

def save_csv_data(results, output_dir):
    """
    Save results to CSV format
    
    Args:
        results (list): List of experiment results
        output_dir (str): Directory to save CSV
    """
    csv_path = os.path.join(output_dir, 'experiment_results.csv')
    
    with open(csv_path, 'w') as f:
        # Write header
        f.write("Feature_Method,Dataset_Type,K_Value,Area,Mean_Accuracy,Std_Accuracy,Total_Images,Total_Features,Experiment_Name\n")
        
        # Write data
        for result in results:
            f.write(f"{result['feature_method']},{result['dataset_type']},{result['k_value']},")
            f.write(f"{result['area']},{result['mean_accuracy']:.6f},{result['std_accuracy']:.6f},")
            f.write(f"{result['total_images']},{result['total_features']},{result['experiment_name']}\n")
    
    print(f"CSV data saved to: {csv_path}")

def main():
    """
    Main analysis function
    """
    base_dir = "/home/brusc/Projects/random_forest/experiments/rgb_gaussian50_kbest"
    output_dir = "/home/brusc/Projects/random_forest/experiments/analysis_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting RGB Random Forest Experiments Analysis...")
    print("=" * 55)
    
    # Extract experiment data
    results = extract_experiment_data(base_dir)
    
    if len(results) == 0:
        print("No experiment data found!")
        return
    
    # Generate detailed report
    report_path = generate_detailed_report(results, output_dir)
    
    # Save CSV data
    save_csv_data(results, output_dir)
    
    print("\nAnalysis completed successfully!")
    print(f"Results saved to: {output_dir}")
    
    # Print quick summary
    print("\nQUICK SUMMARY:")
    print("-" * 14)
    
    # Best overall configuration
    best_config = max(results, key=lambda x: x['mean_accuracy'])
    print(f"Best Configuration: {best_config['feature_method'].upper()}, {best_config['dataset_type'].upper()}, K={best_config['k_value']}, {best_config['area'].upper()}")
    print(f"Best Accuracy: {best_config['mean_accuracy']:.4f} ± {best_config['std_accuracy']:.4f}")
    
    # Method comparison
    methods = {}
    for result in results:
        method = result['feature_method']
        if method not in methods:
            methods[method] = []
        methods[method].append(result['mean_accuracy'])
    
    print("\nMethod Averages:")
    for method, accuracies in sorted(methods.items()):
        avg_acc = statistics.mean(accuracies)
        print(f"  {method.upper()}: {avg_acc:.4f}")

if __name__ == "__main__":
    main()