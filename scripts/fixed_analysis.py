#!/usr/bin/env python3
"""
RGB Random Forest Experiments Analysis - Fixed Version
======================================================

This script analyzes all experiments in the rgb_gaussian50_kbest directory
using data directly from JSON files.
"""

import os
import json
from pathlib import Path
import statistics
from collections import defaultdict

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
            # Find the JSON report file
            json_files = list(exp_dir.glob('*.json'))
            if json_files:
                json_file = json_files[0]  # Take the first JSON file
                
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract data directly from JSON
                    feature_method = data.get('feature_method', 'unknown')
                    dataset_type = data.get('dataset_type', 'unknown')
                    k_features = data.get('k_features', 0)
                    area_name = data.get('area_name', 'unknown')
                    
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
                        'k_value': k_features,
                        'area': area_name,
                        'mean_accuracy': mean_accuracy,
                        'std_accuracy': std_accuracy,
                        'total_images': total_images,
                        'total_features': total_features,
                        'experiment_name': exp_dir.name
                    })
                    
                    print(f"✓ Processed: {exp_dir.name}")
                    
                except Exception as e:
                    print(f"✗ Error processing {json_file}: {e}")
            else:
                print(f"✗ No JSON file found in {exp_dir}")
    
    print(f"\nTotal experiments processed: {len(results)}")
    if results:
        feature_methods = set(r['feature_method'] for r in results)
        dataset_types = set(r['dataset_type'] for r in results)
        k_values = set(r['k_value'] for r in results)
        areas = set(r['area'] for r in results)
        
        print(f"Feature methods: {sorted(feature_methods)}")
        print(f"Dataset types: {sorted(dataset_types)}")
        print(f"K values: {sorted(k_values)}")
        print(f"Areas: {sorted(areas)}")
    
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
    if k_analysis:
        best_k = max(k_analysis.keys(), key=lambda k: k_analysis[k]['mean'])
        k_analysis['optimal_k'] = best_k
    
    return k_analysis

def create_performance_matrix(results):
    """
    Create a performance matrix for different parameter combinations
    
    Args:
        results (list): List of experiment results
        
    Returns:
        dict: Performance matrices
    """
    matrices = {}
    
    # Method vs K value matrix
    method_k_matrix = defaultdict(dict)
    for result in results:
        method = result['feature_method']
        k_value = result['k_value']
        
        if k_value not in method_k_matrix[method]:
            method_k_matrix[method][k_value] = []
        method_k_matrix[method][k_value].append(result['mean_accuracy'])
    
    # Calculate averages
    for method in method_k_matrix:
        for k_value in method_k_matrix[method]:
            accuracies = method_k_matrix[method][k_value]
            method_k_matrix[method][k_value] = statistics.mean(accuracies)
    
    matrices['method_vs_k'] = dict(method_k_matrix)
    
    # Method vs Dataset matrix
    method_dataset_matrix = defaultdict(dict)
    for result in results:
        method = result['feature_method']
        dataset = result['dataset_type']
        
        if dataset not in method_dataset_matrix[method]:
            method_dataset_matrix[method][dataset] = []
        method_dataset_matrix[method][dataset].append(result['mean_accuracy'])
    
    # Calculate averages
    for method in method_dataset_matrix:
        for dataset in method_dataset_matrix[method]:
            accuracies = method_dataset_matrix[method][dataset]
            method_dataset_matrix[method][dataset] = statistics.mean(accuracies)
    
    matrices['method_vs_dataset'] = dict(method_dataset_matrix)
    
    return matrices

def generate_comprehensive_report(results, output_dir):
    """
    Generate a comprehensive analysis report
    
    Args:
        results (list): List of experiment results
        output_dir (str): Directory to save report
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'comprehensive_analysis_report.txt')
    
    # Perform all analyses
    feature_analysis = analyze_by_category(results, 'feature_method')
    dataset_analysis = analyze_by_category(results, 'dataset_type')
    area_analysis = analyze_by_category(results, 'area')
    k_analysis = analyze_k_value_trends(results)
    feature_comparison = compare_feature_methods(results)
    performance_matrices = create_performance_matrix(results)
    
    # Find best configurations
    best_overall = max(results, key=lambda x: x['mean_accuracy'])
    top_10 = sorted(results, key=lambda x: x['mean_accuracy'], reverse=True)[:10]
    
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
        f.write("RGB RANDOM FOREST EXPERIMENTS - COMPREHENSIVE ANALYSIS REPORT\n")
        f.write("=" * 68 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 17 + "\n")
        f.write(f"Total Experiments Analyzed: {len(results)}\n")
        f.write(f"Accuracy Range: {overall_stats['min']:.4f} - {overall_stats['max']:.4f}\n")
        f.write(f"Mean Accuracy: {overall_stats['mean']:.4f} ± {overall_stats['stdev']:.4f}\n")
        f.write(f"Median Accuracy: {overall_stats['median']:.4f}\n\n")
        
        f.write("EXPERIMENT PARAMETERS\n")
        f.write("-" * 21 + "\n")
        f.write(f"Feature Methods: {sorted(set(r['feature_method'] for r in results))}\n")
        f.write(f"Dataset Types: {sorted(set(r['dataset_type'] for r in results))}\n")
        f.write(f"K Values: {sorted(set(r['k_value'] for r in results))}\n")
        f.write(f"Areas: {sorted(set(r['area'] for r in results))}\n\n")
        
        # Feature Method Analysis
        f.write("FEATURE METHOD PERFORMANCE ANALYSIS\n")
        f.write("-" * 37 + "\n")
        method_ranking = sorted(feature_analysis.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        for rank, (method, stats) in enumerate(method_ranking, 1):
            f.write(f"{rank}. {method.upper()}\n")
            f.write(f"   Mean: {stats['mean']:.4f} ± {stats['stdev']:.4f}\n")
            f.write(f"   Range: {stats['min']:.4f} - {stats['max']:.4f}\n")
            f.write(f"   Experiments: {stats['count']}\n")
            f.write(f"   Median: {stats['median']:.4f}\n\n")
        
        # Method Comparison
        f.write("FEATURE METHOD COMPARISON\n")
        f.write("-" * 26 + "\n")
        
        if 'hybrid_vs_wst' in feature_comparison:
            improvement = feature_comparison['hybrid_vs_wst']
            f.write(f"HYBRID vs WST:\n")
            f.write(f"  Accuracy Improvement: +{improvement['improvement']:.4f}\n")
            f.write(f"  Percentage Improvement: {improvement['improvement_pct']:+.2f}%\n")
            
            if improvement['improvement'] > 0:
                f.write(f"  → HYBRID outperforms WST\n")
            else:
                f.write(f"  → WST outperforms HYBRID\n")
            f.write("\n")
        
        if 'hybrid_vs_advanced' in feature_comparison:
            improvement = feature_comparison['hybrid_vs_advanced']
            f.write(f"HYBRID vs ADVANCED_STATS:\n")
            f.write(f"  Accuracy Improvement: +{improvement['improvement']:.4f}\n")
            f.write(f"  Percentage Improvement: {improvement['improvement_pct']:+.2f}%\n")
            
            if improvement['improvement'] > 0:
                f.write(f"  → HYBRID outperforms ADVANCED_STATS\n")
            else:
                f.write(f"  → ADVANCED_STATS outperforms HYBRID\n")
            f.write("\n")
        
        # K Value Analysis
        f.write("K VALUE ANALYSIS\n")
        f.write("-" * 17 + "\n")
        if 'optimal_k' in k_analysis:
            optimal_k = k_analysis['optimal_k']
            f.write(f"Optimal K Value: {optimal_k}\n")
            f.write(f"Best K Performance: {k_analysis[optimal_k]['mean']:.4f}\n\n")
        
        # Sort K values for display
        k_values_sorted = sorted([k for k in k_analysis.keys() if k != 'optimal_k'])
        for k_value in k_values_sorted:
            stats = k_analysis[k_value]
            f.write(f"K={k_value}: {stats['mean']:.4f} ± {stats['stdev']:.4f} (n={stats['count']})\n")
        f.write("\n")
        
        # Performance Matrix - Method vs K
        f.write("PERFORMANCE MATRIX: METHOD vs K VALUE\n")
        f.write("-" * 39 + "\n")
        if 'method_vs_k' in performance_matrices:
            matrix = performance_matrices['method_vs_k']
            methods = sorted(matrix.keys())
            k_values = sorted(set(k for method_data in matrix.values() for k in method_data.keys()))
            
            # Header
            f.write(f"{'Method':<15}")
            for k in k_values:
                f.write(f"K={k:<8}")
            f.write("\n")
            f.write("-" * (15 + len(k_values) * 9) + "\n")
            
            # Data
            for method in methods:
                f.write(f"{method:<15}")
                for k in k_values:
                    if k in matrix[method]:
                        f.write(f"{matrix[method][k]:<8.4f}")
                    else:
                        f.write(f"{'N/A':<8}")
                f.write("\n")
            f.write("\n")
        
        # Dataset Type Analysis
        f.write("DATASET TYPE ANALYSIS\n")
        f.write("-" * 22 + "\n")
        dataset_ranking = sorted(dataset_analysis.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        for rank, (dataset, stats) in enumerate(dataset_ranking, 1):
            f.write(f"{rank}. {dataset.upper()}\n")
            f.write(f"   Mean: {stats['mean']:.4f} ± {stats['stdev']:.4f}\n")
            f.write(f"   Range: {stats['min']:.4f} - {stats['max']:.4f}\n")
            f.write(f"   Experiments: {stats['count']}\n\n")
        
        # Area Analysis
        f.write("AREA ANALYSIS\n")
        f.write("-" * 14 + "\n")
        area_ranking = sorted(area_analysis.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        for rank, (area, stats) in enumerate(area_ranking, 1):
            f.write(f"{rank}. {area.upper()}\n")
            f.write(f"   Mean: {stats['mean']:.4f} ± {stats['stdev']:.4f}\n")
            f.write(f"   Range: {stats['min']:.4f} - {stats['max']:.4f}\n")
            f.write(f"   Experiments: {stats['count']}\n\n")
        
        # Best Overall Configuration
        f.write("BEST OVERALL CONFIGURATION\n")
        f.write("-" * 28 + "\n")
        f.write(f"Feature Method: {best_overall['feature_method'].upper()}\n")
        f.write(f"Dataset Type: {best_overall['dataset_type'].upper()}\n")
        f.write(f"K Value: {best_overall['k_value']}\n")
        f.write(f"Area: {best_overall['area'].upper()}\n")
        f.write(f"Accuracy: {best_overall['mean_accuracy']:.4f} ± {best_overall['std_accuracy']:.4f}\n")
        f.write(f"Total Images: {best_overall['total_images']}\n")
        f.write(f"Total Features: {best_overall['total_features']}\n\n")
        
        # Top 10 Configurations
        f.write("TOP 10 CONFIGURATIONS\n")
        f.write("-" * 22 + "\n")
        for i, config in enumerate(top_10, 1):
            f.write(f"{i:2d}. {config['feature_method'].upper():<12} ")
            f.write(f"{config['dataset_type'].upper():<8} ")
            f.write(f"K={config['k_value']:<3} ")
            f.write(f"{config['area'].upper():<10} ")
            f.write(f"{config['mean_accuracy']:.4f} ± {config['std_accuracy']:.4f}\n")
        f.write("\n")
        
        # Key Insights
        f.write("KEY INSIGHTS AND RECOMMENDATIONS\n")
        f.write("-" * 34 + "\n")
        
        best_method = method_ranking[0][0]
        best_dataset = dataset_ranking[0][0]
        best_area = area_ranking[0][0]
        
        f.write("1. OPTIMAL CONFIGURATION:\n")
        f.write(f"   • Feature Method: {best_method.upper()}\n")
        f.write(f"   • Dataset Type: {best_dataset.upper()}\n")
        f.write(f"   • Area: {best_area.upper()}\n")
        if 'optimal_k' in k_analysis:
            f.write(f"   • K Value: {k_analysis['optimal_k']}\n")
        f.write("\n")
        
        f.write("2. FEATURE METHOD INSIGHTS:\n")
        if best_method == 'hybrid':
            f.write("   • HYBRID approach provides best overall performance\n")
            f.write("   • Combines advantages of both WST and advanced statistics\n")
            f.write("   • Recommended for maximum accuracy applications\n")
        elif best_method == 'wst':
            f.write("   • WST approach provides excellent performance\n")
            f.write("   • Good balance of accuracy and computational efficiency\n")
            f.write("   • Captures spatial patterns effectively\n")
        else:
            f.write("   • Advanced statistics provide solid baseline performance\n")
            f.write("   • Interpretable features for domain experts\n")
        f.write("\n")
        
        f.write("3. SCALABILITY ANALYSIS:\n")
        # Analyze performance vs dataset size
        dataset_sizes = {'mini': 'smallest', 'small': 'medium', 'original': 'largest'}
        if best_dataset in dataset_sizes:
            f.write(f"   • {best_dataset.upper()} dataset ({dataset_sizes[best_dataset]}) performs best\n")
        
        # Sample size vs performance correlation
        sample_perf = []
        for result in results:
            sample_perf.append((result['total_images'], result['mean_accuracy']))
        
        if len(sample_perf) > 1:
            # Simple correlation analysis
            images = [x[0] for x in sample_perf]
            accs = [x[1] for x in sample_perf]
            
            # Calculate correlation coefficient manually
            n = len(images)
            sum_x = sum(images)
            sum_y = sum(accs)
            sum_xy = sum(x * y for x, y in zip(images, accs))
            sum_x2 = sum(x * x for x in images)
            sum_y2 = sum(y * y for y in accs)
            
            correlation = (n * sum_xy - sum_x * sum_y) / ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
            
            if correlation > 0.3:
                f.write("   • Positive correlation between sample size and performance\n")
            elif correlation < -0.3:
                f.write("   • Negative correlation between sample size and performance\n")
            else:
                f.write("   • No strong correlation between sample size and performance\n")
        f.write("\n")
        
        f.write("4. RECOMMENDATIONS FOR DIFFERENT SCENARIOS:\n")
        f.write("   • Maximum Performance: Use best overall configuration\n")
        f.write("   • Balanced Performance: Use HYBRID with K=10-20\n")
        f.write("   • Fast Processing: Use WST with K=5-10\n")
        f.write("   • Interpretability: Use ADVANCED_STATS with K=10-20\n")
        
        f.write(f"\nReport generated and saved to: {report_path}\n")
    
    print(f"Comprehensive analysis report generated: {report_path}")
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
    
    # Generate comprehensive report
    report_path = generate_comprehensive_report(results, output_dir)
    
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
    
    # Performance improvement analysis
    if 'hybrid' in methods and 'wst' in methods:
        hybrid_avg = statistics.mean(methods['hybrid'])
        wst_avg = statistics.mean(methods['wst'])
        improvement = hybrid_avg - wst_avg
        improvement_pct = (improvement / wst_avg) * 100
        print(f"\nHybrid vs WST: {improvement:+.4f} ({improvement_pct:+.2f}%)")
    
    if 'hybrid' in methods and 'advanced_stats' in methods:
        hybrid_avg = statistics.mean(methods['hybrid'])
        advanced_avg = statistics.mean(methods['advanced_stats'])
        improvement = hybrid_avg - advanced_avg
        improvement_pct = (improvement / advanced_avg) * 100
        print(f"Hybrid vs Advanced Stats: {improvement:+.4f} ({improvement_pct:+.2f}%)")

if __name__ == "__main__":
    main()