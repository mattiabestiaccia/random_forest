#!/usr/bin/env python3
"""
Create text-based visualizations for RGB Random Forest Experiments
=================================================================

This script creates ASCII-based charts and visualizations for the analysis results.
"""

import os
import json
from pathlib import Path
import statistics
from collections import defaultdict

def create_bar_chart(data, title, width=60):
    """
    Create a simple ASCII bar chart
    
    Args:
        data (dict): Data to plot {label: value}
        title (str): Chart title
        width (int): Chart width in characters
    """
    if not data:
        return f"{title}\n(No data available)\n"
    
    # Find the maximum value for scaling
    max_value = max(data.values())
    min_value = min(data.values())
    
    # Calculate scale
    scale = (width - 20) / (max_value - min_value) if max_value > min_value else 1
    
    chart = f"\n{title}\n"
    chart += "=" * len(title) + "\n"
    
    # Sort data by value (descending)
    sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)
    
    for label, value in sorted_data:
        bar_length = int((value - min_value) * scale)
        bar = "█" * bar_length
        chart += f"{label:<15} {bar} {value:.4f}\n"
    
    return chart

def create_line_chart(data, title, width=60):
    """
    Create a simple ASCII line chart
    
    Args:
        data (dict): Data to plot {x: y}
        title (str): Chart title
        width (int): Chart width in characters
    """
    if not data:
        return f"{title}\n(No data available)\n"
    
    chart = f"\n{title}\n"
    chart += "=" * len(title) + "\n"
    
    # Sort data by x value
    sorted_data = sorted(data.items())
    
    # Create the line chart
    for x, y in sorted_data:
        chart += f"K={x:<2} │ {y:.4f}\n"
    
    return chart

def create_comparison_table(matrix, title):
    """
    Create a comparison table
    
    Args:
        matrix (dict): Nested dictionary {method: {k: value}}
        title (str): Table title
    """
    if not matrix:
        return f"{title}\n(No data available)\n"
    
    table = f"\n{title}\n"
    table += "=" * len(title) + "\n"
    
    # Get all methods and k values
    methods = sorted(matrix.keys())
    all_k_values = set()
    for method_data in matrix.values():
        all_k_values.update(method_data.keys())
    k_values = sorted(all_k_values)
    
    # Header
    table += f"{'Method':<15}"
    for k in k_values:
        table += f"K={k:<8}"
    table += "\n"
    table += "-" * (15 + len(k_values) * 9) + "\n"
    
    # Data rows
    for method in methods:
        table += f"{method:<15}"
        for k in k_values:
            if k in matrix[method]:
                table += f"{matrix[method][k]:<8.4f}"
            else:
                table += f"{'N/A':<8}"
        table += "\n"
    
    return table

def extract_experiment_data(base_dir):
    """
    Extract data from experiments
    """
    results = []
    base_path = Path(base_dir)
    
    for exp_dir in base_path.iterdir():
        if exp_dir.is_dir():
            json_files = list(exp_dir.glob('*.json'))
            if json_files:
                try:
                    with open(json_files[0], 'r') as f:
                        data = json.load(f)
                    
                    results.append({
                        'feature_method': data.get('feature_method', 'unknown'),
                        'dataset_type': data.get('dataset_type', 'unknown'),
                        'k_value': data.get('k_features', 0),
                        'area': data.get('area_name', 'unknown'),
                        'mean_accuracy': data.get('performance', {}).get('mean_accuracy', 0),
                        'std_accuracy': data.get('performance', {}).get('std_accuracy', 0),
                        'total_images': data.get('dataset_info', {}).get('total_images', 0),
                        'total_features': data.get('dataset_info', {}).get('total_features_available', 0),
                        'experiment_name': exp_dir.name
                    })
                except Exception as e:
                    continue
    
    return results

def generate_visualizations(results, output_dir):
    """
    Generate various visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    viz_path = os.path.join(output_dir, 'visualizations.txt')
    
    with open(viz_path, 'w') as f:
        f.write("RGB RANDOM FOREST EXPERIMENTS - VISUALIZATIONS\n")
        f.write("=" * 50 + "\n\n")
        
        # 1. Feature Method Performance
        method_performance = {}
        for result in results:
            method = result['feature_method']
            if method not in method_performance:
                method_performance[method] = []
            method_performance[method].append(result['mean_accuracy'])
        
        method_avg = {method: statistics.mean(accs) for method, accs in method_performance.items()}
        f.write(create_bar_chart(method_avg, "FEATURE METHOD PERFORMANCE (Average Accuracy)"))
        
        # 2. K Value Trends
        k_performance = {}
        for result in results:
            k = result['k_value']
            if k not in k_performance:
                k_performance[k] = []
            k_performance[k].append(result['mean_accuracy'])
        
        k_avg = {k: statistics.mean(accs) for k, accs in k_performance.items()}
        f.write(create_line_chart(k_avg, "K VALUE PERFORMANCE TREND"))
        
        # 3. Dataset Type Performance
        dataset_performance = {}
        for result in results:
            dataset = result['dataset_type']
            if dataset not in dataset_performance:
                dataset_performance[dataset] = []
            dataset_performance[dataset].append(result['mean_accuracy'])
        
        dataset_avg = {dataset: statistics.mean(accs) for dataset, accs in dataset_performance.items()}
        f.write(create_bar_chart(dataset_avg, "DATASET TYPE PERFORMANCE (Average Accuracy)"))
        
        # 4. Area Performance
        area_performance = {}
        for result in results:
            area = result['area']
            if area not in area_performance:
                area_performance[area] = []
            area_performance[area].append(result['mean_accuracy'])
        
        area_avg = {area: statistics.mean(accs) for area, accs in area_performance.items()}
        f.write(create_bar_chart(area_avg, "AREA PERFORMANCE (Average Accuracy)"))
        
        # 5. Method vs K Value Matrix
        method_k_matrix = defaultdict(dict)
        for result in results:
            method = result['feature_method']
            k = result['k_value']
            
            if k not in method_k_matrix[method]:
                method_k_matrix[method][k] = []
            method_k_matrix[method][k].append(result['mean_accuracy'])
        
        # Calculate averages
        for method in method_k_matrix:
            for k in method_k_matrix[method]:
                method_k_matrix[method][k] = statistics.mean(method_k_matrix[method][k])
        
        f.write(create_comparison_table(dict(method_k_matrix), "METHOD vs K VALUE PERFORMANCE MATRIX"))
        
        # 6. Performance Analysis by Method and K
        f.write("\nDETAILED PERFORMANCE ANALYSIS BY METHOD AND K VALUE\n")
        f.write("=" * 55 + "\n")
        
        for method in sorted(method_k_matrix.keys()):
            f.write(f"\n{method.upper()} Performance by K Value:\n")
            f.write("-" * 30 + "\n")
            
            k_values = sorted(method_k_matrix[method].keys())
            for k in k_values:
                acc = method_k_matrix[method][k]
                f.write(f"K={k:<2}: {acc:.4f}\n")
            
            # Calculate improvement from K=2 to K=20
            if 2 in method_k_matrix[method] and 20 in method_k_matrix[method]:
                improvement = method_k_matrix[method][20] - method_k_matrix[method][2]
                improvement_pct = (improvement / method_k_matrix[method][2]) * 100
                f.write(f"Improvement K=2→K=20: {improvement:+.4f} ({improvement_pct:+.2f}%)\n")
        
        # 7. Best Configurations Analysis
        f.write("\nBEST CONFIGURATIONS ANALYSIS\n")
        f.write("=" * 30 + "\n")
        
        # Overall best
        best_config = max(results, key=lambda x: x['mean_accuracy'])
        f.write(f"OVERALL BEST:\n")
        f.write(f"  Method: {best_config['feature_method'].upper()}\n")
        f.write(f"  Dataset: {best_config['dataset_type'].upper()}\n")
        f.write(f"  K Value: {best_config['k_value']}\n")
        f.write(f"  Area: {best_config['area'].upper()}\n")
        f.write(f"  Accuracy: {best_config['mean_accuracy']:.4f} ± {best_config['std_accuracy']:.4f}\n")
        f.write(f"  Images: {best_config['total_images']}\n\n")
        
        # Best for each method
        f.write("BEST FOR EACH METHOD:\n")
        for method in sorted(method_performance.keys()):
            method_results = [r for r in results if r['feature_method'] == method]
            best_method = max(method_results, key=lambda x: x['mean_accuracy'])
            f.write(f"  {method.upper()}: {best_method['mean_accuracy']:.4f} ")
            f.write(f"(Dataset: {best_method['dataset_type']}, K={best_method['k_value']}, Area: {best_method['area']})\n")
        
        # 8. Statistical Summary
        f.write("\nSTATISTICAL SUMMARY\n")
        f.write("=" * 20 + "\n")
        
        all_accuracies = [r['mean_accuracy'] for r in results]
        f.write(f"Total Experiments: {len(results)}\n")
        f.write(f"Mean Accuracy: {statistics.mean(all_accuracies):.4f}\n")
        f.write(f"Median Accuracy: {statistics.median(all_accuracies):.4f}\n")
        f.write(f"Standard Deviation: {statistics.stdev(all_accuracies):.4f}\n")
        f.write(f"Min Accuracy: {min(all_accuracies):.4f}\n")
        f.write(f"Max Accuracy: {max(all_accuracies):.4f}\n")
        f.write(f"Range: {max(all_accuracies) - min(all_accuracies):.4f}\n")
        
        # 9. Method Comparison Summary
        f.write("\nMETHOD COMPARISON SUMMARY\n")
        f.write("=" * 26 + "\n")
        
        method_stats = {}
        for method, accs in method_performance.items():
            method_stats[method] = {
                'mean': statistics.mean(accs),
                'std': statistics.stdev(accs) if len(accs) > 1 else 0,
                'min': min(accs),
                'max': max(accs),
                'count': len(accs)
            }
        
        # Sort by mean performance
        sorted_methods = sorted(method_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        for rank, (method, stats) in enumerate(sorted_methods, 1):
            f.write(f"{rank}. {method.upper()}\n")
            f.write(f"   Mean: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
            f.write(f"   Range: {stats['min']:.4f} - {stats['max']:.4f}\n")
            f.write(f"   Experiments: {stats['count']}\n\n")
        
        # Performance improvements
        if len(sorted_methods) >= 2:
            best_method = sorted_methods[0]
            second_method = sorted_methods[1]
            improvement = best_method[1]['mean'] - second_method[1]['mean']
            improvement_pct = (improvement / second_method[1]['mean']) * 100
            
            f.write(f"PERFORMANCE ADVANTAGE:\n")
            f.write(f"{best_method[0].upper()} vs {second_method[0].upper()}: ")
            f.write(f"{improvement:+.4f} ({improvement_pct:+.2f}%)\n")
        
        f.write(f"\nVisualizations saved to: {viz_path}\n")
    
    print(f"Text-based visualizations created: {viz_path}")

def main():
    """
    Main function to create visualizations
    """
    base_dir = "/home/brusc/Projects/random_forest/experiments/rgb_gaussian50_kbest"
    output_dir = "/home/brusc/Projects/random_forest/experiments/analysis_results"
    
    print("Creating visualizations...")
    
    # Extract data
    results = extract_experiment_data(base_dir)
    
    if not results:
        print("No data found!")
        return
    
    # Generate visualizations
    generate_visualizations(results, output_dir)
    
    print("Visualizations completed!")

if __name__ == "__main__":
    main()