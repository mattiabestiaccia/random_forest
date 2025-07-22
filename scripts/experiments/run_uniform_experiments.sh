#!/bin/bash

# RGB Random Forest - Uniform Experiments Runner
# ==============================================
# This script runs experiments on uniform datasets in sequence:
# 1. dataset_rgb_uniform_10
# 2. dataset_rgb_uniform_25
# 3. dataset_rgb_uniform_40
# All output folders will be created in: ./uniform_output/
# 
# Parameters:
# - dataset_type: mini, small, original, augmented
# - area_name: assatigue, popolar, sunset
# - feature_method: wst, advanced_stats, hybrid
# - k_features: 2, 5, 10, 20
#
# Usage: ./run_uniform_experiments.sh [base_dataset_dir]

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="/home/brusc/Projects/random_forest/scripts/train_and_save_model.py"
OUTPUT_BASE_DIR="$SCRIPT_DIR/uniform_output"
LOG_DIR="$OUTPUT_BASE_DIR/experiment_logs"
RESULTS_DIR="$OUTPUT_BASE_DIR/all_results"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Default base dataset directory
BASE_DATASET_DIR="${1:-/home/brusc/Projects/random_forest/datasets}"

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Check if base dataset directory exists
if [ ! -d "$BASE_DATASET_DIR" ]; then
    echo "Error: Base dataset directory not found: $BASE_DATASET_DIR"
    exit 1
fi

# Uniform experiment configurations - processing in sequence by intensity
UNIFORM_DATASETS=("dataset_rgb_uniform_10" "dataset_rgb_uniform_25" "dataset_rgb_uniform_40")
DATASET_TYPES=("mini" "small" "original")
AREAS=("assatigue" "popolar" "sunset")
FEATURE_METHODS=("wst" "advanced_stats" "hybrid")
K_VALUES=(2 5 10 20)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log with timestamp
log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")
            echo -e "${GREEN}[$timestamp] INFO: $message${NC}"
            ;;
        "WARNING")
            echo -e "${YELLOW}[$timestamp] WARNING: $message${NC}"
            ;;
        "ERROR")
            echo -e "${RED}[$timestamp] ERROR: $message${NC}"
            ;;
        "EXPERIMENT")
            echo -e "${BLUE}[$timestamp] EXPERIMENT: $message${NC}"
            ;;
    esac
}

# Function to check if experiment is already completed
is_experiment_completed() {
    local uniform_dataset=$1
    local dataset_type=$2
    local area=$3
    local feature_method=$4
    local k_features=$5
    
    # Create the structured output directory path: uniform_dataset/area/dataset_type/k{features}/feature_method/
    local output_dir="$OUTPUT_BASE_DIR/$uniform_dataset/$area/$dataset_type/k${k_features}/$feature_method"
    
    # Check if experiment output directory exists and contains expected files
    if [ -d "$output_dir" ]; then
        # Look for completion indicators: model files, report files, etc.
        local model_files=$(find "$output_dir" -name "*.pkl" -o -name "*.joblib" 2>/dev/null | wc -l)
        local report_files=$(find "$output_dir" -name "*_report.json" 2>/dev/null | wc -l)
        
        if [ $model_files -gt 0 ] || [ $report_files -gt 0 ]; then
            return 0  # Experiment is completed
        fi
    fi
    
    return 1  # Experiment is not completed
}

# Function to run a single experiment
run_experiment() {
    local uniform_dataset=$1
    local dataset_type=$2
    local area=$3
    local feature_method=$4
    local k_features=$5
    
    local experiment_name="${uniform_dataset}_${dataset_type}_${area}_${feature_method}_k${k_features}"
    local log_file="$LOG_DIR/${experiment_name}.log"
    local error_file="$LOG_DIR/${experiment_name}.error"
    
    # Check if experiment is already completed
    if is_experiment_completed "$uniform_dataset" "$dataset_type" "$area" "$feature_method" "$k_features"; then
        log_message "EXPERIMENT" "Skipping completed: $experiment_name"
        return 0
    fi
    
    log_message "EXPERIMENT" "Starting: $experiment_name"
    
    # Create the structured output directory path: uniform_dataset/area/dataset_type/k{features}/feature_method/
    local structured_output_dir="$OUTPUT_BASE_DIR/$uniform_dataset/$area/$dataset_type/k${k_features}/$feature_method"
    
    # Create the output directory structure
    mkdir -p "$structured_output_dir"
    
    # Determine the dataset path
    local dataset_path="$BASE_DATASET_DIR/$uniform_dataset/$dataset_type"
    
    # Check if the specific dataset path exists
    if [ ! -d "$dataset_path" ]; then
        log_message "ERROR" "Dataset path not found: $dataset_path for experiment $experiment_name"
        return 1
    fi
    
    # Activate virtual environment and run the experiment
    if (cd "$OUTPUT_BASE_DIR" && source /home/brusc/Projects/random_forest/venv_rf/bin/activate && python3 "$PYTHON_SCRIPT" "$dataset_path" "$area" "$feature_method" "$k_features" "$structured_output_dir") > "$log_file" 2> "$error_file"; then
        log_message "EXPERIMENT" "Completed: $experiment_name"
        return 0
    else
        log_message "EXPERIMENT" "Failed: $experiment_name (see $error_file)"
        return 1
    fi
}

# Function to generate summary report
generate_summary() {
    local summary_file="$RESULTS_DIR/uniform_experiment_summary.txt"
    local csv_file="$RESULTS_DIR/uniform_experiment_results.csv"
    
    log_message "INFO" "Generating summary report..."
    
    # Create CSV header
    echo "UniformDataset,Dataset,Area,FeatureMethod,K,Accuracy,StdDev,Status" > "$csv_file"
    
    # Create summary header
    {
        echo "RGB Random Forest - Uniform Experiments Summary"
        echo "==============================================="
        echo "Generated: $(date)"
        echo "Base Dataset Directory: $BASE_DATASET_DIR"
        echo "Output Directory: $OUTPUT_BASE_DIR"
        echo ""
        echo "Processed Datasets: ${UNIFORM_DATASETS[*]}"
        echo ""
        echo "Experiment Results:"
        echo "==================="
    } > "$summary_file"
    
    local total_experiments=0
    local successful_experiments=0
    
    # Process each experiment
    for uniform_dataset in "${UNIFORM_DATASETS[@]}"; do
        echo "" >> "$summary_file"
        echo "=== $uniform_dataset ===" >> "$summary_file"
        
        for dataset_type in "${DATASET_TYPES[@]}"; do
            for area in "${AREAS[@]}"; do
                for feature_method in "${FEATURE_METHODS[@]}"; do
                    for k_features in "${K_VALUES[@]}"; do
                        total_experiments=$((total_experiments + 1))
                        
                        local experiment_name="${uniform_dataset}_${dataset_type}_${area}_${feature_method}_k${k_features}"
                        local log_file="$LOG_DIR/${experiment_name}.log"
                        local error_file="$LOG_DIR/${experiment_name}.error"
                        
                        if [ -f "$log_file" ] && [ ! -s "$error_file" ]; then
                            # Extract accuracy from log file
                            local accuracy=$(grep "CV Accuracy:" "$log_file" | tail -1 | sed 's/.*CV Accuracy: \([0-9.]*\) ± \([0-9.]*\).*/\1/')
                            local std_dev=$(grep "CV Accuracy:" "$log_file" | tail -1 | sed 's/.*CV Accuracy: \([0-9.]*\) ± \([0-9.]*\).*/\2/')
                            
                            if [ -n "$accuracy" ] && [ -n "$std_dev" ]; then
                                successful_experiments=$((successful_experiments + 1))
                                
                                # Add to CSV
                                echo "$uniform_dataset,$dataset_type,$area,$feature_method,$k_features,$accuracy,$std_dev,SUCCESS" >> "$csv_file"
                                
                                # Add to summary
                                echo "✓ $experiment_name: ${accuracy} ± ${std_dev}" >> "$summary_file"
                            else
                                echo "$uniform_dataset,$dataset_type,$area,$feature_method,$k_features,,,FAILED" >> "$csv_file"
                                echo "✗ $experiment_name: FAILED (no accuracy found)" >> "$summary_file"
                            fi
                        else
                            echo "$uniform_dataset,$dataset_type,$area,$feature_method,$k_features,,,FAILED" >> "$csv_file"
                            echo "✗ $experiment_name: FAILED" >> "$summary_file"
                        fi
                    done
                done
            done
        done
    done
    
    # Add summary statistics
    {
        echo ""
        echo "Summary Statistics:"
        echo "=================="
        echo "Total experiments: $total_experiments"
        echo "Successful experiments: $successful_experiments"
        echo "Failed experiments: $((total_experiments - successful_experiments))"
        echo "Success rate: $(( successful_experiments * 100 / total_experiments ))%"
        echo ""
        echo "Experiments per dataset:"
        for uniform_dataset in "${UNIFORM_DATASETS[@]}"; do
            local dataset_experiments=$(grep "^$uniform_dataset," "$csv_file" | wc -l)
            local dataset_success=$(grep "^$uniform_dataset,.*SUCCESS$" "$csv_file" | wc -l)
            echo "  $uniform_dataset: $dataset_success/$dataset_experiments successful"
        done
    } >> "$summary_file"
    
    log_message "INFO" "Summary report generated: $summary_file"
    log_message "INFO" "CSV results generated: $csv_file"
}

# Main execution
main() {
    log_message "INFO" "Starting RGB Random Forest - Uniform Experiments"
    log_message "INFO" "Base dataset directory: $BASE_DATASET_DIR"
    log_message "INFO" "Output base directory: $OUTPUT_BASE_DIR"
    log_message "INFO" "Python script: $PYTHON_SCRIPT"
    log_message "INFO" "Log directory: $LOG_DIR"
    log_message "INFO" "Results directory: $RESULTS_DIR"
    log_message "INFO" "Processing datasets in sequence: ${UNIFORM_DATASETS[*]}"
    
    # Calculate total experiments
    local total_experiments=$(( ${#UNIFORM_DATASETS[@]} * ${#DATASET_TYPES[@]} * ${#AREAS[@]} * ${#FEATURE_METHODS[@]} * ${#K_VALUES[@]} ))
    log_message "INFO" "Total experiments to run: $total_experiments"
    
    local current_experiment=0
    local failed_experiments=0
    local skipped_experiments=0
    local successful_experiments=0
    
    # Run all experiments - process uniform datasets in sequence
    for uniform_dataset in "${UNIFORM_DATASETS[@]}"; do
        log_message "INFO" "Starting uniform dataset: $uniform_dataset"
        for dataset_type in "${DATASET_TYPES[@]}"; do
            log_message "INFO" "Starting dataset type: $dataset_type"
            for area in "${AREAS[@]}"; do
                log_message "INFO" "Starting area: $area"
                for feature_method in "${FEATURE_METHODS[@]}"; do
                    log_message "INFO" "Starting feature method: $feature_method"
                    for k_features in "${K_VALUES[@]}"; do
                        current_experiment=$((current_experiment + 1))
                        local experiment_name="${uniform_dataset}_${dataset_type}_${area}_${feature_method}_k${k_features}"
                        
                        echo ""
                        log_message "INFO" "Progress: $current_experiment/$total_experiments - $experiment_name"
                        
                        # Check if already completed before running
                        if is_experiment_completed "$uniform_dataset" "$dataset_type" "$area" "$feature_method" "$k_features"; then
                            log_message "INFO" "Experiment already completed, skipping: $experiment_name"
                            skipped_experiments=$((skipped_experiments + 1))
                            successful_experiments=$((successful_experiments + 1))
                        else
                            if run_experiment "$uniform_dataset" "$dataset_type" "$area" "$feature_method" "$k_features"; then
                                successful_experiments=$((successful_experiments + 1))
                                log_message "INFO" "Successfully completed experiment: $experiment_name"
                            else
                                failed_experiments=$((failed_experiments + 1))
                                log_message "ERROR" "Failed experiment: $experiment_name"
                            fi
                        fi
                        
                        # Brief pause between experiments
                        sleep 1
                    done
                    log_message "INFO" "Completed feature method: $feature_method"
                done
                log_message "INFO" "Completed area: $area"
            done
            log_message "INFO" "Completed dataset type: $dataset_type"
        done
        log_message "INFO" "Completed uniform dataset: $uniform_dataset"
        echo ""
    done
    
    # Generate summary
    echo ""
    log_message "INFO" "All uniform experiments completed!"
    log_message "INFO" "Total experiments: $current_experiment"
    log_message "INFO" "Successful: $successful_experiments"
    log_message "INFO" "Failed: $failed_experiments"
    log_message "INFO" "Skipped (already completed): $skipped_experiments"
    log_message "INFO" "Success rate: $(( successful_experiments * 100 / current_experiment ))%"
    
    generate_summary
    
    # Final message
    echo ""
    log_message "INFO" "All uniform experiments finished!"
    log_message "INFO" "Individual experiment folders created in: $OUTPUT_BASE_DIR"
    log_message "INFO" "Check logs in: $LOG_DIR"
    log_message "INFO" "Check results in: $RESULTS_DIR"
}

# Trap to handle interruption
trap 'log_message "WARNING" "Uniform experiments interrupted by user"; exit 1' INT

# Check if running as main script
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi