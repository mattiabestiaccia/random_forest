#!/bin/bash

# RGB Random Forest - All Experiments Runner
# ==========================================
# This script runs all possible combinations of experiments in a systematic order
# All output folders will be created in: ./rgb_gaussian30_kbest/
# 
# Parameters:
# - dataset_type: mini, small, original, augmented
# - area_name: assatigue, popolar, sunset
# - feature_method: wst, advanced_stats, hybrid
# - k_features: 2, 5, 10, 20, 30, 50
#
# Usage: ./run_all_experiments.sh [base_dataset_dir]
#        ./run_all_experiments.sh /custom/path/to/dataset_rgb

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/rgb_rf_generalized.py"
OUTPUT_BASE_DIR="$SCRIPT_DIR/rgb__kbest"
LOG_DIR="$OUTPUT_BASE_DIR/experiment_logs"
RESULTS_DIR="$OUTPUT_BASE_DIR/all_results"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Default base dataset directory
BASE_DATASET_DIR="${1:-/home/brusc/Projects/random_forest/dataset_rgb}"

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

# Experiment parameters
DATASET_TYPES=("mini" "small" "original" "augmented")
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

# Function to run a single experiment
run_experiment() {
    local dataset_type=$1
    local area=$2
    local feature_method=$3
    local k_features=$4
    
    local experiment_name="${dataset_type}_${area}_${feature_method}_k${k_features}"
    local log_file="$LOG_DIR/${experiment_name}.log"
    local error_file="$LOG_DIR/${experiment_name}.error"
    
    log_message "EXPERIMENT" "Starting: $experiment_name"
    
    # Run the experiment (change to output directory so experiments are created there)
    if (cd "$OUTPUT_BASE_DIR" && python3 "$PYTHON_SCRIPT" "$dataset_type" "$area" "$feature_method" "$k_features" "$BASE_DATASET_DIR") > "$log_file" 2> "$error_file"; then
        log_message "INFO" "Completed: $experiment_name"
        return 0
    else
        log_message "ERROR" "Failed: $experiment_name (see $error_file)"
        return 1
    fi
}

# Function to generate summary report
generate_summary() {
    local summary_file="$RESULTS_DIR/experiment_summary.txt"
    local csv_file="$RESULTS_DIR/experiment_results.csv"
    
    log_message "INFO" "Generating summary report..."
    
    # Create CSV header
    echo "Dataset,Area,FeatureMethod,K,Accuracy,StdDev,Status" > "$csv_file"
    
    # Create summary header
    {
        echo "RGB Random Forest - All Experiments Summary"
        echo "=========================================="
        echo "Generated: $(date)"
        echo "Base Dataset Directory: $BASE_DATASET_DIR"
        echo "Output Directory: $OUTPUT_BASE_DIR"
        echo ""
        echo "Experiment Results:"
        echo "==================="
    } > "$summary_file"
    
    local total_experiments=0
    local successful_experiments=0
    
    # Process each experiment
    for dataset_type in "${DATASET_TYPES[@]}"; do
        for area in "${AREAS[@]}"; do
            for feature_method in "${FEATURE_METHODS[@]}"; do
                for k_features in "${K_VALUES[@]}"; do
                    total_experiments=$((total_experiments + 1))
                    
                    local experiment_name="${dataset_type}_${area}_${feature_method}_k${k_features}"
                    local log_file="$LOG_DIR/${experiment_name}.log"
                    local error_file="$LOG_DIR/${experiment_name}.error"
                    
                    if [ -f "$log_file" ] && [ ! -s "$error_file" ]; then
                        # Extract accuracy from log file
                        local accuracy=$(grep "Accuracy:" "$log_file" | tail -1 | sed 's/.*Accuracy: \([0-9.]*\) ± \([0-9.]*\).*/\1/')
                        local std_dev=$(grep "Accuracy:" "$log_file" | tail -1 | sed 's/.*Accuracy: \([0-9.]*\) ± \([0-9.]*\).*/\2/')
                        
                        if [ -n "$accuracy" ] && [ -n "$std_dev" ]; then
                            successful_experiments=$((successful_experiments + 1))
                            
                            # Add to CSV
                            echo "$dataset_type,$area,$feature_method,$k_features,$accuracy,$std_dev,SUCCESS" >> "$csv_file"
                            
                            # Add to summary
                            echo "✓ $experiment_name: ${accuracy} ± ${std_dev}" >> "$summary_file"
                        else
                            echo "$dataset_type,$area,$feature_method,$k_features,,,FAILED" >> "$csv_file"
                            echo "✗ $experiment_name: FAILED (no accuracy found)" >> "$summary_file"
                        fi
                    else
                        echo "$dataset_type,$area,$feature_method,$k_features,,,FAILED" >> "$csv_file"
                        echo "✗ $experiment_name: FAILED" >> "$summary_file"
                    fi
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
    } >> "$summary_file"
    
    log_message "INFO" "Summary report generated: $summary_file"
    log_message "INFO" "CSV results generated: $csv_file"
}

# Main execution
main() {
    log_message "INFO" "Starting RGB Random Forest - All Experiments"
    log_message "INFO" "Base dataset directory: $BASE_DATASET_DIR"
    log_message "INFO" "Output base directory: $OUTPUT_BASE_DIR"
    log_message "INFO" "Python script: $PYTHON_SCRIPT"
    log_message "INFO" "Log directory: $LOG_DIR"
    log_message "INFO" "Results directory: $RESULTS_DIR"
    
    # Calculate total experiments
    local total_experiments=$(( ${#DATASET_TYPES[@]} * ${#AREAS[@]} * ${#FEATURE_METHODS[@]} * ${#K_VALUES[@]} ))
    log_message "INFO" "Total experiments to run: $total_experiments"
    
    local current_experiment=0
    local failed_experiments=0
    
    # Run all experiments
    for dataset_type in "${DATASET_TYPES[@]}"; do
        for area in "${AREAS[@]}"; do
            for feature_method in "${FEATURE_METHODS[@]}"; do
                for k_features in "${K_VALUES[@]}"; do
                    current_experiment=$((current_experiment + 1))
                    
                    echo ""
                    log_message "INFO" "Progress: $current_experiment/$total_experiments"
                    
                    if ! run_experiment "$dataset_type" "$area" "$feature_method" "$k_features"; then
                        failed_experiments=$((failed_experiments + 1))
                    fi
                    
                    # Brief pause between experiments
                    sleep 1
                done
            done
        done
    done
    
    # Generate summary
    echo ""
    log_message "INFO" "All experiments completed!"
    log_message "INFO" "Successful: $((current_experiment - failed_experiments))/$current_experiment"
    log_message "INFO" "Failed: $failed_experiments/$current_experiment"
    
    generate_summary
    
    # Final message
    echo ""
    log_message "INFO" "All experiments finished!"
    log_message "INFO" "Individual experiment folders created in: $OUTPUT_BASE_DIR"
    log_message "INFO" "Check logs in: $LOG_DIR"
    log_message "INFO" "Check results in: $RESULTS_DIR"
}

# Trap to handle interruption
trap 'log_message "WARNING" "Experiments interrupted by user"; exit 1' INT

# Check if running as main script
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi