#!/bin/bash

# Batch Runner for All Datasets - RGB Random Forest Experiments
# =============================================================
# This script runs experiments on all datasets in /home/brusc/Projects/random_forest/datasets
# and saves results to /home/brusc/Projects/random_forest/experiments_organized_2

# Configuration
DATASETS_DIR="/home/brusc/Projects/random_forest/datasets"
OUTPUT_BASE_DIR="/home/brusc/Projects/random_forest/experiments_organized_2"
RUN_EXPERIMENTS_SCRIPT="/home/brusc/Projects/random_forest/experiments_organized/run_all_experiments.sh"
LOG_FILE="$OUTPUT_BASE_DIR/batch_run.log"

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
            echo "[$timestamp] INFO: $message" >> "$LOG_FILE"
            ;;
        "WARNING")
            echo -e "${YELLOW}[$timestamp] WARNING: $message${NC}"
            echo "[$timestamp] WARNING: $message" >> "$LOG_FILE"
            ;;
        "ERROR")
            echo -e "${RED}[$timestamp] ERROR: $message${NC}"
            echo "[$timestamp] ERROR: $message" >> "$LOG_FILE"
            ;;
        "DATASET")
            echo -e "${BLUE}[$timestamp] DATASET: $message${NC}"
            echo "[$timestamp] DATASET: $message" >> "$LOG_FILE"
            ;;
    esac
}

# Create output directory
mkdir -p "$OUTPUT_BASE_DIR"

# Initialize log file
echo "Batch Run Started: $(date)" > "$LOG_FILE"
echo "Datasets Directory: $DATASETS_DIR" >> "$LOG_FILE"
echo "Output Directory: $OUTPUT_BASE_DIR" >> "$LOG_FILE"
echo "Run Experiments Script: $RUN_EXPERIMENTS_SCRIPT" >> "$LOG_FILE"
echo "=======================================" >> "$LOG_FILE"

# Check if directories and script exist
if [ ! -d "$DATASETS_DIR" ]; then
    log_message "ERROR" "Datasets directory not found: $DATASETS_DIR"
    exit 1
fi

if [ ! -f "$RUN_EXPERIMENTS_SCRIPT" ]; then
    log_message "ERROR" "Run experiments script not found: $RUN_EXPERIMENTS_SCRIPT"
    exit 1
fi

# Make sure run_all_experiments.sh is executable
chmod +x "$RUN_EXPERIMENTS_SCRIPT"

# Get list of dataset directories
datasets=($(find "$DATASETS_DIR" -maxdepth 1 -type d -name "dataset_rgb_*" | sort))

if [ ${#datasets[@]} -eq 0 ]; then
    log_message "ERROR" "No dataset directories found in $DATASETS_DIR"
    exit 1
fi

log_message "INFO" "Found ${#datasets[@]} datasets to process"
log_message "INFO" "Starting batch processing..."

# Process each dataset
total_datasets=${#datasets[@]}
current_dataset=0
successful_datasets=0
failed_datasets=0

for dataset_path in "${datasets[@]}"; do
    current_dataset=$((current_dataset + 1))
    dataset_name=$(basename "$dataset_path")
    
    echo ""
    log_message "DATASET" "Processing dataset $current_dataset/$total_datasets: $dataset_name"
    
    # Create a temporary copy of run_all_experiments.sh with modified OUTPUT_BASE_DIR
    temp_script="/tmp/run_experiments_${dataset_name}.sh"
    cp "$RUN_EXPERIMENTS_SCRIPT" "$temp_script"
    
    # Modify the output directory for this specific dataset
    output_dir_name="rgb_$(echo "$dataset_name" | sed 's/dataset_rgb_//')_kbest"
    sed -i "s|OUTPUT_BASE_DIR=\"\$SCRIPT_DIR/rgb_poisson100_kbest\"|OUTPUT_BASE_DIR=\"$OUTPUT_BASE_DIR/$output_dir_name\"|g" "$temp_script"
    
    # Make temporary script executable
    chmod +x "$temp_script"
    
    # Run experiments for this dataset
    start_time=$(date +%s)
    
    if "$temp_script" "$dataset_path" >> "$LOG_FILE" 2>&1; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        log_message "INFO" "Completed dataset $dataset_name in ${duration}s"
        successful_datasets=$((successful_datasets + 1))
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        log_message "ERROR" "Failed to process dataset $dataset_name after ${duration}s"
        failed_datasets=$((failed_datasets + 1))
    fi
    
    # Clean up temporary script
    rm -f "$temp_script"
done

# Generate final summary
echo "" | tee -a "$LOG_FILE"
log_message "INFO" "Batch processing completed!"
log_message "INFO" "Total datasets processed: $total_datasets"
log_message "INFO" "Successful: $successful_datasets"
log_message "INFO" "Failed: $failed_datasets"
log_message "INFO" "Success rate: $(( successful_datasets * 100 / total_datasets ))%"
log_message "INFO" "Results saved in: $OUTPUT_BASE_DIR"
log_message "INFO" "Full log available at: $LOG_FILE"

# Create summary file
summary_file="$OUTPUT_BASE_DIR/batch_summary.txt"
{
    echo "Batch Processing Summary"
    echo "======================="
    echo "Completed: $(date)"
    echo "Total datasets: $total_datasets"
    echo "Successful: $successful_datasets"
    echo "Failed: $failed_datasets"
    echo "Success rate: $(( successful_datasets * 100 / total_datasets ))%"
    echo ""
    echo "Processed datasets:"
    for dataset_path in "${datasets[@]}"; do
        dataset_name=$(basename "$dataset_path")
        echo "  - $dataset_name"
    done
} > "$summary_file"

log_message "INFO" "Summary saved to: $summary_file"

# Exit with appropriate code
if [ $failed_datasets -eq 0 ]; then
    exit 0
else
    exit 1
fi