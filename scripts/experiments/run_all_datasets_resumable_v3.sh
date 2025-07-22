#!/bin/bash

# Resumable Batch Runner for All Datasets - RGB Random Forest Experiments (v3)
# =============================================================================
# This script runs experiments on all datasets with granular resume capability
# that can resume individual experiments within datasets, not just entire datasets

# Configuration
DATASETS_DIR="/home/brusc/Projects/random_forest/datasets"
OUTPUT_BASE_DIR="/home/brusc/Projects/random_forest/experiments_organized_2"
RUN_EXPERIMENTS_SCRIPT="/home/brusc/Projects/random_forest/experiments_organized/run_all_experiments.sh"
LOG_FILE="$OUTPUT_BASE_DIR/batch_run.log"
PROGRESS_FILE="$OUTPUT_BASE_DIR/batch_progress.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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
        "RESUME")
            echo -e "${CYAN}[$timestamp] RESUME: $message${NC}"
            echo "[$timestamp] RESUME: $message" >> "$LOG_FILE"
            ;;
    esac
}

# Function to get dataset progress from log file
get_dataset_progress() {
    local dataset_name=$1
    local output_dir_name="rgb_$(echo "$dataset_name" | sed 's/dataset_rgb_//')_kbest"
    local dataset_output_dir="$OUTPUT_BASE_DIR/$output_dir_name"
    
    # Expected experiments per dataset
    local areas=("assatigue" "popolar" "sunset")
    local dataset_types=("mini" "original" "small" "augmented")  
    local feature_methods=("advanced_stats" "hybrid" "wst")
    local k_values=(2 5 10 20)
    
    local total_experiments=$(( ${#areas[@]} * ${#dataset_types[@]} * ${#feature_methods[@]} * ${#k_values[@]} ))
    local completed_experiments=0
    
    # Check each possible experiment using the new structured path: area/dataset_type/k{value}/feature_method/
    for area in "${areas[@]}"; do
        for dataset_type in "${dataset_types[@]}"; do
            for feature_method in "${feature_methods[@]}"; do
                for k_value in "${k_values[@]}"; do
                    local experiment_dir="$dataset_output_dir/$area/$dataset_type/k${k_value}/$feature_method"
                    
                    # Check if experiment is completed (has model files or report files)
                    if [ -d "$experiment_dir" ]; then
                        local model_files=$(find "$experiment_dir" -name "*.pkl" -o -name "*.joblib" 2>/dev/null | wc -l)
                        local report_files=$(find "$experiment_dir" -name "*_report.json" 2>/dev/null | wc -l)
                        
                        if [ $model_files -gt 0 ] || [ $report_files -gt 0 ]; then
                            completed_experiments=$((completed_experiments + 1))
                        fi
                    fi
                done
            done
        done
    done
    
    echo "$completed_experiments $total_experiments"
}

# Function to check if dataset needs processing
should_process_dataset() {
    local dataset_name=$1
    local progress=($(get_dataset_progress "$dataset_name"))
    local completed=${progress[0]}
    local total=${progress[1]}
    
    if [ $completed -eq $total ]; then
        log_message "RESUME" "Dataset $dataset_name is complete ($completed/$total experiments)"
        return 1  # Don't process
    else
        log_message "RESUME" "Dataset $dataset_name needs processing ($completed/$total experiments completed)"
        return 0  # Process it
    fi
}

# Function to save progress
save_progress() {
    local current_dataset=$1
    local total_datasets=$2
    local successful=$3
    local failed=$4
    local current_dataset_name=$5
    
    {
        echo "PROGRESS_TIMESTAMP=\"$(date '+%Y-%m-%d %H:%M:%S')\""
        echo "CURRENT_DATASET_INDEX=$current_dataset"
        echo "CURRENT_DATASET_NAME=\"$current_dataset_name\""
        echo "TOTAL_DATASETS=$total_datasets"
        echo "SUCCESSFUL_DATASETS=$successful"
        echo "FAILED_DATASETS=$failed"
    } > "$PROGRESS_FILE"
}

# Function to load previous progress
load_progress() {
    if [ -f "$PROGRESS_FILE" ]; then
        source "$PROGRESS_FILE" 2>/dev/null
        echo "$CURRENT_DATASET_INDEX $TOTAL_DATASETS $SUCCESSFUL_DATASETS $FAILED_DATASETS $CURRENT_DATASET_NAME"
    else
        echo "0 0 0 0 \"\""
    fi
}

# Create output directory
mkdir -p "$OUTPUT_BASE_DIR"

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

# Check for resume capability
resume_mode=false
previous_progress=($(load_progress))
prev_current=${previous_progress[0]}
prev_total=${previous_progress[1]} 
prev_successful=${previous_progress[2]}
prev_failed=${previous_progress[3]}
prev_dataset_name="${previous_progress[4]}"

if [ -f "$LOG_FILE" ] && [ -f "$PROGRESS_FILE" ]; then
    resume_mode=true
    log_message "RESUME" "Previous session found, will resume from where left off"
    log_message "RESUME" "Previous progress: $prev_current/$prev_total datasets, last working on: $prev_dataset_name"
else
    log_message "INFO" "Starting fresh run"
fi

# Initialize or append to log file
if [ "$resume_mode" = false ]; then
    echo "Batch Run Started: $(date)" > "$LOG_FILE"
    echo "Datasets Directory: $DATASETS_DIR" >> "$LOG_FILE"
    echo "Output Directory: $OUTPUT_BASE_DIR" >> "$LOG_FILE"
    echo "Run Experiments Script: $RUN_EXPERIMENTS_SCRIPT" >> "$LOG_FILE"
    echo "=======================================" >> "$LOG_FILE"
else
    echo "" >> "$LOG_FILE"
    echo "RESUME SESSION: $(date)" >> "$LOG_FILE"
    echo "Resuming from dataset index: $prev_current" >> "$LOG_FILE"
    echo "Last dataset being processed: $prev_dataset_name" >> "$LOG_FILE"
    echo "=======================================" >> "$LOG_FILE"
fi

log_message "INFO" "Found ${#datasets[@]} datasets to process"
log_message "INFO" "Starting batch processing..."

# Process each dataset
total_datasets=${#datasets[@]}
current_dataset=${prev_current:-0}
successful_datasets=${prev_successful:-0}
failed_datasets=${prev_failed:-0}

# Set up signal handling for graceful interruption
cleanup() {
    echo ""
    log_message "WARNING" "Script interrupted by user"
    log_message "INFO" "Progress saved. You can resume by running this script again."
    
    # Create interruption summary
    interruption_summary="$OUTPUT_BASE_DIR/interruption_summary.txt"
    {
        echo "Script Interrupted: $(date)"
        echo "================================"
        echo "Progress when interrupted:"
        echo "  Total datasets: $total_datasets"
        echo "  Processed: $current_dataset"
        echo "  Successful: $successful_datasets"
        echo "  Failed: $failed_datasets"
        echo "  Remaining: $((total_datasets - current_dataset))"
        echo ""
        echo "To resume, run the script again:"
        echo "$0"
        echo ""
        echo "The script will automatically resume individual experiments"
        echo "within each dataset where they left off."
    } > "$interruption_summary"
    
    log_message "INFO" "Interruption summary saved to: $interruption_summary"
    exit 130
}

trap cleanup INT TERM

# Start from where we left off
for (( i=current_dataset; i<total_datasets; i++ )); do
    dataset_path="${datasets[$i]}"
    dataset_name=$(basename "$dataset_path")
    current_dataset=$((i + 1))
    
    echo ""
    log_message "DATASET" "Processing dataset $current_dataset/$total_datasets: $dataset_name"
    
    # Save progress before starting each dataset
    save_progress "$current_dataset" "$total_datasets" "$successful_datasets" "$failed_datasets" "$dataset_name"
    
    # Check if this dataset needs processing (has incomplete experiments)
    if ! should_process_dataset "$dataset_name"; then
        log_message "DATASET" "Skipping fully completed dataset: $dataset_name"
        successful_datasets=$((successful_datasets + 1))
        continue
    fi
    
    # Create a temporary copy of run_all_experiments.sh with modified OUTPUT_BASE_DIR
    temp_script="/tmp/run_experiments_${dataset_name}.sh"
    cp "$RUN_EXPERIMENTS_SCRIPT" "$temp_script"
    
    # Modify the output directory for this specific dataset
    output_dir_name="rgb_$(echo "$dataset_name" | sed 's/dataset_rgb_//')_kbest"
    sed -i "s|OUTPUT_BASE_DIR=\"\$SCRIPT_DIR/rgb_poisson100_kbest\"|OUTPUT_BASE_DIR=\"$OUTPUT_BASE_DIR/$output_dir_name\"|g" "$temp_script"
    
    # Make temporary script executable
    chmod +x "$temp_script"
    
    # Run experiments for this dataset (will automatically skip completed experiments)
    start_time=$(date +%s)
    
    if "$temp_script" "$dataset_path" >> "$LOG_FILE" 2>&1; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        # Get final progress for this dataset
        local progress=($(get_dataset_progress "$dataset_name"))
        local completed=${progress[0]}
        local total=${progress[1]}
        
        log_message "INFO" "Completed dataset $dataset_name in ${duration}s ($completed/$total experiments)"
        successful_datasets=$((successful_datasets + 1))
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        log_message "ERROR" "Failed to process dataset $dataset_name after ${duration}s"
        failed_datasets=$((failed_datasets + 1))
    fi
    
    # Clean up temporary script
    rm -f "$temp_script"
    
    # Update progress after each dataset
    save_progress "$current_dataset" "$total_datasets" "$successful_datasets" "$failed_datasets" "$dataset_name"
done

# Generate final summary
echo "" | tee -a "$LOG_FILE"
log_message "INFO" "Batch processing completed!"
log_message "INFO" "Total datasets: $total_datasets"
log_message "INFO" "Successful: $successful_datasets"
log_message "INFO" "Failed: $failed_datasets"
log_message "INFO" "Overall success rate: $(( successful_datasets * 100 / total_datasets ))%"
log_message "INFO" "Results saved in: $OUTPUT_BASE_DIR"
log_message "INFO" "Full log available at: $LOG_FILE"

# Create detailed summary file
summary_file="$OUTPUT_BASE_DIR/batch_summary.txt"
{
    echo "Batch Processing Summary"
    echo "======================="
    echo "Completed: $(date)"
    if [ "$resume_mode" = true ]; then
        echo "Mode: RESUMED"
        echo "Resumed from dataset: $prev_current/$prev_total"
    else
        echo "Mode: FRESH RUN"
    fi
    echo "Total datasets: $total_datasets"
    echo "Successful: $successful_datasets"
    echo "Failed: $failed_datasets"
    echo "Success rate: $(( successful_datasets * 100 / total_datasets ))%"
    echo ""
    echo "Dataset progress details:"
    for dataset_path in "${datasets[@]}"; do
        dataset_name=$(basename "$dataset_path")
        progress=($(get_dataset_progress "$dataset_name"))
        completed=${progress[0]}
        total=${progress[1]}
        percentage=$((completed * 100 / total))
        
        if [ $completed -eq $total ]; then
            echo "  ✓ $dataset_name: $completed/$total experiments ($percentage%)"
        else
            echo "  ○ $dataset_name: $completed/$total experiments ($percentage%)"
        fi
    done
} > "$summary_file"

log_message "INFO" "Summary saved to: $summary_file"

# Clean up progress file on successful completion
rm -f "$PROGRESS_FILE"

# Exit with appropriate code
if [ $failed_datasets -eq 0 ]; then
    exit 0
else
    exit 1
fi