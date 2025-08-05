#!/usr/bin/env bash
#
# HRM Training Recovery Script
# Version: 1.0 (2025-08-05)
# Owner: @zshinyg
#
# This script automatically recovers from training failures and system issues
# for the HRM vs Transformer hypothesis validation. It's designed to be robust
# for unattended operation, with special optimizations for M1 Macs.
#
# Features:
# - Automatic detection of training failures
# - Checkpoint validation and selection
# - Memory cleanup and optimization
# - M1-specific MPS cache handling
# - Retry logic with exponential backoff
# - System requirement validation
# - Comprehensive logging and notifications
#

# ===== Configuration =====
# Set strict error handling
set -eo pipefail

# Base directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${BASE_DIR}/logs"
CHECKPOINT_DIR="${BASE_DIR}/checkpoints"
CONFIG_DIR="${BASE_DIR}/configs"

# Default configuration
DEFAULT_CONFIG="configs/m1_optimized_training.yaml"
MAX_RETRIES=3
BACKOFF_FACTOR=2
INITIAL_WAIT=60  # seconds
ALERT_WEBHOOK=""  # Set this to your webhook URL for notifications
HEARTBEAT_FILE="${LOG_DIR}/heartbeat.txt"
RECOVERY_LOG="${LOG_DIR}/recovery.log"
ALERT_LOG="${LOG_DIR}/alerts.log"

# M1 Mac optimization settings
MPS_HIGH_WATERMARK="0.9"
MPS_ENABLE_FALLBACK="1"

# ===== Utility Functions =====

# Logging function
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] [$level] $message" | tee -a "$RECOVERY_LOG"
    
    # Also log errors and critical messages to alert log
    if [[ "$level" == "ERROR" || "$level" == "CRITICAL" ]]; then
        echo "[$timestamp] [$level] $message" >> "$ALERT_LOG"
    fi
}

# Create required directories
setup_directories() {
    mkdir -p "$LOG_DIR"
    mkdir -p "$CHECKPOINT_DIR"
    touch "$RECOVERY_LOG"
    touch "$ALERT_LOG"
    
    log "INFO" "Recovery directories initialized"
}

# Send alert notification
send_alert() {
    local level="$1"
    local message="$2"
    
    log "$level" "$message"
    
    # Send to webhook if configured
    if [[ -n "$ALERT_WEBHOOK" ]]; then
        local payload="{\"level\":\"$level\",\"message\":\"$message\",\"timestamp\":\"$(date "+%Y-%m-%d %H:%M:%S")\"}"
        curl -s -X POST -H "Content-Type: application/json" -d "$payload" "$ALERT_WEBHOOK" || true
    fi
}

# Check if a process is running
is_process_running() {
    local pid="$1"
    if [[ -z "$pid" ]]; then
        return 1
    fi
    
    if ps -p "$pid" > /dev/null; then
        return 0
    else
        return 1
    fi
}

# Check if training process is active via heartbeat
is_training_active() {
    if [[ ! -f "$HEARTBEAT_FILE" ]]; then
        log "INFO" "Heartbeat file not found, training not active"
        return 1
    fi
    
    local heartbeat_age=$(( $(date +%s) - $(stat -f %m "$HEARTBEAT_FILE") ))
    if (( heartbeat_age > 300 )); then  # 5 minutes
        log "WARNING" "Heartbeat file stale (${heartbeat_age}s old), training likely inactive"
        return 1
    fi
    
    # Check if the PID in the heartbeat file is actually running
    local pid=$(grep -o '"pid":[0-9]*' "$HEARTBEAT_FILE" | cut -d':' -f2)
    if ! is_process_running "$pid"; then
        log "WARNING" "Training process (PID $pid) not running"
        return 1
    fi
    
    return 0
}

# Kill any existing training processes
kill_training_processes() {
    log "INFO" "Checking for existing training processes"
    
    # Find and kill any python processes running train_codegen.py
    local pids=$(pgrep -f "python.*train_codegen.py" || echo "")
    if [[ -n "$pids" ]]; then
        log "WARNING" "Found existing training processes: $pids"
        for pid in $pids; do
            log "WARNING" "Killing process $pid"
            kill -15 "$pid" 2>/dev/null || true
            sleep 2
            # Force kill if still running
            if is_process_running "$pid"; then
                log "WARNING" "Process $pid didn't terminate gracefully, force killing"
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
    else
        log "INFO" "No existing training processes found"
    fi
}

# Clean up memory and temporary files
cleanup_system() {
    log "INFO" "Performing system cleanup"
    
    # Clear disk cache if we have sudo (optional, be careful)
    # if sudo -n true 2>/dev/null; then
    #     log "INFO" "Clearing disk cache"
    #     sudo purge
    # fi
    
    # Clean up temporary files
    find /tmp -name "torch_*" -type d -mmin +60 -exec rm -rf {} \; 2>/dev/null || true
    find /tmp -name "pytest_*" -type d -mmin +60 -exec rm -rf {} \; 2>/dev/null || true
    
    # Clean up any core dumps
    find "$BASE_DIR" -name "core.*" -delete 2>/dev/null || true
    
    log "INFO" "System cleanup completed"
}

# Find the most recent valid checkpoint
find_latest_checkpoint() {
    log "INFO" "Finding latest valid checkpoint"
    
    # Look for checkpoint files
    local checkpoints=($(find "$CHECKPOINT_DIR" -name "*.pt" -type f -print0 | xargs -0 ls -t))
    
    if [[ ${#checkpoints[@]} -eq 0 ]]; then
        log "WARNING" "No checkpoints found"
        return 1
    fi
    
    # Try each checkpoint until we find a valid one
    for checkpoint in "${checkpoints[@]}"; do
        log "INFO" "Validating checkpoint: $checkpoint"
        
        # Check file size (should be at least 100MB for a valid model)
        local size=$(stat -f %z "$checkpoint")
        if (( size < 100000000 )); then
            log "WARNING" "Checkpoint too small, might be corrupted: $checkpoint ($size bytes)"
            continue
        fi
        
        # Try to load the checkpoint with a simple script
        if python -c "
import torch
try:
    checkpoint = torch.load('$checkpoint', map_location='cpu')
    print('Valid keys:', ', '.join(checkpoint.keys()))
    exit(0)
except Exception as e:
    print('Error:', e)
    exit(1)
" > /dev/null 2>&1; then
            log "INFO" "Found valid checkpoint: $checkpoint"
            echo "$checkpoint"
            return 0
        else
            log "WARNING" "Checkpoint validation failed: $checkpoint"
        fi
    done
    
    log "ERROR" "No valid checkpoints found"
    return 1
}

# Check if the system meets requirements
validate_system_requirements() {
    log "INFO" "Validating system requirements"
    
    # Check Python version
    if ! python --version | grep -q "Python 3"; then
        log "ERROR" "Python 3 is required"
        return 1
    fi
    
    # Check PyTorch installation
    if ! python -c "import torch; print(f'PyTorch {torch.__version__}')" > /dev/null 2>&1; then
        log "ERROR" "PyTorch not installed or not working"
        return 1
    fi
    
    # Check for M1 Mac and MPS availability
    if [[ "$(uname -m)" == "arm64" && "$(uname -s)" == "Darwin" ]]; then
        log "INFO" "Detected Apple Silicon (M1/M2/M3)"
        
        if ! python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')" | grep -q "MPS available: True"; then
            log "WARNING" "MPS not available on Apple Silicon, performance will be degraded"
            send_alert "WARNING" "MPS not available on Apple Silicon, performance will be degraded"
        fi
    fi
    
    # Check disk space (need at least 5GB free)
    local free_space=$(df -k . | awk 'NR==2 {print $4}')
    if (( free_space < 5000000 )); then
        log "ERROR" "Insufficient disk space: $(( free_space / 1000000 ))GB free, need at least 5GB"
        return 1
    fi
    
    # Check memory (need at least 8GB)
    local total_mem=$(sysctl -n hw.memsize 2>/dev/null || free -b | grep Mem | awk '{print $2}')
    if (( total_mem < 8000000000 )); then
        log "ERROR" "Insufficient memory: $(( total_mem / 1000000000 ))GB, need at least 8GB"
        return 1
    }
    
    log "INFO" "System requirements validated"
    return 0
}

# Set up environment variables for optimal training
setup_environment() {
    log "INFO" "Setting up training environment"
    
    # General settings
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS=$(( $(sysctl -n hw.ncpu 2>/dev/null || nproc) / 2 ))
    
    # M1 Mac specific settings
    if [[ "$(uname -m)" == "arm64" && "$(uname -s)" == "Darwin" ]]; then
        log "INFO" "Configuring M1 Mac specific environment"
        export PYTORCH_ENABLE_MPS_FALLBACK="$MPS_ENABLE_FALLBACK"
        export PYTORCH_MPS_HIGH_WATERMARK_RATIO="$MPS_HIGH_WATERMARK"
        
        # Clear MPS cache
        python -c "
import torch
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
    print('MPS cache cleared')
" || true
    fi
    
    log "INFO" "Environment configured"
}

# Run a command with retry logic and exponential backoff
run_with_retry() {
    local cmd="$1"
    local retry_count=0
    local wait_time=$INITIAL_WAIT
    
    while (( retry_count < MAX_RETRIES )); do
        log "INFO" "Running command (attempt $((retry_count + 1))/$MAX_RETRIES): $cmd"
        
        if eval "$cmd"; then
            log "INFO" "Command succeeded"
            return 0
        else
            local exit_code=$?
            retry_count=$((retry_count + 1))
            
            if (( retry_count >= MAX_RETRIES )); then
                log "ERROR" "Command failed after $MAX_RETRIES attempts: $cmd (exit code: $exit_code)"
                return $exit_code
            fi
            
            log "WARNING" "Command failed (exit code: $exit_code), retrying in $wait_time seconds"
            sleep $wait_time
            wait_time=$((wait_time * BACKOFF_FACTOR))
        fi
    done
}

# Start training from checkpoint
start_training() {
    local checkpoint="$1"
    local config_file="${2:-$DEFAULT_CONFIG}"
    
    log "INFO" "Starting training from checkpoint: $checkpoint"
    log "INFO" "Using config: $config_file"
    
    # Create the command
    local cmd="cd \"$BASE_DIR\" && python scripts/train_codegen.py"
    
    if [[ -n "$checkpoint" ]]; then
        cmd="$cmd --resume \"$checkpoint\""
    fi
    
    cmd="$cmd --config \"$config_file\""
    
    # Add logging redirection
    local timestamp=$(date "+%Y%m%d_%H%M%S")
    local log_file="${LOG_DIR}/training_${timestamp}.log"
    cmd="$cmd > \"$log_file\" 2>&1 &"
    
    # Run the command
    eval "$cmd"
    local pid=$!
    
    log "INFO" "Training started with PID $pid"
    log "INFO" "Log file: $log_file"
    
    # Create a PID file
    echo "$pid" > "${LOG_DIR}/training.pid"
    
    # Wait a bit to ensure process started successfully
    sleep 5
    if ! is_process_running "$pid"; then
        log "ERROR" "Training process failed to start or terminated immediately"
        send_alert "ERROR" "Training process failed to start or terminated immediately"
        return 1
    fi
    
    send_alert "INFO" "Training restarted successfully from checkpoint: $checkpoint"
    return 0
}

# Check for common errors in log files and suggest fixes
analyze_failure() {
    log "INFO" "Analyzing training failure"
    
    # Find the most recent log file
    local recent_logs=($(find "$LOG_DIR" -name "training_*.log" -type f -print0 | xargs -0 ls -t | head -5))
    
    if [[ ${#recent_logs[@]} -eq 0 ]]; then
        log "WARNING" "No recent log files found for analysis"
        return 1
    fi
    
    local latest_log="${recent_logs[0]}"
    log "INFO" "Analyzing log file: $latest_log"
    
    # Check for common errors
    if grep -q "CUDA out of memory" "$latest_log"; then
        log "WARNING" "CUDA out of memory error detected"
        send_alert "WARNING" "CUDA out of memory error detected - reducing batch size"
        # Create a temporary config with smaller batch size
        python -c "
import yaml
config_path = '$DEFAULT_CONFIG'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
# Reduce batch size
if 'data' in config and 'batch_size' in config['data']:
    config['data']['batch_size'] = max(1, config['data']['batch_size'] // 2)
    print(f\"Reduced batch size to {config['data']['batch_size']}\")
# Increase gradient accumulation
if 'training' in config and 'gradient_accumulation_steps' in config['training']:
    config['training']['gradient_accumulation_steps'] *= 2
    print(f\"Increased gradient accumulation to {config['training']['gradient_accumulation_steps']}\")
# Save modified config
modified_config_path = '$CONFIG_DIR/reduced_batch.yaml'
with open(modified_config_path, 'w') as f:
    yaml.dump(config, f)
print(f\"Created modified config at {modified_config_path}\")
"
        return "$CONFIG_DIR/reduced_batch.yaml"
    elif grep -q "RuntimeError: MPS backend out of memory" "$latest_log"; then
        log "WARNING" "MPS out of memory error detected"
        send_alert "WARNING" "MPS out of memory error detected - reducing batch size"
        # Create a temporary config with smaller batch size
        python -c "
import yaml
config_path = '$DEFAULT_CONFIG'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
# Reduce batch size
if 'data' in config and 'batch_size' in config['data']:
    config['data']['batch_size'] = max(1, config['data']['batch_size'] // 2)
    print(f\"Reduced batch size to {config['data']['batch_size']}\")
# Increase gradient accumulation
if 'training' in config and 'gradient_accumulation_steps' in config['training']:
    config['training']['gradient_accumulation_steps'] *= 2
    print(f\"Increased gradient accumulation to {config['training']['gradient_accumulation_steps']}\")
# Enable memory optimization
if 'memory' not in config:
    config['memory'] = {}
config['memory']['optimize_memory_usage'] = True
config['memory']['empty_cache_freq'] = 50
# Save modified config
modified_config_path = '$CONFIG_DIR/reduced_batch_mps.yaml'
with open(modified_config_path, 'w') as f:
    yaml.dump(config, f)
print(f\"Created modified config at {modified_config_path}\")
"
        return "$CONFIG_DIR/reduced_batch_mps.yaml"
    elif grep -q "nan" "$latest_log" || grep -q "inf" "$latest_log"; then
        log "WARNING" "NaN/Inf values detected in training"
        send_alert "WARNING" "NaN/Inf values detected - reducing learning rate"
        # Create a temporary config with lower learning rate
        python -c "
import yaml
config_path = '$DEFAULT_CONFIG'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
# Reduce learning rate
if 'training' in config and 'optimizer' in config['training'] and 'lr' in config['training']['optimizer']:
    config['training']['optimizer']['lr'] /= 5.0
    print(f\"Reduced learning rate to {config['training']['optimizer']['lr']}\")
# Enable gradient clipping
if 'training' in config:
    config['training']['max_grad_norm'] = 1.0
    print(f\"Enabled gradient clipping with max_grad_norm=1.0\")
# Save modified config
modified_config_path = '$CONFIG_DIR/reduced_lr.yaml'
with open(modified_config_path, 'w') as f:
    yaml.dump(config, f)
print(f\"Created modified config at {modified_config_path}\")
"
        return "$CONFIG_DIR/reduced_lr.yaml"
    elif grep -q "ImportError: No module named" "$latest_log" || grep -q "ModuleNotFoundError" "$latest_log"; then
        log "WARNING" "Missing Python module detected"
        send_alert "WARNING" "Missing Python module - attempting to reinstall requirements"
        run_with_retry "cd \"$BASE_DIR\" && pip install -r requirements.txt"
    fi
    
    return 0
}

# ===== Main Recovery Process =====

main() {
    log "INFO" "=== HRM Training Recovery Process Started ==="
    
    # Setup
    setup_directories
    
    # Check if training is already active
    if is_training_active; then
        log "INFO" "Training is already active, no recovery needed"
        exit 0
    fi
    
    # Kill any zombie training processes
    kill_training_processes
    
    # Clean up system
    cleanup_system
    
    # Validate system requirements
    if ! validate_system_requirements; then
        log "CRITICAL" "System requirements not met, recovery aborted"
        send_alert "CRITICAL" "System requirements not met, recovery aborted"
        exit 1
    fi
    
    # Analyze previous failure
    local custom_config=$(analyze_failure)
    
    # Find latest checkpoint
    local checkpoint=$(find_latest_checkpoint)
    local checkpoint_status=$?
    
    # Setup environment
    setup_environment
    
    # Start training
    if [[ $checkpoint_status -eq 0 ]]; then
        if [[ -n "$custom_config" ]]; then
            log "INFO" "Using custom config based on failure analysis: $custom_config"
            start_training "$checkpoint" "$custom_config"
        else
            start_training "$checkpoint"
        fi
    else
        log "WARNING" "No valid checkpoint found, starting from scratch"
        send_alert "WARNING" "No valid checkpoint found, starting from scratch"
        
        if [[ -n "$custom_config" ]]; then
            start_training "" "$custom_config"
        else
            start_training ""
        fi
    fi
    
    log "INFO" "=== HRM Training Recovery Process Completed ==="
}

# Run the main function
main "$@"
