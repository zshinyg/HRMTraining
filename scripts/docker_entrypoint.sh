#!/bin/bash
# Docker entrypoint script for HRM-CodeGen
# Handles various execution modes and operational aspects of the containerized application
# Supports both CPU and GPU environments

set -e

# =====================================================================
# Environment variables and defaults
# =====================================================================
: "${MODE:=serve}"                         # Default mode: serve
: "${LOG_LEVEL:=info}"                     # Default log level: info
: "${PORT:=8000}"                          # Default port: 8000
: "${WORKERS:=4}"                          # Default number of workers
: "${MAX_BATCH_SIZE:=16}"                  # Default max batch size
: "${MAX_SEQUENCE_LENGTH:=2048}"           # Default max sequence length
: "${MODEL_DIR:=/app/models}"              # Default model directory
: "${DATA_DIR:=/app/data}"                 # Default data directory
: "${CONFIG_DIR:=/app/configs}"            # Default config directory
: "${LOG_DIR:=/app/logs}"                  # Default log directory
: "${CHECKPOINT_DIR:=/app/checkpoints}"    # Default checkpoint directory
: "${CONFIG_FILE:=}"                       # Default config file (empty)
: "${ENVIRONMENT:=production}"             # Default environment: production
: "${ENABLE_MONITORING:=true}"             # Default: enable monitoring
: "${HEALTH_CHECK_PATH:=/health}"          # Default health check path
: "${TIMEOUT:=60}"                         # Default timeout in seconds
: "${DEBUG:=false}"                        # Default debug mode: disabled
: "${CHECKPOINT:=}"                        # Default checkpoint (empty)
: "${ENABLE_WANDB:=false}"                 # Default W&B: disabled
: "${WANDB_PROJECT:=hrm-codegen}"          # Default W&B project
: "${WANDB_ENTITY:=}"                      # Default W&B entity (empty)
: "${ENABLE_PROFILING:=false}"             # Default profiling: disabled

# =====================================================================
# Helper functions
# =====================================================================

# Function to log messages with timestamp and log level
log() {
    local level=$1
    local message=$2
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    
    # Map log levels to numeric values for filtering
    local level_value=0
    case $LOG_LEVEL in
        debug) level_value=0 ;;
        info) level_value=1 ;;
        warning) level_value=2 ;;
        error) level_value=3 ;;
        *) level_value=1 ;;  # Default to info
    esac
    
    local msg_level_value=0
    case $level in
        DEBUG) msg_level_value=0 ;;
        INFO) msg_level_value=1 ;;
        WARNING) msg_level_value=2 ;;
        ERROR) msg_level_value=3 ;;
        *) msg_level_value=1 ;;  # Default to info
    esac
    
    # Only log if message level is >= configured log level
    if [ $msg_level_value -ge $level_value ]; then
        # Color the output in development mode
        if [ "$ENVIRONMENT" = "development" ]; then
            case $level in
                DEBUG) echo -e "\033[36m[$timestamp] [DEBUG] $message\033[0m" ;;
                INFO) echo -e "\033[32m[$timestamp] [INFO] $message\033[0m" ;;
                WARNING) echo -e "\033[33m[$timestamp] [WARNING] $message\033[0m" ;;
                ERROR) echo -e "\033[31m[$timestamp] [ERROR] $message\033[0m" ;;
                *) echo "[$timestamp] [$level] $message" ;;
            esac
        else
            echo "[$timestamp] [$level] $message"
        fi
    fi
    
    # In production, also log to file
    if [ "$ENVIRONMENT" = "production" ]; then
        echo "[$timestamp] [$level] $message" >> "$LOG_DIR/container.log"
    fi
}

# Function to check if a directory exists and is writable
check_directory() {
    local dir=$1
    local dir_name=$2
    
    if [ ! -d "$dir" ]; then
        log "WARNING" "$dir_name directory ($dir) does not exist, creating it"
        mkdir -p "$dir" || { log "ERROR" "Failed to create $dir_name directory ($dir)"; return 1; }
    fi
    
    if [ ! -w "$dir" ]; then
        log "ERROR" "$dir_name directory ($dir) is not writable"
        return 1
    fi
    
    return 0
}

# Function to check for GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi &> /dev/null
        if [ $? -eq 0 ]; then
            log "INFO" "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
            return 0
        else
            log "WARNING" "nvidia-smi command failed, GPU may not be properly configured"
            return 1
        fi
    else
        log "WARNING" "nvidia-smi not found, running without GPU"
        return 1
    fi
}

# Function to check Python dependencies
check_dependencies() {
    log "INFO" "Checking Python dependencies..."
    
    # Check if key packages are installed
    python -c "import torch" 2>/dev/null || { log "ERROR" "PyTorch is not installed"; return 1; }
    python -c "import transformers" 2>/dev/null || { log "ERROR" "Transformers is not installed"; return 1; }
    
    # Check PyTorch CUDA availability if GPU is detected
    if check_gpu; then
        if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
            log "INFO" "PyTorch CUDA is available"
            export CUDA_AVAILABLE=true
        else
            log "WARNING" "PyTorch CUDA is not available despite GPU being detected"
            export CUDA_AVAILABLE=false
        fi
    else
        log "INFO" "Running in CPU mode"
        export CUDA_AVAILABLE=false
    fi
    
    return 0
}

# Function to validate environment
validate_environment() {
    log "INFO" "Validating environment..."
    
    # Check required directories
    check_directory "$MODEL_DIR" "Model" || return 1
    check_directory "$DATA_DIR" "Data" || return 1
    check_directory "$CONFIG_DIR" "Config" || return 1
    check_directory "$LOG_DIR" "Log" || return 1
    check_directory "$CHECKPOINT_DIR" "Checkpoint" || return 1
    
    # Check dependencies
    check_dependencies || return 1
    
    # Check for config file if specified
    if [ -n "$CONFIG_FILE" ]; then
        if [ ! -f "$CONFIG_FILE" ]; then
            log "ERROR" "Config file not found: $CONFIG_FILE"
            return 1
        fi
        log "INFO" "Using config file: $CONFIG_FILE"
    fi
    
    # Check for checkpoint if specified
    if [ -n "$CHECKPOINT" ]; then
        if [ ! -f "$CHECKPOINT" ]; then
            log "ERROR" "Checkpoint file not found: $CHECKPOINT"
            return 1
        fi
        log "INFO" "Using checkpoint: $CHECKPOINT"
    fi
    
    log "INFO" "Environment validation completed successfully"
    return 0
}

# Function to set up signal handlers for graceful shutdown
setup_signal_handlers() {
    log "INFO" "Setting up signal handlers for graceful shutdown"
    
    # Variables to track shutdown state
    export SHUTDOWN_REQUESTED=false
    export SHUTDOWN_SIGNAL=""
    
    # Function to handle shutdown signals
    shutdown_handler() {
        local signal=$1
        export SHUTDOWN_SIGNAL=$signal
        export SHUTDOWN_REQUESTED=true
        
        log "INFO" "Received signal $signal, initiating graceful shutdown..."
        
        # If there's a PID file, get the main process PID and forward the signal
        if [ -f "/tmp/main_process.pid" ]; then
            local pid=$(cat /tmp/main_process.pid)
            if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
                log "INFO" "Forwarding signal $signal to main process (PID: $pid)"
                kill -$signal $pid
            fi
        fi
        
        # For SIGTERM and SIGINT, we'll let the main handler do its job
        # For SIGKILL, we can't catch it anyway
        if [ "$signal" = "SIGTERM" ] || [ "$signal" = "SIGINT" ]; then
            # Wait for the main process to finish (max 30 seconds)
            local timeout=30
            local count=0
            while [ -f "/tmp/main_process.pid" ] && [ $count -lt $timeout ]; do
                sleep 1
                count=$((count + 1))
            done
            
            if [ $count -ge $timeout ]; then
                log "WARNING" "Timeout waiting for main process to exit, forcing shutdown"
                exit 1
            else
                log "INFO" "Main process exited cleanly"
                exit 0
            fi
        fi
    }
    
    # Set up trap handlers for different signals
    trap 'shutdown_handler SIGTERM' TERM
    trap 'shutdown_handler SIGINT' INT
    trap 'shutdown_handler SIGHUP' HUP
}

# Function to check health of the application
check_health() {
    log "DEBUG" "Performing health check..."
    
    # Check if the main process is running
    if [ -f "/tmp/main_process.pid" ]; then
        local pid=$(cat /tmp/main_process.pid)
        if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
            log "DEBUG" "Main process (PID: $pid) is running"
        else
            log "ERROR" "Main process is not running"
            return 1
        fi
    else
        log "ERROR" "Main process PID file not found"
        return 1
    fi
    
    # If in serve mode, check if the API is responding
    if [ "$MODE" = "serve" ]; then
        if curl -s "http://localhost:$PORT$HEALTH_CHECK_PATH" | grep -q "ok"; then
            log "DEBUG" "API health check passed"
        else
            log "ERROR" "API health check failed"
            return 1
        fi
    fi
    
    # Check system resources
    local mem_usage=$(free -m | awk 'NR==2{printf "%.2f%%", $3*100/$2}')
    local disk_usage=$(df -h / | awk 'NR==2{print $5}')
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')
    
    log "DEBUG" "System health: Memory: $mem_usage, Disk: $disk_usage, CPU: $cpu_usage%"
    
    # Check if resources are critically low
    if [ "${mem_usage%.*}" -gt 95 ]; then
        log "WARNING" "Memory usage is critically high: $mem_usage"
    fi
    
    if [ "${disk_usage%\%}" -gt 90 ]; then
        log "WARNING" "Disk usage is critically high: $disk_usage"
    fi
    
    return 0
}

# Function to set up monitoring if enabled
setup_monitoring() {
    if [ "$ENABLE_MONITORING" = "true" ]; then
        log "INFO" "Setting up monitoring..."
        
        # Create monitoring directory if it doesn't exist
        mkdir -p "$LOG_DIR/metrics"
        
        # Start background process to collect metrics periodically
        (
            while true; do
                if [ "$SHUTDOWN_REQUESTED" = "true" ]; then
                    break
                fi
                
                # Collect system metrics
                date +"%Y-%m-%d %H:%M:%S" > "$LOG_DIR/metrics/system.txt"
                echo "Memory:" >> "$LOG_DIR/metrics/system.txt"
                free -h >> "$LOG_DIR/metrics/system.txt"
                echo "Disk:" >> "$LOG_DIR/metrics/system.txt"
                df -h >> "$LOG_DIR/metrics/system.txt"
                echo "CPU:" >> "$LOG_DIR/metrics/system.txt"
                top -bn1 | head -n 5 >> "$LOG_DIR/metrics/system.txt"
                
                # If GPU is available, collect GPU metrics
                if [ "$CUDA_AVAILABLE" = "true" ]; then
                    echo "GPU:" >> "$LOG_DIR/metrics/system.txt"
                    nvidia-smi >> "$LOG_DIR/metrics/system.txt"
                fi
                
                # Sleep for 60 seconds
                sleep 60
            done
        ) &
        
        log "INFO" "Monitoring set up successfully"
    else
        log "INFO" "Monitoring is disabled"
    fi
}

# Function to initialize Weights & Biases integration
setup_wandb() {
    if [ "$ENABLE_WANDB" = "true" ]; then
        log "INFO" "Setting up Weights & Biases integration..."
        
        # Check if WANDB_API_KEY is set
        if [ -z "$WANDB_API_KEY" ]; then
            log "WARNING" "WANDB_API_KEY is not set, W&B integration may not work properly"
        fi
        
        # Set W&B environment variables
        export WANDB_PROJECT="$WANDB_PROJECT"
        if [ -n "$WANDB_ENTITY" ]; then
            export WANDB_ENTITY="$WANDB_ENTITY"
        fi
        
        # Set W&B mode based on environment
        if [ "$ENVIRONMENT" = "development" ]; then
            export WANDB_MODE="dryrun"
        else
            export WANDB_MODE="online"
        fi
        
        log "INFO" "W&B integration set up successfully"
    else
        # Disable W&B
        export WANDB_MODE="disabled"
        log "INFO" "W&B integration is disabled"
    fi
}

# Function to load model and data
load_model_and_data() {
    log "INFO" "Loading model and data..."
    
    # Determine which model checkpoint to use
    local model_path=""
    if [ -n "$CHECKPOINT" ]; then
        model_path="$CHECKPOINT"
    else
        # Find the latest checkpoint in the checkpoint directory
        model_path=$(find "$CHECKPOINT_DIR" -name "*.pt" -type f -printf "%T@ %p\n" | sort -nr | head -n 1 | cut -d' ' -f2-)
        
        if [ -z "$model_path" ]; then
            log "WARNING" "No checkpoint found in $CHECKPOINT_DIR, will initialize model from scratch"
        else
            log "INFO" "Using latest checkpoint: $model_path"
        fi
    fi
    
    # Export the model path for the Python application
    export MODEL_PATH="$model_path"
    
    log "INFO" "Model and data loading completed"
}

# Function to run the application in the specified mode
run_application() {
    local mode=$1
    log "INFO" "Starting application in $mode mode..."
    
    # Create command based on mode
    local cmd=""
    case $mode in
        train)
            # Training mode
            cmd="python scripts/train.py"
            
            # Add configuration
            if [ -n "$CONFIG_FILE" ]; then
                cmd="$cmd --config $CONFIG_FILE"
            else
                cmd="$cmd --config $CONFIG_DIR/default.yaml"
            fi
            
            # Add checkpoint if specified
            if [ -n "$CHECKPOINT" ]; then
                cmd="$cmd --resume $CHECKPOINT"
            fi
            
            # Add other training parameters
            cmd="$cmd --data-path $DATA_DIR --out-dir $CHECKPOINT_DIR"
            
            # Enable mixed precision for GPU training
            if [ "$CUDA_AVAILABLE" = "true" ]; then
                cmd="$cmd --use-mixed-precision"
            fi
            ;;
            
        eval)
            # Evaluation mode
            cmd="python scripts/evaluate.py"
            
            # Add configuration
            if [ -n "$CONFIG_FILE" ]; then
                cmd="$cmd --config $CONFIG_FILE"
            else
                cmd="$cmd --config $CONFIG_DIR/default.yaml"
            fi
            
            # Add checkpoint
            if [ -n "$CHECKPOINT" ]; then
                cmd="$cmd --ckpt $CHECKPOINT"
            else
                log "ERROR" "Checkpoint is required for evaluation mode"
                return 1
            fi
            
            # Add evaluation parameters
            cmd="$cmd --split test --k 1 5 10"
            ;;
            
        benchmark)
            # Benchmarking mode
            cmd="python scripts/benchmark.py"
            
            # Add configuration
            if [ -n "$CONFIG_FILE" ]; then
                cmd="$cmd --config $CONFIG_FILE"
            else
                cmd="$cmd --config $CONFIG_DIR/default.yaml"
            fi
            
            # Add checkpoint if specified
            if [ -n "$CHECKPOINT" ]; then
                cmd="$cmd --ckpt $CHECKPOINT"
            fi
            
            # Add benchmarking parameters
            cmd="$cmd --output $LOG_DIR/benchmark_results.json"
            
            # Enable profiling if requested
            if [ "$ENABLE_PROFILING" = "true" ]; then
                cmd="$cmd --profile"
            fi
            ;;
            
        serve)
            # Serving mode (API)
            cmd="python -m uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers $WORKERS"
            
            # Add timeout
            cmd="$cmd --timeout $TIMEOUT"
            
            # Add log level
            case $LOG_LEVEL in
                debug) cmd="$cmd --log-level debug" ;;
                info) cmd="$cmd --log-level info" ;;
                warning) cmd="$cmd --log-level warning" ;;
                error) cmd="$cmd --log-level error" ;;
                *) cmd="$cmd --log-level info" ;;
            esac
            ;;
            
        shell)
            # Interactive shell for debugging
            log "INFO" "Starting interactive shell"
            cmd="bash"
            ;;
            
        *)
            # Unknown mode
            log "ERROR" "Unknown mode: $mode"
            log "INFO" "Available modes: train, eval, benchmark, serve, shell"
            return 1
            ;;
    esac
    
    # Add debug flag if enabled
    if [ "$DEBUG" = "true" ]; then
        cmd="$cmd --debug"
        
        # Enable core dumps in debug mode
        ulimit -c unlimited
        log "INFO" "Debug mode enabled with core dumps"
    fi
    
    # Log the command
    log "INFO" "Executing: $cmd"
    
    # Run the command and save its PID
    eval "$cmd" &
    local pid=$!
    echo $pid > /tmp/main_process.pid
    
    # Wait for the process to complete
    wait $pid
    local exit_code=$?
    
    # Remove PID file
    rm -f /tmp/main_process.pid
    
    # Log completion
    if [ $exit_code -eq 0 ]; then
        log "INFO" "Application completed successfully"
    else
        log "ERROR" "Application exited with code $exit_code"
    fi
    
    return $exit_code
}

# Function to print debug information
print_debug_info() {
    log "INFO" "Printing debug information..."
    
    # System information
    log "DEBUG" "System information:"
    uname -a
    
    # Environment variables
    log "DEBUG" "Environment variables:"
    env | sort
    
    # Python version
    log "DEBUG" "Python version:"
    python --version
    
    # PyTorch version and CUDA availability
    log "DEBUG" "PyTorch information:"
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
    
    # Disk space
    log "DEBUG" "Disk space:"
    df -h
    
    # Memory usage
    log "DEBUG" "Memory usage:"
    free -h
    
    # If GPU is available, show GPU information
    if command -v nvidia-smi &> /dev/null; then
        log "DEBUG" "GPU information:"
        nvidia-smi
    fi
    
    # Check directory permissions
    log "DEBUG" "Directory permissions:"
    ls -la /app
    ls -la "$MODEL_DIR"
    ls -la "$DATA_DIR"
    ls -la "$CONFIG_DIR"
    ls -la "$LOG_DIR"
    ls -la "$CHECKPOINT_DIR"
    
    log "INFO" "Debug information printed"
}

# =====================================================================
# Main execution
# =====================================================================

main() {
    # Print banner
    echo "======================================================================"
    echo "  HRM-CodeGen Docker Container"
    echo "  Mode: $MODE"
    echo "  Environment: $ENVIRONMENT"
    echo "======================================================================"
    
    # Initialize logging
    mkdir -p "$LOG_DIR"
    if [ "$ENVIRONMENT" = "production" ]; then
        # Rotate logs in production
        if [ -f "$LOG_DIR/container.log" ]; then
            mv "$LOG_DIR/container.log" "$LOG_DIR/container.log.$(date +%Y%m%d%H%M%S)"
            # Keep only the last 5 log files
            ls -t "$LOG_DIR"/container.log.* | tail -n +6 | xargs -r rm
        fi
        touch "$LOG_DIR/container.log"
    fi
    
    log "INFO" "Starting HRM-CodeGen container in $MODE mode (environment: $ENVIRONMENT)"
    
    # Print debug information if in debug mode
    if [ "$DEBUG" = "true" ]; then
        print_debug_info
    fi
    
    # Validate environment
    validate_environment || { log "ERROR" "Environment validation failed"; exit 1; }
    
    # Set up signal handlers
    setup_signal_handlers
    
    # Set up monitoring
    setup_monitoring
    
    # Set up W&B integration
    setup_wandb
    
    # Load model and data
    load_model_and_data
    
    # Run the application in the specified mode
    run_application "$MODE"
    exit_code=$?
    
    # Final log message
    if [ $exit_code -eq 0 ]; then
        log "INFO" "Container execution completed successfully"
    else
        log "ERROR" "Container execution failed with exit code $exit_code"
    fi
    
    exit $exit_code
}

# Run the main function
main
