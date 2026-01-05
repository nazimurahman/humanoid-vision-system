#!/bin/bash
# Entrypoint script for Humanoid Vision System

set -e

# Function to check GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "Checking GPU availability..."
        if nvidia-smi &> /dev/null; then
            echo "✅ GPU detected"
            return 0
        else
            echo "⚠️  GPU driver found but nvidia-smi failed"
            return 1
        fi
    else
        echo "❌ nvidia-smi not found. GPU may not be available."
        return 1
    fi
}

# Function to wait for dependencies
wait_for_deps() {
    local max_attempts=30
    local attempt=1
    
    echo "Waiting for dependencies to be ready..."
    
    # Wait for Redis if configured
    if [ -n "${REDIS_HOST}" ]; then
        until nc -z ${REDIS_HOST} ${REDIS_PORT:-6379} || [ ${attempt} -eq ${max_attempts} ]; do
            echo "Waiting for Redis... (attempt ${attempt}/${max_attempts})"
            sleep 2
            attempt=$((attempt + 1))
        done
        
        if [ ${attempt} -eq ${max_attempts} ]; then
            echo "Redis not available after ${max_attempts} attempts"
            exit 1
        fi
        echo "✅ Redis is ready"
    fi
}

# Function to start API server
start_api_server() {
    echo "Starting API server..."
    
    # Check if model exists
    if [ ! -f "${MODEL_PATH}" ]; then
        echo "❌ Model not found at ${MODEL_PATH}"
        exit 1
    fi
    
    # Start the API server
    exec python -m src.deployment.api_server \
        --model "${MODEL_PATH}" \
        --config "${CONFIG_PATH}" \
        --host "${HOST:-0.0.0.0}" \
        --port "${PORT:-8000}" \
        --workers "${WORKERS:-4}" \
        --log-level "${LOG_LEVEL:-INFO}"
}

# Function to start training
start_training() {
    echo "Starting training..."
    
    # Check if data directory exists
    if [ ! -d "${DATA_PATH}" ]; then
        echo "❌ Data directory not found at ${DATA_PATH}"
        exit 1
    fi
    
    # Start training
    exec python scripts/train.py \
        --config "${CONFIG_PATH}" \
        --data_dir "${DATA_PATH}" \
        --log_dir "${LOG_DIR:-/logs}" \
        --checkpoint_dir "${CHECKPOINT_DIR:-/checkpoints}"
}

# Main execution
main() {
    echo "========================================"
    echo "Humanoid Vision System"
    echo "Version: ${VERSION:-0.1.0}"
    echo "Mode: ${MODE:-inference}"
    echo "========================================"
    
    # Check GPU if needed
    if [ "${DISABLE_GPU_CHECK:-0}" != "1" ]; then
        check_gpu || {
            echo "⚠️  GPU check failed but continuing..."
        }
    fi
    
    # Wait for dependencies
    wait_for_deps
    
    # Set Python path
    export PYTHONPATH="${PYTHONPATH}:${APP_HOME}"
    
    # Determine mode and start appropriate service
    case "${MODE:-inference}" in
        "inference" | "api")
            start_api_server
            ;;
        "training" | "train")
            start_training
            ;;
        *)
            echo "❌ Unknown mode: ${MODE}"
            echo "Available modes: inference, training"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"