# HRM-CodeGen Infrastructure Setup Guide

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [CI/CD Pipeline Setup](#cicd-pipeline-setup)
4. [Docker Deployment](#docker-deployment)
5. [Environment Configuration](#environment-configuration)
6. [Performance Benchmarking](#performance-benchmarking)
7. [Monitoring & Observability](#monitoring--observability)
8. [Security Framework](#security-framework)
9. [Troubleshooting](#troubleshooting)
10. [Production Checklist](#production-checklist)

## Overview

This guide provides comprehensive instructions for setting up the HRM-CodeGen validation infrastructure. This infrastructure is specifically designed to prove that the HRM architecture (27M parameters) outperforms standard transformers (GPT-2-117M) on code generation tasks through:

- **Automated Pass@k Evaluation**: Statistical validation of code generation quality
- **Performance Comparison**: Direct benchmarking against transformer baselines
- **Experiment Tracking**: Comprehensive metrics collection and visualization
- **Reproducible Research**: Environment capture for scientific validation

The infrastructure includes:

- **CI/CD Pipelines**: Automated testing, quality gates, and deployment
- **Docker Containers**: Reproducible execution environments
- **Performance Benchmarking**: Comparison framework for model evaluation
- **Monitoring & Observability**: Training metrics, performance tracking
- **Security Framework**: Secure code execution and evaluation

## Prerequisites

### Required Software

- **Docker**: Version 20.10 or later
- **Docker Compose**: Version 2.0 or later
- **Git**: Version 2.30 or later
- **Python**: Version 3.10 or later
- **Node.js**: Version 16 or later (for some tools)

### Required Accounts

- **GitHub**: For code repository and Actions
- **Docker Hub**: For container registry
- **Weights & Biases**: For experiment tracking
- **Slack**: For notifications (optional)

### Required Secrets

Set up the following GitHub secrets:

```bash
# Core secrets
WANDB_API_KEY             # For experiment tracking
DOCKER_USERNAME           # For container registry
DOCKER_PASSWORD           # For container registry
BASELINE_MODEL_PATH       # Path to GPT-2-117M baseline model

# Monitoring
SLACK_WEBHOOK             # For notifications
MONITORING_EMAIL          # For alerts

# Security
SNYK_TOKEN                # For vulnerability scanning
```

## CI/CD Pipeline Setup

### 1. GitHub Actions Configuration

The CI/CD pipeline consists of multiple workflows:

#### Main CI/CD Pipeline (`.github/workflows/ci.yml`)

```bash
# Enable the main CI/CD pipeline
git push origin main  # Triggers on push to main

# Manual trigger with options
gh workflow run ci.yml \
  -f deploy_docs=true \
  -f run_benchmarks=true
```

#### Performance Benchmarking (`.github/workflows/benchmark.yml`)

```bash
# Manual benchmarking run
gh workflow run benchmark.yml \
  -f configurations="base,small,large" \
  -f datasets="mbpp,humaneval" \
  -f compare_baseline=true
```

#### Monitoring Pipeline (`.github/workflows/monitoring.yml`)

```bash
# Manual monitoring run
gh workflow run monitoring.yml \
  -f run_health_check=true \
  -f analyze_training_metrics=true \
  -f generate_reports=true
```

#### Security Pipeline (`.github/workflows/security.yml`)

```bash
# Manual security scan
gh workflow run security.yml \
  -f run_all_scans=true
```

### 2. Pipeline Configuration

#### Setting Up Branch Protection

```bash
# Configure branch protection rules
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["CI/CD Pipeline"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":1}'
```

#### Configuring Environments

1. Go to repository Settings → Environments
2. Create environments: `development`, `staging`, `production`
3. Configure protection rules and secrets for each environment

## Docker Deployment

### 1. Building Images

#### CPU Version

```bash
# Build CPU image
docker build --target cpu-runtime -t hrm-codegen:cpu .

# Test CPU image
docker run --rm -p 8000:8000 \
  -e MODE=serve \
  -e ENVIRONMENT=development \
  hrm-codegen:cpu
```

#### GPU Version

```bash
# Build GPU image
docker build --target gpu-runtime -t hrm-codegen:gpu .

# Test GPU image
docker run --rm --gpus all -p 8000:8000 \
  -e MODE=serve \
  -e ENVIRONMENT=development \
  hrm-codegen:gpu
```

### 2. Multi-Container Setup

#### Docker Compose for Training

```yaml
# docker-compose.train.yml
version: '3.8'
services:
  hrm-trainer:
    build:
      context: .
      target: gpu-runtime
    environment:
      - MODE=train
      - CONFIG_FILE=/app/configs/mbpp_base.yaml
      - ENABLE_WANDB=true
      - WANDB_API_KEY=${WANDB_API_KEY}
      - WANDB_PROJECT=hrm-codegen
    volumes:
      - ./data:/app/data
      - ./configs:/app/configs
      - ./checkpoints:/app/checkpoints
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

#### Docker Compose for Serving

```yaml
# docker-compose.serve.yml
version: '3.8'
services:
  hrm-api:
    build:
      context: .
      target: gpu-runtime
    ports:
      - "8000:8000"
    environment:
      - MODE=serve
      - PORT=8000
      - WORKERS=4
      - CHECKPOINT=/app/checkpoints/best_model.pt
      - ENABLE_MONITORING=true
    volumes:
      - ./checkpoints:/app/checkpoints:ro
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      replicas: 2
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
        limits:
          memory: 8G
          cpus: '4'
```

### 3. Running Containers

#### Training

```bash
# Start training
docker-compose -f docker-compose.train.yml up

# Monitor logs
docker-compose -f docker-compose.train.yml logs -f hrm-trainer
```

#### Serving

```bash
# Start API service
docker-compose -f docker-compose.serve.yml up -d

# Check health
curl http://localhost:8000/health

# Scale service
docker-compose -f docker-compose.serve.yml up --scale hrm-api=4 -d
```

## Environment Configuration

### 1. Development Environment

```bash
# Clone repository
git clone https://github.com/zshinyg/HRMTraining.git
cd HRMTraining

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .  # Install in development mode

# Install pre-commit hooks
pre-commit install
```

### 2. Configuration Files

```bash
# Copy example configuration
cp configs/mbpp_base.yaml.example configs/mbpp_base.yaml

# Edit configuration
vim configs/mbpp_base.yaml

# Set environment variables
cp .env.example .env
vim .env
```

### 3. Environment Variables

Key environment variables for HRM validation:

```
# Core settings
PYTHONPATH=.
HRM_CONFIG_PATH=configs/mbpp_base.yaml
HRM_CHECKPOINT_DIR=checkpoints

# Experiment tracking
WANDB_API_KEY=your_api_key
WANDB_PROJECT=hrm-codegen
WANDB_ENTITY=your_entity

# Evaluation
MBPP_DATASET_PATH=data/mbpp/mbpp.jsonl
HUMANEVAL_DATASET_PATH=data/humaneval/humaneval.jsonl
BASELINE_MODEL_PATH=checkpoints/gpt2-117m

# Performance targets
PASS_AT_1_TARGET=0.30  # 30%
PASS_AT_10_TARGET=0.45  # 45%
```

## Performance Benchmarking

### 1. Benchmark Configuration

#### Setting Up Baseline Models

```bash
# Download GPT-2-117M baseline
python scripts/download_baseline_models.py --model gpt2-117m

# Download CodeT5-small (optional)
python scripts/download_baseline_models.py --model codet5-small
```

#### Configuring Benchmark Datasets

```bash
# Download and prepare MBPP dataset
python scripts/prepare_mbpp_dataset.py

# Download and prepare HumanEval dataset (optional)
python scripts/prepare_humaneval_dataset.py
```

### 2. Running Benchmarks

#### Training Performance

```bash
# Run training benchmark
python scripts/benchmark_training.py \
  --config configs/mbpp_base.yaml \
  --baseline gpt2-117m \
  --metrics memory,speed,convergence \
  --output results/training_benchmark.json
```

#### Inference Performance

```bash
# Run inference benchmark
python scripts/benchmark_inference.py \
  --model checkpoints/hrm_best.pt \
  --baseline checkpoints/gpt2-117m \
  --dataset data/mbpp/mbpp_test.jsonl \
  --metrics latency,throughput,memory \
  --samples 500 \
  --output results/inference_benchmark.json
```

#### Pass@k Evaluation

```bash
# Run Pass@k evaluation
python scripts/evaluate.py \
  --model checkpoints/hrm_best.pt \
  --baseline checkpoints/gpt2-117m \
  --dataset data/mbpp/mbpp_test.jsonl \
  --k 1,5,10,100 \
  --samples 500 \
  --output results/pass_at_k.json
```

### 3. Analyzing Results

```bash
# Generate benchmark report
python scripts/generate_benchmark_report.py \
  --training results/training_benchmark.json \
  --inference results/inference_benchmark.json \
  --pass-at-k results/pass_at_k.json \
  --output results/benchmark_report.html
```

## Monitoring & Observability

### 1. Weights & Biases Integration

```bash
# Install W&B
pip install wandb

# Login to W&B
wandb login

# Set project configuration
export WANDB_PROJECT="hrm-codegen"
export WANDB_ENTITY="your-entity"
```

### 2. Prometheus & Grafana Setup

```bash
# Start monitoring services
docker-compose -f docker-compose.monitoring.yml up -d

# Access Grafana at http://localhost:3000
# Default credentials: admin/admin123
```

## Security Framework

### 1. Code Execution Sandbox

```bash
# Build sandbox Docker image
docker build -t hrm-codegen-sandbox -f Dockerfile.sandbox .

# Test sandbox
python scripts/security/safe_code_executor.py \
  --code "def add(a, b): return a + b" \
  --test-cases '[{"input": "add(1, 2)", "expected_output": "3"}]' \
  --timeout 5
```

### 2. Security Scanning

```bash
# Run security scan
python scripts/security/scan_codebase.py

# Run vulnerability scan
docker run --rm -v $(pwd):/app snyk/snyk:latest test --all-projects
```

## Troubleshooting

### 1. Common Issues

#### Docker Issues

```bash
# Docker build fails
# Clear Docker cache
docker system prune -a

# GPU not available in container
# Install NVIDIA Docker runtime
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

#### Training Issues

```bash
# CUDA out of memory
# Reduce batch size or sequence length
export MAX_BATCH_SIZE=8
export MAX_SEQUENCE_LENGTH=1024

# Training not converging
# Check learning rate and configuration
# Enable debug logging
export LOG_LEVEL=debug
```

## Production Checklist

### Pre-Deployment

- [ ] All tests passing
- [ ] Security scans completed
- [ ] Performance benchmarks meet requirements
- [ ] Configuration reviewed
- [ ] Secrets properly configured
- [ ] Monitoring dashboards set up
- [ ] Documentation updated

### Performance

- [ ] Resource limits configured
- [ ] Auto-scaling rules defined
- [ ] Health checks implemented
- [ ] Graceful shutdown handling
- [ ] Model serving optimized

---

**Last Updated**: 2025-08-05  
**Version**: 1.0.0  
**Maintained By**: HRM-CodeGen Infrastructure Team
