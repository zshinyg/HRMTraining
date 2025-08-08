# HRM-CodeGen Monitoring & Observability Guide

## Table of Contents

1. [Overview](#overview)
2. [Monitoring Architecture](#monitoring-architecture)
3. [Weights & Biases Integration](#weights--biases-integration)
4. [Performance Monitoring](#performance-monitoring)
5. [Error Tracking](#error-tracking)
6. [Resource Monitoring](#resource-monitoring)
7. [Training Metrics](#training-metrics)
8. [Inference Monitoring](#inference-monitoring)
9. [Dashboards](#dashboards)
10. [Alerting](#alerting)
11. [Troubleshooting](#troubleshooting)

## Overview

The HRM-CodeGen monitoring and observability stack provides comprehensive insights into:

- **Training Performance**: Loss curves, convergence patterns, hyperparameter tracking
- **Inference Performance**: Latency, throughput, error rates
- **System Health**: Resource usage, service availability, error tracking
- **Business Metrics**: Pass@k scores, model quality, user satisfaction
- **Security Events**: Anomalies, security incidents, compliance metrics

### Key Components

- **Weights & Biases (W&B)**: Experiment tracking and model management
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboarding
- **ELK Stack**: Log aggregation and analysis
- **Alertmanager**: Alert routing and notification
- **Custom Scripts**: Automated analysis and reporting

## Monitoring Architecture

### Monitoring Stack Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Sources                             │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │   Training  │ │  Inference  │ │   System    │ │  Security   │ │
│ │   Scripts   │ │    API      │ │   Metrics   │ │   Events    │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────┬───────────────┬───────────────┬───────────────┬───────┘
          │               │               │               │
          ▼               ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Collection Layer                             │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │     W&B     │ │ Prometheus  │ │  Fluentd    │ │   Custom    │ │
│ │   Tracking  │ │   Metrics   │ │    Logs     │ │ Collectors  │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────┬───────────────┬───────────────┬───────────────┬───────┘
          │               │               │               │
          ▼               ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Storage Layer                               │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │  W&B Cloud  │ │ Prometheus  │ │Elasticsearch│ │  InfluxDB   │ │
│ │   Storage   │ │  Database   │ │   Cluster   │ │  (Optional) │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────┬───────────────┬───────────────┬───────────────┬───────┘
          │               │               │               │
          ▼               ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Visualization Layer                            │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │ W&B Reports │ │   Grafana   │ │   Kibana    │ │   Custom    │ │
│ │ & Dashboard │ │ Dashboards  │ │ Dashboard   │ │   Reports   │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────┬───────────────┬───────────────┬───────────────┬───────┘
          │               │               │               │
          ▼               ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Alert Layer                                 │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │ Alertmanager│ │    Slack    │ │    Email    │ │  PagerDuty  │ │
│ │   Routing   │ │    Hooks    │ │   Alerts    │ │   (Prod)    │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Collection**: Applications and systems emit metrics, logs, and traces
2. **Aggregation**: Collectors gather and preprocess data
3. **Storage**: Time-series databases and log stores persist data
4. **Visualization**: Dashboards present real-time and historical views
5. **Alerting**: Rules trigger notifications for anomalies

## Weights & Biases Integration

### Setup and Configuration

#### Installation

```bash
# Install W&B
pip install wandb

# Login to W&B
wandb login
# Or set API key
export WANDB_API_KEY=your_api_key_here
```

#### Project Configuration

```python
# wandb_config.py
import wandb
import os
from hrm.config import HRMConfig

class WandBLogger:
    def __init__(self, config: HRMConfig, run_name: str = None):
        self.config = config
        self.run = None
        self.run_name = run_name or f"hrm-{config.name}"
        
    def init_run(self, tags: list = None, notes: str = None):
        """Initialize W&B run."""
        self.run = wandb.init(
            project="hrm-codegen",
            entity=os.getenv("WANDB_ENTITY"),
            name=self.run_name,
            config=self.config.to_dict(),
            tags=tags or [],
            notes=notes,
            job_type="training",
            reinit=True
        )
        
        # Log system information
        self.run.log({
            "system/python_version": os.sys.version,
            "system/cuda_available": torch.cuda.is_available(),
            "system/gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        })
        
        return self.run
    
    def log_metrics(self, metrics: dict, step: int = None):
        """Log training metrics."""
        if self.run:
            self.run.log(metrics, step=step)
    
    def log_model(self, model_path: str, name: str = "model"):
        """Log model artifact."""
        if self.run:
            artifact = wandb.Artifact(name, type="model")
            artifact.add_file(model_path)
            self.run.log_artifact(artifact)
    
    def log_dataset(self, dataset_path: str, name: str = "dataset"):
        """Log dataset artifact."""
        if self.run:
            artifact = wandb.Artifact(name, type="dataset")
            artifact.add_dir(dataset_path)
            self.run.log_artifact(artifact)
    
    def finish(self):
        """Finish W&B run."""
        if self.run:
            self.run.finish()
```

### Training Integration

```python
# In your training script
from wandb_config import WandBLogger
import wandb

# Initialize logger
wandb_logger = WandBLogger(config, run_name=f"training-{timestamp}")
run = wandb_logger.init_run(
    tags=["training", "mbpp", config.model.architecture],
    notes="Training HRM model on MBPP dataset"
)

# Training loop
for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        # Forward pass
        loss = model(batch)
        
        # Log metrics
        if step % log_interval == 0:
            metrics = {
                "train/loss": loss.item(),
                "train/learning_rate": scheduler.get_last_lr()[0],
                "train/epoch": epoch,
                "train/step": global_step,
                "train/gradient_norm": grad_norm,
                "train/memory_usage": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            }
            
            wandb_logger.log_metrics(metrics, step=global_step)
        
        # Validation
        if global_step % validation_interval == 0:
            val_metrics = validate(model, val_dataloader)
            wandb_logger.log_metrics({
                "val/loss": val_metrics["loss"],
                "val/accuracy": val_metrics["accuracy"],
                "val/pass@1": val_metrics["pass@1"],
                "val/pass@10": val_metrics["pass@10"]
            }, step=global_step)
            
            # Log model checkpoint
            if val_metrics["pass@1"] > best_pass_at_1:
                best_pass_at_1 = val_metrics["pass@1"]
                checkpoint_path = f"checkpoints/model_{global_step}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                wandb_logger.log_model(checkpoint_path)

# Finish run
wandb_logger.finish()
```

### Experiment Tracking

#### Hyperparameter Sweeps

```python
# sweep_config.py
sweep_config = {
    "method": "bayes",  # Bayesian optimization
    "metric": {"name": "val/pass@1", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"min": 1e-5, "max": 1e-3, "distribution": "log_uniform"},
        "batch_size": {"values": [16, 32, 64, 128]},
        "num_layers": {"values": [4, 6, 8, 12]},
        "hidden_size": {"values": [256, 512, 768, 1024]},
        "dropout": {"min": 0.0, "max": 0.5},
        "weight_decay": {"min": 0.0, "max": 0.1}
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="hrm-codegen")

# Sweep agent
def sweep_agent():
    wandb.init()
    
    # Get hyperparameters from wandb
    config = wandb.config
    
    # Create model with these hyperparameters
    model = create_model(
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        dropout=config.dropout
    )
    
    # Train and evaluate
    train_and_evaluate(
        model,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        weight_decay=config.weight_decay
    )

# Run sweep
wandb.agent(sweep_id, function=sweep_agent, count=20)
```

#### Experiment Comparison

```python
# Compare experiments
import wandb
api = wandb.Api()

# Get runs from project
runs = api.runs("your-entity/hrm-codegen", 
                filters={"tags": {"$in": ["training", "mbpp"]}})

# Collect metrics
results = []
for run in runs:
    if run.state == "finished":
        results.append({
            "run_id": run.id,
            "name": run.name,
            "pass@1": run.summary.get("val/pass@1", 0),
            "pass@10": run.summary.get("val/pass@10", 0),
            "loss": run.summary.get("val/loss", 0),
            "learning_rate": run.config.get("learning_rate", 0),
            "batch_size": run.config.get("batch_size", 0),
            "model_size": run.config.get("model_size", "unknown"),
            "training_time": run.summary.get("training_time", 0)
        })

# Sort by performance
results.sort(key=lambda x: x["pass@1"], reverse=True)

# Print top 5 experiments
for i, result in enumerate(results[:5]):
    print(f"{i+1}. {result['name']} - Pass@1: {result['pass@1']:.4f}, Pass@10: {result['pass@10']:.4f}")
```

### W&B Reports

```python
# Create a report programmatically
import wandb
api = wandb.Api()

# Create a new report
report = api.create_report(
    project="hrm-codegen",
    title="HRM Model Performance Analysis",
    description="Comparison of HRM models with different configurations",
    blocks=[
        {
            "type": "panel",
            "title": "Training Loss",
            "panels": [{
                "type": "line",
                "fields": ["train/loss"],
                "layout": {"x": 0, "y": 0, "w": 6, "h": 4}
            }]
        },
        {
            "type": "panel",
            "title": "Validation Metrics",
            "panels": [{
                "type": "line",
                "fields": ["val/pass@1", "val/pass@10"],
                "layout": {"x": 6, "y": 0, "w": 6, "h": 4}
            }]
        },
        {
            "type": "panel",
            "title": "Resource Usage",
            "panels": [{
                "type": "line",
                "fields": ["train/memory_usage"],
                "layout": {"x": 0, "y": 4, "w": 12, "h": 4}
            }]
        }
    ]
)

# Share the report
print(f"Report created: {report.url}")
```

## Performance Monitoring

### Prometheus Setup

#### Docker Compose for Prometheus

```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123

  node-exporter:
    image: prom/node-exporter:latest
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.ignored-mount-points=^/(sys|proc|dev|host|etc)($$|/)'

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro

volumes:
  prometheus_data:
  grafana_data:
```

#### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'hrm-api'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['hrm-api:8000']

  - job_name: 'hrm-training'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['hrm-trainer:8001']
```

### FastAPI Metrics Integration

```python
# api/main.py
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
import time

app = FastAPI()

# Initialize Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Custom metrics
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
inference_requests = Counter(
    "hrm_inference_requests_total",
    "Total number of inference requests",
    ["model", "status"]
)

inference_latency = Histogram(
    "hrm_inference_latency_seconds",
    "Inference latency in seconds",
    ["model"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

model_memory_usage = Gauge(
    "hrm_model_memory_usage_bytes",
    "Memory usage of the model in bytes",
    ["model"]
)

# Middleware for tracking metrics
@app.middleware("http")
async def track_metrics(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    # Only track metrics for inference endpoints
    if request.url.path.startswith("/api/generate"):
        model = request.query_params.get("model", "default")
        status = "success" if response.status_code < 400 else "error"
        
        # Update metrics
        inference_requests.labels(model=model, status=status).inc()
        inference_latency.labels(model=model).observe(duration)
    
    return response

# Endpoint for generating code
@app.post("/api/generate")
async def generate_code(request: CodeGenerationRequest):
    model_name = request.model or "hrm-default"
    
    # Track memory before inference
    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    # Generate code
    result = model.generate(request.prompt, **request.params)
    
    # Track memory after inference
    end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    memory_used = end_memory - start_memory
    model_memory_usage.labels(model=model_name).set(memory_used)
    
    return {"generated_code": result}
```

### Training Metrics Collection

```python
# scripts/train.py
from prometheus_client import start_http_server, Gauge, Counter, Summary
import time
import threading

# Start Prometheus metrics server on a separate port
start_http_server(8001)

# Define metrics
training_loss = Gauge("hrm_training_loss", "Training loss", ["dataset"])
validation_loss = Gauge("hrm_validation_loss", "Validation loss", ["dataset"])
pass_at_k = Gauge("hrm_pass_at_k", "Pass@k score", ["dataset", "k"])
training_step = Counter("hrm_training_steps_total", "Total training steps")
epoch_duration = Summary("hrm_epoch_duration_seconds", "Duration of training epoch")
memory_usage = Gauge("hrm_memory_usage_bytes", "Memory usage during training")
learning_rate = Gauge("hrm_learning_rate", "Current learning rate")

# Function to periodically collect system metrics
def collect_system_metrics():
    while True:
        if torch.cuda.is_available():
            memory_usage.set(torch.cuda.memory_allocated())
        time.sleep(5)

# Start system metrics collection in a background thread
threading.Thread(target=collect_system_metrics, daemon=True).start()

# Training loop with metrics
for epoch in range(num_epochs):
    epoch_start = time.time()
    
    for step, batch in enumerate(dataloader):
        # Forward pass
        loss = model(batch)
        
        # Update metrics
        training_loss.labels(dataset="mbpp").set(loss.item())
        training_step.inc()
        learning_rate.set(scheduler.get_last_lr()[0])
        
        # Validation
        if global_step % validation_interval == 0:
            val_metrics = validate(model, val_dataloader)
            validation_loss.labels(dataset="mbpp").set(val_metrics["loss"])
            pass_at_k.labels(dataset="mbpp", k="1").set(val_metrics["pass@1"])
            pass_at_k.labels(dataset="mbpp", k="10").set(val_metrics["pass@10"])
    
    # Record epoch duration
    epoch_duration.observe(time.time() - epoch_start)
```

## Error Tracking

### ELK Stack Setup

```yaml
# docker-compose.elk.yml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.15.0
    ports:
      - "9200:9200"
      - "9300:9300"
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  kibana:
    image: docker.elastic.co/kibana/kibana:7.15.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

  logstash:
    image: docker.elastic.co/logstash/logstash:7.15.0
    ports:
      - "5044:5044"
      - "9600:9600"
    volumes:
      - ./monitoring/logstash/pipeline:/usr/share/logstash/pipeline
    depends_on:
      - elasticsearch

  filebeat:
    image: docker.elastic.co/beats/filebeat:7.15.0
    volumes:
      - ./monitoring/filebeat/filebeat.yml:/usr/share/filebeat/filebeat.yml:ro
      - ./logs:/logs:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
    depends_on:
      - elasticsearch
      - logstash

volumes:
  elasticsearch_data:
```

### Structured Logging

```python
# logger.py
import json
import logging
import sys
import traceback
from datetime import datetime

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "path": record.pathname,
            "line": record.lineno,
            "function": record.funcName
        }
        
        # Add exception info if available
        if record.exc_info:
            log_record["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, "extra"):
            log_record.update(record.extra)
        
        return json.dumps(log_record)

def setup_logger(name, level=logging.INFO):
    """Set up a logger with JSON formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JsonFormatter())
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(f"logs/{name}.log")
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)
    
    return logger

# Example usage
logger = setup_logger("hrm")

# Log with extra fields
logger.info(
    "Model training started",
    extra={
        "model_name": "hrm-base",
        "dataset": "mbpp",
        "batch_size": 32,
        "learning_rate": 1e-4,
        "run_id": "run-123456"
    }
)

# Log errors
try:
    # Some operation that might fail
    result = 1 / 0
except Exception as e:
    logger.error(
        "Error during training",
        exc_info=True,
        extra={
            "step": 1000,
            "epoch": 2,
            "batch_id": 45
        }
    )
```

### Kibana Dashboard Setup

1. **Log Index Pattern**:
   - Go to Kibana → Stack Management → Index Patterns
   - Create index pattern `logstash-*`
   - Set time field to `@timestamp`

2. **Create Visualizations**:
   - Error Rate: Count of logs with level=ERROR
   - Error Distribution: Pie chart of error types
   - Log Volume: Line chart of log count over time
   - Training Progress: Line chart of training metrics

3. **Create Dashboard**:
   - Go to Kibana → Dashboard → Create Dashboard
   - Add visualizations
   - Save as "HRM Training Monitoring"

## Resource Monitoring

### GPU Monitoring

```python
# scripts/gpu_monitor.py
import time
import subprocess
import json
from prometheus_client import start_http_server, Gauge

# Start Prometheus metrics server
start_http_server(8002)

# Define GPU metrics
gpu_utilization = Gauge("gpu_utilization_percent", "GPU utilization percentage", ["gpu"])
gpu_memory_used = Gauge("gpu_memory_used_bytes", "GPU memory used in bytes", ["gpu"])
gpu_memory_total = Gauge("gpu_memory_total_bytes", "GPU total memory in bytes", ["gpu"])
gpu_temperature = Gauge("gpu_temperature_celsius", "GPU temperature in Celsius", ["gpu"])
gpu_power_usage = Gauge("gpu_power_usage_watts", "GPU power usage in watts", ["gpu"])

def get_gpu_stats():
    """Get GPU statistics using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        
        for line in result.stdout.strip().split("\n"):
            values = [v.strip() for v in line.split(",")]
            if len(values) >= 6:
                gpu_id, util, mem_used, mem_total, temp, power = values
                
                # Update metrics
                gpu_utilization.labels(gpu=gpu_id).set(float(util))
                gpu_memory_used.labels(gpu=gpu_id).set(float(mem_used) * 1024 * 1024)  # MiB to bytes
                gpu_memory_total.labels(gpu=gpu_id).set(float(mem_total) * 1024 * 1024)  # MiB to bytes
                gpu_temperature.labels(gpu=gpu_id).set(float(temp))
                gpu_power_usage.labels(gpu=gpu_id).set(float(power))
                
    except Exception as e:
        print(f"Error collecting GPU stats: {e}")

# Main monitoring loop
def main():
    while True:
        get_gpu_stats()
        time.sleep(5)

if __name__ == "__main__":
    main()
```

### System Resource Monitoring

```python
# scripts/system_monitor.py
import psutil
import time
from prometheus_client import start_http_server, Gauge

# Start Prometheus metrics server
start_http_server(8003)

# Define system metrics
cpu_usage = Gauge("system_cpu_usage_percent", "CPU usage percentage")
memory_usage = Gauge("system_memory_usage_bytes", "Memory usage in bytes")
memory_total = Gauge("system_memory_total_bytes", "Total memory in bytes")
disk_usage = Gauge("system_disk_usage_bytes", "Disk usage in bytes", ["mount"])
disk_total = Gauge("system_disk_total_bytes", "Total disk space in bytes", ["mount"])
network_sent = Gauge("system_network_sent_bytes", "Network bytes sent", ["interface"])
network_recv = Gauge("system_network_recv_bytes", "Network bytes received", ["interface"])

def collect_metrics():
    """Collect system metrics."""
    # CPU metrics
    cpu_usage.set(psutil.cpu_percent())
    
    # Memory metrics
    memory = psutil.virtual_memory()
    memory_usage.set(memory.used)
    memory_total.set(memory.total)
    
    # Disk metrics
    for partition in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            disk_usage.labels(mount=partition.mountpoint).set(usage.used)
            disk_total.labels(mount=partition.mountpoint).set(usage.total)
        except:
            pass
    
    # Network metrics
    net_io = psutil.net_io_counters(pernic=True)
    for interface, counters in net_io.items():
        network_sent.labels(interface=interface).set(counters.bytes_sent)
        network_recv.labels(interface=interface).set(counters.bytes_recv)

# Main monitoring loop
def main():
    while True:
        collect_metrics()
        time.sleep(5)

if __name__ == "__main__":
    main()
```

## Training Metrics

### Automated Metrics Analysis

```python
# scripts/analyze_training_metrics.py
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_training_run(run_id):
    """Analyze a training run and generate report."""
    api = wandb.Api()
    run = api.run(f"your-entity/hrm-codegen/{run_id}")
    
    # Get run history as DataFrame
    history = run.scan_history()
    df = pd.DataFrame(history)
    
    # Calculate key statistics
    stats = {
        "run_name": run.name,
        "start_time": run.created_at,
        "duration": (run.finished_at - run.created_at).total_seconds() / 3600,  # hours
        "final_loss": df["train/loss"].iloc[-1] if "train/loss" in df else None,
        "best_pass@1": df["val/pass@1"].max() if "val/pass@1" in df else None,
        "best_pass@10": df["val/pass@10"].max() if "val/pass@10" in df else None,
        "convergence_epoch": df[df["val/pass@1"] == df["val/pass@1"].max()]["train/epoch"].iloc[0] if "val/pass@1" in df else None,
        "peak_memory_gb": df["train/memory_usage"].max() if "train/memory_usage" in df else None,
        "avg_step_time": df["train/step_time"].mean() if "train/step_time" in df else None
    }
    
    # Generate plots
    plt.figure(figsize=(12, 8))
    
    # Loss curve
    plt.subplot(2, 2, 1)
    if "train/loss" in df:
        plt.plot(df["train/step"], df["train/loss"], label="Training Loss")
    if "val/loss" in df:
        plt.plot(df["train/step"][df["val/loss"].notna()], df["val/loss"].dropna(), label="Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    
    # Pass@k metrics
    plt.subplot(2, 2, 2)
    if "val/pass@1" in df:
        plt.plot(df["train/step"][df["val/pass@1"].notna()], df["val/pass@1"].dropna(), label="Pass@1")
    if "val/pass@10" in df:
        plt.plot(df["train/step"][df["val/pass@10"].notna()], df["val/pass@10"].dropna(), label="Pass@10")
    plt.xlabel("Steps")
    plt.ylabel("Pass@k Score")
    plt.title("Code Generation Performance")
    plt.legend()
    
    # Learning rate
    plt.subplot(2, 2, 3)
    if "train/learning_rate" in df:
        plt.plot(df["train/step"], df["train/learning_rate"])
    plt.xlabel("Steps")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    
    # Memory usage
    plt.subplot(2, 2, 4)
    if "train/memory_usage" in df:
        plt.plot(df["train/step"], df["train/memory_usage"])
    plt.xlabel("Steps")
    plt.ylabel("Memory Usage (GB)")
    plt.title("GPU Memory Usage")
    
    # Save plot
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"reports/{run_id}_{timestamp}_analysis.png"
    plt.savefig(plot_path)
    
    # Generate report
    report = f"""
    # Training Run Analysis: {run.name}
    
    ## Overview
    - **Run ID**: {run_id}
    - **Start Time**: {stats['start_time']}
    - **Duration**: {stats['duration']:.2f} hours
    
    ## Performance Metrics
    - **Final Training Loss**: {stats['final_loss']:.4f}
    - **Best Pass@1**: {stats['best_pass@1']:.4f}
    - **Best Pass@10**: {stats['best_pass@10']:.4f}
    - **Convergence Epoch**: {stats['convergence_epoch']}
    
    ## Resource Usage
    - **Peak Memory Usage**: {stats['peak_memory_gb']:.2f} GB
    - **Average Step Time**: {stats['avg_step_time']:.4f} seconds
    
    ## Analysis
    ![Training Metrics]({plot_path})
    
    ## Recommendations
    - {'Consider early stopping as convergence was reached at epoch ' + str(stats['convergence_epoch']) if stats['convergence_epoch'] else 'No convergence data available'}
    - {'Memory usage is high, consider reducing batch size' if stats['peak_memory_gb'] and stats['peak_memory_gb'] > 10 else 'Memory usage is acceptable'}
    - {'Step time is slow, consider optimizing data loading' if stats['avg_step_time'] and stats['avg_step_time'] > 0.5 else 'Step time is good'}
    """
    
    # Save report
    report_path = f"reports/{run_id}_{timestamp}_analysis.md"
    with open(report_path, "w") as f:
        f.write(report)
    
    return {
        "stats": stats,
        "report_path": report_path,
        "plot_path": plot_path
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze training metrics")
    parser.add_argument("--run_id", required=True, help="W&B run ID to analyze")
    args = parser.parse_args()
    
    result = analyze_training_run(args.run_id)
    print(f"Analysis complete. Report saved to {result['report_path']}")
```

### Training Comparison Dashboard

```python
# scripts/generate_comparison_dashboard.py
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def generate_comparison_dashboard(run_ids, output_dir="reports"):
    """Generate a comparison dashboard for multiple training runs."""
    api = wandb.Api()
    runs = [api.run(f"your-entity/hrm-codegen/{run_id}") for run_id in run_ids]
    
    # Collect summary metrics
    summary_data = []
    for run in runs:
        summary_data.append({
            "run_id": run.id,
            "run_name": run.name,
            "model_size": run.config.get("model_size", "unknown"),
            "batch_size": run.config.get("batch_size", 0),
            "learning_rate": run.config.get("learning_rate", 0),
            "pass@1": run.summary.get("val/pass@1", 0),
            "pass@10": run.summary.get("val/pass@10", 0),
            "training_time": run.summary.get("training_time", 0),
            "peak_memory": run.summary.get("peak_memory_usage", 0)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Get detailed metrics for each run
    detailed_data = {}
    for run in runs:
        history = run.scan_history()
        detailed_data[run.name] = pd.DataFrame(history)
    
    # Generate plots
    plt.figure(figsize=(15, 12))
    
    # Pass@1 comparison
    plt.subplot(2, 2, 1)
    sns.barplot(x="run_name", y="pass@1", data=summary_df)
    plt.title("Pass@1 Comparison")
    plt.xticks(rotation=45)
    plt.ylim(0, max(summary_df["pass@1"]) * 1.2)
    
    # Pass@10 comparison
    plt.subplot(2, 2, 2)
    sns.barplot(x="run_name", y="pass@10", data=summary_df)
    plt.title("Pass@10 Comparison")
    plt.xticks(rotation=45)
    plt.ylim(0, max(summary_df["pass@10"]) * 1.2)
    
    # Training time comparison
    plt.subplot(2, 2, 3)
    sns.barplot(x="run_name", y="training_time", data=summary_df)
    plt.title("Training Time (hours)")
    plt.xticks(rotation=45)
    
    # Memory usage comparison
    plt.subplot(2, 2, 4)
    sns.barplot(x="run_name", y="peak_memory", data=summary_df)
    plt.title("Peak Memory Usage (GB)")
    plt.xticks(rotation=45)
    
    # Save comparison plot
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_plot_path = f"{output_dir}/comparison_{timestamp}.png"
    plt.savefig(comparison_plot_path)
    
    # Loss curves comparison
    plt.figure(figsize=(12, 6))
    for name, df in detailed_data.items():
        if "train/loss" in df:
            plt.plot(df["train/step"], df["train/loss"], label=f"{name} (Train)")
        if "val/loss" in df:
            plt.plot(df["train/step"][df["val/loss"].notna()], 
                    df["val/loss"].dropna(), 
                    label=f"{name} (Val)",
                    linestyle="--")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss Curves Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save loss comparison plot
    loss_plot_path = f"{output_dir}/loss_comparison_{timestamp}.png"
    plt.savefig(loss_plot_path)
    
    # Pass@k curves comparison
    plt.figure(figsize=(12, 6))
    for name, df in detailed_data.items():
        if "val/pass@1" in df:
            plt.plot(df["train/step"][df["val/pass@1"].notna()], 
                    df["val/pass@1"].dropna(), 
                    label=f"{name} (Pass@1)")
    plt.xlabel("Steps")
    plt.ylabel("Pass@1 Score")
    plt.title("Pass@1 Curves Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save Pass@k comparison plot
    pass_at_k_plot_path = f"{output_dir}/pass_at_k_comparison_{timestamp}.png"
    plt.savefig(pass_at_k_plot_path)
    
    # Generate comparison report
    report = f"""
    # Training Runs Comparison
    
    ## Overview
    Comparison of {len(runs)} training runs:
    {', '.join([run.name for run in runs])}
    
    ## Performance Comparison
    ![Performance Metrics]({comparison_plot_path})
    
    ## Loss Curves
    ![Loss Comparison]({loss_plot_path})
    
    ## Pass@k Curves
    ![Pass@k Comparison]({pass_at_k_plot_path})
    
    ## Detailed Metrics
    
    | Run | Model Size | Batch Size | Learning Rate | Pass@1 | Pass@10 | Training Time (h) | Peak Memory (GB) |
    |-----|------------|------------|---------------|--------|---------|-------------------|------------------|
    {summary_df.apply(lambda row: f"| {row['run_name']} | {row['model_size']} | {row['batch_size']} | {row['learning_rate']:.1e} | {row['pass@1']:.4f} | {row['pass@10']:.4f} | {row['training_time']:.2f} | {row['peak_memory']:.2f} |", axis=1).str.cat(sep='\\n')}
    
    ## Analysis
    
    - **Best Pass@1**: {summary_df.loc[summary_df['pass@1'].idxmax()]['run_name']} ({summary_df['pass@1'].max():.4f})
    - **Best Pass@10**: {summary_df.loc[summary_df['pass@10'].idxmax()]['run_name']} ({summary_df['pass@10'].max():.4f})
    - **Fastest Training**: {summary_df.loc[summary_df['training_time'].idxmin()]['run_name']} ({summary_df['training_time'].min():.2f} hours)
    - **Most Memory Efficient**: {summary_df.loc[summary_df['peak_memory'].idxmin()]['run_name']} ({summary_df['peak_memory'].min():.2f} GB)
    
    ## Recommendations
    
    - {'Consider using ' + summary_df.loc[summary_df['pass@1'].idxmax()]['run_name'] + ' configuration for best performance'}
    - {'For resource-constrained environments, ' + summary_df.loc[summary_df['peak_memory'].idxmin()]['run_name'] + ' offers the best efficiency'}
    """
    
    # Save report
    report_path = f"{output_dir}/comparison_report_{timestamp}.md"
    with open(report_path, "w") as f:
        f.write(report)
    
    return {
        "summary_df": summary_df,
        "report_path": report_path,
        "comparison_plot_path": comparison_plot_path,
        "loss_plot_path": loss_plot_path,
        "pass_at_k_plot_path": pass_at_k_plot_path
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comparison dashboard")
    parser.add_argument("--run_ids", required=True, nargs="+", help="W&B run IDs to compare")
    parser.add_argument("--output_dir", default="reports", help="Output directory for reports")
    args = parser.parse_args()
    
    result = generate_comparison_dashboard(args.run_ids, args.output_dir)
    print(f"Comparison complete. Report saved to {result['report_path']}")
```

## Inference Monitoring

### Inference Performance Tracking

```python
# api/monitoring.py
import time
import threading
import queue
from prometheus_client import Histogram, Counter, Gauge
import torch

# Define metrics
inference_latency = Histogram(
    "hrm_inference_latency_seconds",
    "Inference latency in seconds",
    ["model", "batch_size"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

tokens_per_second = Gauge(
    "hrm_tokens_per_second",
    "Tokens generated per second",
    ["model"]
)

inference_requests = Counter(
    "hrm_inference_requests_total",
    "Total number of inference requests",
    ["model", "status"]
)

inference_tokens = Counter(
    "hrm_inference_tokens_total",
    "Total number of tokens generated",
    ["model"]
)

gpu_memory_allocated = Gauge(
    "hrm_gpu_memory_allocated_bytes",
    "GPU memory allocated during inference",
    ["model"]
)

# Inference performance tracker
class InferenceTracker:
    def __init__(self, model_name):
        self.model_name = model_name
        self.metrics_queue = queue.Queue()
        self.running = True
        
        # Start background thread for aggregating metrics
        self.thread = threading.Thread(target=self._process_metrics)
        self.thread.daemon = True
        self.thread.start()
    
    def track_inference(self, batch_size=1):
        """Context manager for tracking inference performance."""
        class InferenceContext:
            def __init__(self, tracker, batch_size):
                self.tracker = tracker
                self.batch_size = batch_size
                self.start_time = None
                self.tokens_generated = 0
                self.status = "success"
            
            def __enter__(self):
                self.start_time = time.time()
                # Record initial GPU memory
                if torch.cuda.is_available():
                    self.start_memory = torch.cuda.memory_allocated()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                # Calculate latency
                latency = time.time() - self.start_time
                
                # Record metrics
                if exc_type is not None:
                    self.status = "error"
                
                # Calculate memory used
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() - self.start_memory
                    gpu_memory_allocated.labels(model=self.tracker.model_name).set(memory_used)
                
                # Add to metrics queue
                self.tracker.metrics_queue.put({
                    "latency": latency,
                    "tokens": self.tokens_generated,
                    "batch_size": self.batch_size,
                    "status": self.status
                })
                
                return False  # Don't suppress exceptions
            
            def record_tokens(self, num_tokens):
                """Record number of tokens generated."""
                self.tokens_generated += num_tokens
        
        return InferenceContext(self, batch_size)
    
    def _process_metrics(self):
        """Background thread for processing metrics."""
        while self.running:
            try:
                # Get metrics from queue with timeout
                metrics = self.metrics_queue.get(timeout=1)
                
                # Update Prometheus metrics
                inference_latency.labels(
                    model=self.model_name,
                    batch_size=str(metrics["batch_size"])
                ).observe(metrics["latency"])
                
                inference_requests.labels(
                    model=self.model_name,
                    status=metrics["status"]
                ).inc()
                
                if metrics["tokens"] > 0:
                    inference_tokens.labels(model=self.model_name).inc(metrics["tokens"])
                    tokens_per_second.labels(model=self.model_name).set(
                        metrics["tokens"] / metrics["latency"] if metrics["latency"] > 0 else 0
                    )
                
                self.metrics_queue.task_done()
            
            except queue.Empty:
                # No metrics to process
                pass
            except Exception as e:
                print(f"Error processing metrics: {e}")
    
    def shutdown(self):
        """Shutdown the tracker."""
        self.running = False
        self.thread.join(timeout=5)

# Example usage
tracker = InferenceTracker("hrm-base")

def generate_code(prompt, max_tokens=100):
    """Generate code with performance tracking."""
    with tracker.track_inference() as context:
        # Actual code generation
        start_time = time.time()
        result = model.generate(prompt, max_tokens=max_tokens)
        
        # Record tokens generated
        num_tokens = len(result.split())
        context.record_tokens(num_tokens)
        
        return result
```

### Batch Inference Monitoring

```python
# api/batch_inference.py
from prometheus_client import Histogram, Summary, Counter, Gauge
import time
import threading
import torch

# Define batch metrics
batch_size = Gauge(
    "hrm_batch_size",
    "Current batch size for inference",
    ["model"]
)

batch_queue_size = Gauge(
    "hrm_batch_queue_size",
    "Number of requests in the batch queue",
    ["model"]
)

batch_latency = Histogram(
    "hrm_batch_latency_seconds",
    "Batch processing latency in seconds",
    ["model"],
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 30.0)
)

request_queue_time = Histogram(
    "hrm_request_queue_time_seconds",
    "Time requests spend in queue before processing",
    ["model"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

batch_throughput = Gauge(
    "hrm_batch_throughput_requests_per_second",
    "Batch throughput in requests per second",
    ["model"]
)

# Batch inference processor with monitoring
class BatchInferenceProcessor:
    def __init__(self, model_name, model, max_batch_size=8, max_wait_time=0.1):
        self.model_name = model_name
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        
        self.queue = []
        self.queue_lock = threading.Lock()
        self.running = True
        
        # Start batch processing thread
        self.thread = threading.Thread(target=self._process_batches)
        self.thread.daemon = True
        self.thread.start()
        
        # Update queue size metric
        batch_queue_size.labels(model=model_name).set(0)
        batch_size.labels(model=model_name).set(0)
    
    def add_request(self, prompt, max_tokens=100):
        """Add a request to the batch queue."""
        result_future = torch.Future()
        
        with self.queue_lock:
            enqueue_time = time.time()
            self.queue.append({
                "prompt": prompt,
                "max_tokens": max_tokens,
                "future": result_future,
                "enqueue_time": enqueue_time
            })
            batch_queue_size.labels(model=self.model_name).set(len(self.queue))
        
        return result_future
    
    def _process_batches(self):
        """Process batches in background thread."""
        while self.running:
            batch = []
            
            # Get batch from queue
            with self.queue_lock:
                if len(self.queue) > 0:
                    # Take up to max_batch_size requests
                    batch = self.queue[:self.max_batch_size]
                    self.queue = self.queue[self.max_batch_size:]
                    batch_queue_size.labels(model=self.model_name).set(len(self.queue))
            
            if batch:
                # Process batch
                batch_size.labels(model=self.model_name).set(len(batch))
                
                # Record queue times
                current_time = time.time()
                for request in batch:
                    queue_time = current_time - request["enqueue_time"]
                    request_queue_time.labels(model=self.model_name).observe(queue_time)
                
                # Batch inference
                try:
                    batch_start = time.time()
                    
                    # Prepare inputs
                    prompts = [req["prompt"] for req in batch]
                    max_tokens = max([req["max_tokens"] for req in batch])
                    
                    # Generate outputs
                    outputs = self.model.generate_batch(prompts, max_tokens=max_tokens)
                    
                    # Record batch latency
                    batch_duration = time.time() - batch_start
                    batch_latency.labels(model=self.model_name).observe(batch_duration)
                    
                    # Calculate throughput
                    throughput = len(batch) / batch_duration if batch_duration > 0 else 0
                    batch_throughput.labels(model=self.model_name).set(throughput)
                    
                    # Set results
                    for i, request in enumerate(batch):
                        request["future"].set_result(outputs[i])
                
                except Exception as e:
                    # Set error for all requests in batch
                    for request in batch:
                        request["future"].set_exception(e)
            
            else:
                # No requests, wait a bit
                time.sleep(0.01)
    
    def shutdown(self):
        """Shutdown the batch processor."""
        self.running = False
        self.thread.join(timeout=5)
        
        # Set remaining futures to cancelled
        with self.queue_lock:
            for request in self.queue:
                if not request["future"].done():
                    request["future"].set_exception(Exception("Batch processor shutdown"))

# Example usage
processor = BatchInferenceProcessor("hrm-base", model, max_batch_size=8)

async def generate_code_batched(prompt, max_tokens=100):
    """Generate code using batch processor."""
    future = processor.add_request(prompt, max_tokens)
    
    # Wait for result
    result = await future
    return result
```

## Dashboards

### Grafana Dashboard Setup

#### Training Dashboard

```json
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": "-- Grafana --",
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "gnetId": null,
  "graphTooltip": 0,
  "id": 1,
  "links": [],
  "panels": [
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "custom": {}
        },
        "overrides": []
      },
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 0
      },
      "hiddenSeries": false,
      "id": 2,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "alertThreshold": true
      },
      "percentage": false,
      "pluginVersion": "7.3.7",
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "expr": "hrm_training_loss{dataset=\"mbpp\"}",
          "interval": "",
          "legendFormat": "Training Loss",
          "refId": "A"
        },
        {
          "expr": "hrm_validation_loss{dataset=\"mbpp\"}",
          "interval": "",
          "legendFormat": "Validation Loss",
          "refId": "B"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "Loss Curves",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    },
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "custom": {}
        },
        "overrides": []
      },
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 0
      },
      "hiddenSeries": false,
      "id": 4,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "alertThreshold": true
      },
      "percentage": false,
      "pluginVersion": "7.3.7",
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "expr": "hrm_pass_at_k{dataset=\"mbpp\", k=\"1\"}",
          "interval": "",
          "legendFormat": "Pass@1",
          "refId": "A"
        },
        {
          "expr": "hrm_pass_at_k{dataset=\"mbpp\", k=\"10\"}",
          "interval": "",
          "legendFormat": "Pass@10",
          "refId": "B"
        }
      ],
      "thresholds": [
        {
          "colorMode": "critical",
          "fill": true,
          "line": true,
          "op": "lt",
          "value": 0.3,
          "yaxis": "left"
        }
      ],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "Pass@k Metrics",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "percentunit",
          "label": null,
          "logBase": 1,
          "max": "1",
          "min": "0",
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    },
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "custom": {}
        },
        "overrides": []
      },
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 8
      },
      "hiddenSeries": false,
      "id": 6,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "alertThreshold": true
      },
      "percentage": false,
      "pluginVersion": "7.3.7",
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "expr": "hrm_learning_rate",
          "interval": "",
          "legendFormat": "Learning Rate",
          "refId": "A"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "Learning Rate",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    },
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "custom": {}
        },
        "overrides": []
      },
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 8
      },
      "hiddenSeries": false,
      "id": 8,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "alertThreshold": true
      },
      "percentage": false,
      "pluginVersion": "7.3.7",
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "expr": "hrm_memory_usage_bytes / 1024 / 1024 / 1024",
          "interval": "",
          "legendFormat": "GPU Memory (GB)",
          "refId": "A"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "GPU Memory Usage",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "gbytes",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    }
  ],
  "refresh": "5s",
  "schemaVersion": 26,
  "style": "dark",
  "tags": ["training", "hrm"],
  "templating": {
    "list": []
  },
  "time": {
    "from": "now-1h",
    "to": "now"
  },
  "timepicker": {
    "refresh_intervals": [
      "5s",
      "10s",
      "30s",
      "1m",
      "5m",
      "15m",
      "30m",
      "1h",
      "2h",
      "1d"
    ]
  },
  "timezone": "",
  "title": "HRM Training Monitoring",
  "uid": "hrm-training",
  "version": 1
}
```

#### Inference Dashboard

```json
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": "-- Grafana --",
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "gnetId": null,
  "graphTooltip": 0,
  "id": 2,
  "links": [],
  "panels": [
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "custom": {},
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              },
              {
                "color": "red",
                "value": 80
              }
            ]
          }
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 6,
        "x": 0,
        "y": 0
      },
      "id": 2,
      "options": {
        "colorMode": "value",
        "graphMode": "area",
        "justifyMode": "auto",
        "orientation": "auto",
        "reduceOptions": {
          "calcs": [
            "mean"
          ],
          "fields": "",
          "values": false
        },
        "textMode": "auto"
      },
      "pluginVersion": "7.3.7",
      "targets": [
        {
          "expr": "sum(rate(hrm_inference_requests_total[5m]))",
          "interval": "",
          "legendFormat": "",
          "refId": "A"
        }
      ],
      "timeFrom": null,
      "timeShift": null,
      "title": "Requests Per Second",
      "type": "stat"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "custom": {},
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              },
              {
                "color": "red",
                "value": 80
              }
            ]
          },
          "unit": "s"
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 6,
        "x": 6,
        "y": 0
      },
      "id": 4,
      "options": {
        "colorMode": "value",
        "graphMode": "area",
        "justifyMode": "auto",
        "orientation": "auto",
        "reduceOptions": {
          "calcs": [
            "mean"
          ],
          "fields": "",
          "values": false
        },
        "textMode": "auto"
      },
      "pluginVersion": "7.3.7",
      "targets": [
        {
          "expr": "histogram_quantile(0.95, sum(rate(hrm_inference_latency_seconds_bucket[5m])) by (le))",
          "interval": "",
          "legendFormat": "",
          "refId": "A"
        }
      ],
      "timeFrom": null,
      "timeShift": null,
      "title": "P95 Latency",
      "type": "stat"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "custom": {},
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              },
              {
                "color": "red",
                "value": 80
              }
            ]
          }
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 6,
        "x": 12,
        "y": 0
      },
      "id": 6,
      "options": {
        "colorMode": "value",
        "graphMode": "area",
        "justifyMode": "auto",
        "orientation": "auto",
        "reduceOptions": {
          "calcs": [
            "mean"
          ],
          "fields": "",
          "values": false
        },
        "textMode": "auto"
      },
      "pluginVersion": "7.3.7",
      "targets": [
        {
          "expr": "avg(hrm_tokens_per_second)",
          "interval": "",
          "legendFormat": "",
          "refId": "A"
        }
      ],
      "timeFrom": null,
      "timeShift": null,
      "title": "Tokens Per Second",
      "type": "stat"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "custom": {},
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              },
              {
                "color": "yellow",
                "value": 0.01
              },
              {
                "color": "red",
                "value": 0.05
              }
            ]
          },
          "unit": "percentunit"
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 6,
        "x": 18,
        "y": 0
      },
      "id": 8,
      "options": {
        "colorMode": "value",
        "graphMode": "area",
        "justifyMode": "auto",
        "orientation": "auto",
        "reduceOptions": {
          "calcs": [
            "mean"
          ],
          "fields": "",
          "values": false
        },
        "textMode": "auto"
      },
      "pluginVersion": "7.3.7",
      "targets": [
        {
          "expr": "sum(rate(hrm_inference_requests_total{status=\"error\"}[5m])) / sum(rate(hrm_inference_requests_total[5m]))",
          "interval": "",
          "legendFormat": "",
          "refId": "A"
        }
      ],
      "timeFrom": null,
      "timeShift": null,
      "title": "Error Rate",
      "type": "stat"
    },
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,