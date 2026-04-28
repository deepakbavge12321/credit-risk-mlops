# 💳 Credit Risk Prediction Platform - End-to-End MLOps

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22+-FF4B4B.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)
![DVC](https://img.shields.io/badge/DVC-Versioning-945DD6.svg)

**Author:** Bavge Deepak Rajkumar 
**Roll Number:** NA22B031

## 📖 Project Overview

Financial institutions face the critical challenge of deciding whether to approve or reject credit applications while balancing default risk and profitability.

This project addresses this by building an **AI-powered credit risk analysis system** that predicts the probability of loan default using the Home Credit Default Risk dataset. The system provides actionable business decision recommendations (Approve/Reject) along with explainable risk scores.

### ⚙️ MLOps Architecture & Stack

This project strictly adheres to MLOps best practices with a "No Cloud" approach, ensuring full reproducibility, containerization, and local environment parity.

| Component | Technology |
|---|---|
| Data Engineering & Orchestration | Apache Airflow / Apache Spark |
| Data & Model Versioning | DVC, Git, Git LFS |
| Experiment Tracking & Registry | MLflow |
| Model Training | LightGBM (Gradient Boosting Decision Tree) |
| Model Serving | FastAPI |
| User Interface | Streamlit |
| Containerization | Docker & Docker Compose |
| Monitoring | Prometheus & Grafana |

---

## 📂 Project Structure

```text
credit-risk-mlops/
├── artifacts/              # Serialized ML models (e.g., model.pkl)
├── config.yaml             # Centralized config (Model parameters, business logic)
├── dags/                   # Airflow DAGs for workflow automation
│   └── credit_risk_dag.py
├── Data/                   # Raw and processed datasets managed by DVC
├── docker/                 # Dockerfiles for microservices
│   ├── Dockerfile.api
│   └── Dockerfile.ui
├── docker-compose.yml      # Multi-container orchestration
├── docs/                   # Documentation, HLD, LLD, and Architecture diagrams
│   ├── Architecture_diagram.png
│   ├── HLD.pdf
│   └── LLD.pdf
├── dvc.yaml                # DVC pipeline DAG (features -> split -> train -> evaluate)
├── features.json           # Expected schema for FastAPI inference
├── metrics.json            # Model evaluation metrics
├── monitoring/             # Prometheus and alerting configurations
│   ├── alert.rules.yml
│   └── prometheus.yml
├── src/                    # Core ML and API codebase
│   ├── api.py              # FastAPI inference backend
│   ├── data.py             # Data splitting logic
│   ├── evaluate.py         # Model evaluation script
│   ├── features.py         # Data preprocessing and feature engineering
│   ├── model.py            # Model architecture logic
│   └── train.py            # Model training script
├── ui/                     # Frontend application
│   └── app.py              # Streamlit dashboard
└── requirements.txt        # Python dependencies
```

---

## 🚀 Setup & Installation

### Prerequisites

- Docker & Docker Compose
- Python 3.9+
- Git & DVC

### Option 1: Running with Docker Compose (Recommended)

This method ensures environment parity and spins up the API and UI microservices simultaneously.

```bash
# Clone the repository
git clone https://github.com/yourusername/credit-risk-mlops.git
cd credit-risk-mlops

# Build and start the containers in detached mode
docker-compose up --build -d
```

| Service | URL |
|---|---|
| API Swagger Docs | `http://localhost:8000/docs` |
| Streamlit UI Dashboard | `http://localhost:8501` |

### Option 2: Local Development Setup

If you wish to debug locally without Docker:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the backend API
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Run the frontend UI (in a separate terminal)
streamlit run ui/app.py
```

---

## 🧪 Model Pipeline Lifecycle (DVC & MLflow)

The machine learning lifecycle is fully versioned and reproducible using DVC. The pipeline is defined in `dvc.yaml` and consists of 4 stages:

```
features → split → train → evaluate
```

To reproduce the entire pipeline locally:

```bash
dvc repro
```

### Experiment Tracking

Model hyperparameters (learning rate, leaves, max depth) and metrics (AUC) are tracked using MLflow.

```bash
# Launch the MLflow Tracking Server
mlflow ui
```

Access MLflow at `http://localhost:5000` to view training runs and registered models.

---

## 🎯 API Inference Details

The backend is strictly decoupled from the UI and communicates via REST APIs.

**Endpoint:** `POST /predict`

> **Note:** The FastAPI backend dynamically aligns input data with the model's expected features using `features.json`. Any missing features are safely padded with `0` to prevent runtime crashes.

**Example Request:**

```json
{
  "AMT_INCOME_TOTAL": 250000.0,
  "AMT_CREDIT": 500000.0,
  "DAYS_BIRTH": -12000,
  "EXT_SOURCE_1": 0.55,
  "EXT_SOURCE_2": 0.60,
  "EXT_SOURCE_3": 0.70
}
```

**Example Response:**

```json
{
  "default_probability": 0.1245,
  "decision": "APPROVE"
}
```

---

## 📈 Monitoring & Alerts (Prometheus + Grafana)

The application is instrumented to track operational and ML performance metrics in near real-time.

| Component | Detail |
|---|---|
| Metrics Endpoint | `http://localhost:8000/metrics` |
| Prometheus | Scrapes API throughput, latencies, and prediction distributions |
| Alerts | Triggers on high error rates or significant data drift over a rolling window |

---

## 📚 Documentation Reference

For deeper insights into the design decisions and software principles adhered to in this project, refer to the `/docs/` folder:

| Document | Path |
|---|---|
| Architecture Diagram | `docs/Architecture_diagram.png` |
| High-Level Design (HLD) | `docs/HLD.pdf` |
| Low-Level Design (LLD) | `docs/LLD.pdf` |
| Test Plan / MLOps Report | `docs/MLOps_Report.pdf` |

---

*Built adhering to strict ML Product Lifecycle and MLOps guidelines.*
