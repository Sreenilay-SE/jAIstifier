# jAIstifier
### Lightweight AI Decision Auditor

jAIstifier is a lightweight, easy-to-integrate audit system for AI models. It acts as a transparent layer over your scikit-learn compatible models, dynamically computing feature importance via perturbation-based SHAP approximation at inference time, and transmitting comprehensive audit telemetry to a central monitoring dashboard.

## 🚀 Key Features

*   **SDK Interceptor:** An easy-to-use Python SDK `jaistifier_sdk.py` that wraps around any model equipped with `.predict()` or `.predict_proba()`.
*   **Real-time Feature Importance:** Uses perturbation-based SHAP approximations to calculate dynamic feature weights on each individual prediction.
*   **Centralized Audit Server:** A lightweight FastAPI server (`server.py`) acting as a hub to collect and serve the audit data.
*   **Performance Benchmarking:** Built-in benchmarking script (`benchmark_test.py`) to test and monitor local compute overhead.
*   **Comprehensive Demos:** Includes a simulated real-world use case (`friend_laptop_demo.py`) based on an insurance claim approval model.

## 📂 Project Structure

*   `server.py`: The centralized FastAPI server that acts as the audit hub.
*   `jaistifier_sdk.py`: The python SDK for clients to intercept and audit model requests. 
*   `friend_laptop_demo.py`: A client-side demonstration tracking a random forest training and inference workflow using the SDK.
*   `benchmark_test.py`: Testing tool indicating the latency cost of integrating the auditing tool into inference workflows.
*   `index.html`: The HTML frontend designed to act as the live monitoring dashboard.

## 🛠️ Getting Started

### Prerequisites

You will need Python 3.7+ installed.
Ensure you install required dependencies. Typical requirements:
```bash
pip install fastapi uvicorn requests numpy scikit-learn pydantic
```

### 1. Start the Server

Run the central auditory hub to monitor requests:
```bash
python server.py
```
This will start a local server at `http://localhost:8000` (by default).
- Dashboard frontend: http://localhost:8000
- Audit API POST: http://localhost:8000/api/audit 

### 2. Run the Demo client

In a different terminal, run the simulated client script:
```bash
python friend_laptop_demo.py
```
This will train a test insurance model, infer a result upon a test case, and then intercept the request via the SDK to log the results on the server. Watch the server terminal to see the telemetry logged.

### 3. Benchmarking

You can monitor the overhead added by the perturbation approximation:
```bash
python benchmark_test.py
```

## 📖 Integrating the SDK

Integrating jAIstifier into a new or existing project is extremely straightforward. Simply use the `audit_decision` function inside your prediction pipeline:

```python
from jaistifier_sdk import audit_decision

# ...Your existing model & data...

# Intercept and Audit the prediction!
audit_result = audit_decision(
    model=my_model,
    X=my_input,            # numpy array or list, single sample
    feature_names=["age", "income", "credit_score"],
    server_url="http://localhost:8000/api/audit",
    model_name="My Production Model",
    model_type="classifier"
)
```

## 💡 Production Tip
As a note for production implementation: currently, the jAIstifier SDK blocks the main thread waiting for the HTTP POST request to finish. In high-load production environments, updating the payload transmission mechanism in `jaistifier_sdk.py` to work asynchronously via a background thread is highly recommended!
