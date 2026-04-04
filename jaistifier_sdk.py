"""
jAIstifier SDK — Lightweight AI Decision Auditor
==================================================
This is the SDK file that your friend installs on their machine.
It wraps around ANY scikit-learn compatible model, computes real 
SHAP-like feature importance via perturbation, and sends the 
audit payload to the jAIstifier server.

Usage:
    from jaistifier_sdk import audit_decision
    
    audit_decision(
        model=my_model,
        X=input_data,            # numpy array or list, single sample
        feature_names=["age", "income", ...],
        server_url="http://<SERVER_IP>:8000/api/audit",
        model_name="My Insurance Model",
    )
"""

import numpy as np
import requests
import json
from datetime import datetime


def audit_decision(
    model,
    X,
    feature_names: list,
    server_url: str = "http://localhost:8000/api/audit",
    model_name: str = "Unknown Model",
    model_type: str = "classifier",
    prediction_labels: dict = None,
    num_perturbations: int = 50,
):
    """
    Intercepts an AI model's decision, computes feature importance
    via perturbation-based SHAP approximation, and sends the audit
    to the jAIstifier server.

    Parameters:
    -----------
    model : sklearn-compatible model
        Must have a .predict() method. If classifier, should also 
        have .predict_proba().
    X : array-like
        A single input sample (1D array or 2D with shape (1, n_features)).
    feature_names : list of str
        Human-readable names for each feature.
    server_url : str
        The jAIstifier server endpoint.
    model_name : str
        A friendly name for the model being audited.
    model_type : str
        "classifier" or "regressor".
    prediction_labels : dict, optional
        Maps prediction indices to human labels, e.g. {0: "Denied", 1: "Approved"}.
    num_perturbations : int
        Number of random perturbations per feature for SHAP estimation.
    """

    X = np.array(X).flatten()
    n_features = len(X)

    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) != input features ({n_features})"
        )

    # ── Step 1: Get the model's original prediction ──
    X_input = X.reshape(1, -1)
    original_pred = model.predict(X_input)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)[0]
        confidence = float(np.max(proba))
    
    prediction_label = None
    if prediction_labels and original_pred in prediction_labels:
        prediction_label = prediction_labels[original_pred]
    elif prediction_labels:
        prediction_label = prediction_labels.get(int(original_pred), str(original_pred))

    print(f"\n🤖 Model Prediction: {original_pred} ({prediction_label or 'N/A'})")
    print(f"   Confidence: {confidence:.3f}" if confidence else "")
    print(f"   Input: {dict(zip(feature_names, X))}")

    # ── Step 2: SHAP-like Perturbation Analysis ──
    # For each feature, we perturb it many times and measure
    # how much the prediction probability changes.
    print(f"\n🛡️  jAIstifier: Running {num_perturbations} perturbations per feature...")

    shap_weights = {}

    for i, feat_name in enumerate(feature_names):
        prob_diffs = []

        for _ in range(num_perturbations):
            X_perturbed = X.copy()

            # Perturb this feature: replace with random value from a
            # reasonable range (mean ± 2*std approximation using the value itself)
            original_val = X[i]
            if original_val == 0:
                noise_range = 1.0
            else:
                noise_range = abs(original_val) * 0.5

            X_perturbed[i] = original_val + np.random.uniform(
                -noise_range, noise_range
            )

            X_pert_input = X_perturbed.reshape(1, -1)

            if hasattr(model, "predict_proba"):
                original_proba = model.predict_proba(X_input)[0]
                perturbed_proba = model.predict_proba(X_pert_input)[0]
                # Measure how the probability of the chosen class changed
                pred_class = int(original_pred)
                diff = original_proba[pred_class] - perturbed_proba[pred_class]
            else:
                # For regressors, measure the output change
                original_out = model.predict(X_input)[0]
                perturbed_out = model.predict(X_pert_input)[0]
                diff = original_out - perturbed_out

            prob_diffs.append(diff)

        # The average absolute change = how "important" this feature is
        # The sign of the mean tells us directionality
        mean_diff = float(np.mean(prob_diffs))
        shap_weights[feat_name] = round(mean_diff, 4)

    # ── Step 3: Sort features by impact ──
    sorted_features = sorted(shap_weights.items(), key=lambda x: abs(x[1]), reverse=True)
    top_positive = [f for f, w in sorted_features if w > 0][:3]
    top_negative = [f for f, w in sorted_features if w <= 0][:3]

    print(f"\n📊 Feature Importance (SHAP Weights):")
    for feat, weight in sorted_features:
        bar_len = int(abs(weight) * 100)
        bar = "█" * min(bar_len, 40)
        sign = "+" if weight > 0 else "-"
        print(f"   {feat:>25s}: {sign}{abs(weight):.4f}  {bar}")

    # ── Step 4: Build and send the audit payload ──
    payload = {
        "model_name": model_name,
        "model_type": model_type,
        "input_features": {name: float(val) for name, val in zip(feature_names, X)},
        "feature_names": feature_names,
        "prediction": int(original_pred) if isinstance(original_pred, (np.integer, int)) else float(original_pred),
        "prediction_label": prediction_label,
        "confidence": confidence,
        "shap_weights": shap_weights,
        "top_positive_features": top_positive,
        "top_negative_features": top_negative,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n📡 Sending audit to jAIstifier server: {server_url}")

    try:
        response = requests.post(server_url, json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Audit stored! ID: #{result.get('audit_id', '?')}")
        else:
            print(f"⚠️  Server returned status {response.status_code}: {response.text}")
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to jAIstifier server at {server_url}")
        print(f"   Make sure the server is running: python server.py")
    except Exception as e:
        print(f"❌ Error sending audit: {e}")

    return payload
