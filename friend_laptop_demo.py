"""
Friend's Laptop Demo — Simulating a Remote AI Model
=====================================================
This script simulates what your friend would run on THEIR laptop.
It trains a simple insurance claim classifier using scikit-learn,
makes a prediction, and then calls jAIstifier SDK to audit it.

Usage:
    1. Make sure server.py is running on the jAIstifier host machine
    2. Run this script: python friend_laptop_demo.py
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from jaistifier_sdk import audit_decision

# ──────────────────────────────────────────────
# STEP 1: Train a dummy insurance claim AI model
# ──────────────────────────────────────────────
print("=" * 60)
print("  🏥 Friend's Laptop: Insurance Claim AI Model")
print("=" * 60)

# Generate synthetic training data
# Features: [Age, Bill_Amount, Policy_Level, Treatment_Type, Prior_Claims]
# Policy_Level: 0=Basic, 1=Standard, 2=Premium
# Treatment_Type: 0=General, 1=Surgery, 2=ICU, 3=Diagnostics
# Target: 0=Denied, 1=Partially Approved, 2=Fully Approved

np.random.seed(42)
n_samples = 500

age = np.random.randint(18, 90, n_samples)
bill = np.random.randint(10000, 500000, n_samples)
policy = np.random.choice([0, 1, 2], n_samples)
treatment = np.random.choice([0, 1, 2, 3], n_samples)
prior_claims = np.random.randint(0, 10, n_samples)

# Create labels based on realistic rules
labels = []
for i in range(n_samples):
    score = 0
    score += (2 - policy[i]) * 2       # Lower policy = more likely denied
    score += (1 if age[i] > 60 else 0)  # Older = higher risk
    score += (1 if bill[i] > 200000 else 0)  # High bill = risk
    score += (1 if prior_claims[i] > 3 else 0)  # Many claims = risk
    score += (1 if treatment[i] == 2 else 0)  # ICU = risk

    if score >= 5:
        labels.append(0)  # Denied
    elif score >= 3:
        labels.append(1)  # Partially Approved
    else:
        labels.append(2)  # Fully Approved

X_train = np.column_stack([age, bill, policy, treatment, prior_claims])
y_train = np.array(labels)

# Train a Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"✅ Model trained on {n_samples} samples")
print(f"   Classes: {dict(zip([0,1,2], ['Denied','Partial','Approved']))}")

# ──────────────────────────────────────────────
# STEP 2: Make a real prediction (the grandmother's case)
# ──────────────────────────────────────────────
print("\n" + "-" * 60)
print("📋 New Claim: 78-year-old, ₹2,00,000 bill, Standard policy, ICU")
print("-" * 60)

test_input = [32, 200000, 1, 2, 2]  # Age=78, Bill=2L, Standard, ICU, 2 prior claims
feature_names = ["Patient_Age", "Bill_Amount", "Policy_Level", "Treatment_Type", "Prior_Claims"]
prediction_labels = {0: "DENIED", 1: "PARTIALLY APPROVED", 2: "FULLY APPROVED"}

# The model makes its decision
raw_prediction = model.predict([test_input])[0]
print(f"🤖 AI Decision: {prediction_labels[raw_prediction]}")

# ──────────────────────────────────────────────
# STEP 3: jAIstifier SDK intercepts and audits!
# ──────────────────────────────────────────────
print("\n" + "-" * 60)
print("🛡️  jAIstifier SDK: Intercepting decision for audit...")
print("-" * 60)

# Change server_url to your jAIstifier server's IP if running on another machine
# e.g., server_url="http://192.168.1.5:8000/api/audit"
audit_result = audit_decision(
    model=model,
    X=test_input,
    feature_names=feature_names,
    server_url="http://localhost:8000/api/audit",
    model_name="Insurance Claim Classifier",
    model_type="classifier",
    prediction_labels=prediction_labels,
    num_perturbations=100,
)

print("\n" + "=" * 60)
print("✅ Done! Check the jAIstifier dashboard to see the audit.")
print("=" * 60)
