"""
jAIstifier — SaaS Backend Server
=================================
A lightweight FastAPI server that acts as the central audit hub.
It receives SHAP audit payloads from remote SDKs and serves them
to the live monitoring dashboard.

Run with:  python server.py
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import uvicorn

app = FastAPI(title="jAIstifier Audit Server", version="1.0.0")

# ── CORS: Allow the frontend dashboard to access the API ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, lock this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-Memory Audit Store (upgradeable to MongoDB/Redis) ──
audit_log: List[Dict[str, Any]] = []


# ── Data Model for incoming audit payloads ──
class AuditPayload(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    model_name: str
    model_type: str
    input_features: Dict[str, Any]
    feature_names: List[str]
    prediction: Any
    prediction_label: Optional[str] = None
    confidence: Optional[float] = None
    shap_weights: Dict[str, float]
    top_positive_features: List[str]
    top_negative_features: List[str]
    timestamp: Optional[str] = None
    source_ip: Optional[str] = "remote-client"


# ── POST /api/audit — Receive audit from SDK ──
@app.post("/api/audit")
async def receive_audit(payload: AuditPayload):
    entry = payload.dict()
    entry["id"] = len(audit_log) + 1
    entry["received_at"] = datetime.now().isoformat()
    if not entry.get("timestamp"):
        entry["timestamp"] = entry["received_at"]

    audit_log.append(entry)

    print(f"\n{'='*60}")
    print(f"🛡️  jAIstifier AUDIT RECEIVED  #{entry['id']}")
    print(f"{'='*60}")
    print(f"  Model:      {entry['model_name']} ({entry['model_type']})")
    print(f"  Prediction:  {entry['prediction']} ({entry.get('prediction_label', 'N/A')})")
    print(f"  Confidence:  {entry.get('confidence', 'N/A')}")
    print(f"  Features:    {entry['input_features']}")
    print(f"  SHAP Weights:")
    for feat, weight in entry['shap_weights'].items():
        bar = "█" * int(abs(weight) * 20)
        sign = "+" if weight > 0 else "-"
        print(f"    {feat:>25s}: {sign}{abs(weight):.3f}  {bar}")
    print(f"  Top +ve:     {entry['top_positive_features']}")
    print(f"  Top -ve:     {entry['top_negative_features']}")
    print(f"  Time:        {entry['received_at']}")
    print(f"{'='*60}\n")

    return {"status": "ok", "audit_id": entry["id"]}


# ── GET /api/feed — Serve the latest audits to the dashboard ──
@app.get("/api/feed")
async def get_feed(limit: int = 20):
    # Return the most recent audits, newest first
    recent = list(reversed(audit_log[-limit:]))
    return {"audits": recent, "total": len(audit_log)}


# ── GET / — Health check ──
@app.get("/")
async def root():
    return {
        "service": "jAIstifier Audit Server",
        "version": "1.0.0",
        "total_audits": len(audit_log),
        "status": "running"
    }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🛡️  jAIstifier Audit Server")
    print("  Real-Time AI Decision Auditor")
    print("=" * 60)
    print("  Dashboard:  http://localhost:8000")
    print("  Audit API:  POST http://localhost:8000/api/audit")
    print("  Feed API:   GET  http://localhost:8000/api/feed")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
