import os
import time
import json
import pickle
import hashlib
import hmac
import base64
from typing import Dict, List, Optional

from flask import Flask, request
from firebase_admin import credentials, firestore, initialize_app
import xgboost as xgb
import numpy as np

# ── CONFIG ───────────────────────────────────────────────────────────────────
# Initialize Firebase Admin
# If running in Cloud Functions, it uses default credentials.
# For local testing with a key, use: credentials.Certificate("path/to/key.json")
initialize_app()
db = firestore.client()

# Constants
AGRIAI_NUMBER = os.environ.get("AGRIAI_NUMBER", "1800AGRIAI")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
RATE_LIMIT_S = 3600  # 1 hour
MAX_SMS_PER_HOUR = 10

# Load Model Artifacts (Expected to be in the same directory as main.py)
MODEL_PATH = "agriai_model.pkl"
PCA_PATH = "agriai_pca.pkl"
LABELS_PATH = "agriai_labels.json"

model = None
pca = None
labels_list = None

def load_artifacts():
    global model, pca, labels_list
    if model is None and os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    if pca is None and os.path.exists(PCA_PATH):
        with open(PCA_PATH, "rb") as f:
            pca = pickle.load(f)
    if labels_list is None and os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r") as f:
            labels_list = json.load(f)

# ── HLPER FUNCTIONS ──────────────────────────────────────────────────────────
def get_phone_hash(phone: str) -> str:
    return hashlib.sha256(phone.encode()).hexdigest()

def check_rate_limit(phone: str) -> bool:
    """Check if the phone has exceeded the SMS rate limit using Firestore."""
    now = time.time()
    one_hour_ago = now - RATE_LIMIT_S
    
    # Query Firestore for recent logs from this phone
    docs = db.collection("rate_log") \
             .where("phone", "==", phone) \
             .where("timestamp", ">", one_hour_ago) \
             .stream()
    
    count = sum(1 for _ in docs)
    if count >= MAX_SMS_PER_HOUR:
        return False
        
    # Log this attempt
    db.collection("rate_log").add({
        "phone": phone,
        "timestamp": now
    })
    return True

def verify_signature(body: str, signature: str) -> bool:
    if not WEBHOOK_SECRET: return True
    expected = hmac.new(
        WEBHOOK_SECRET.encode(),
        body.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)

def send_sms_stub(to: str, text: str):
    """Integration for Twilio SMS gateway API."""
    print(f"DEBUG: Sending SMS to {to}: {text}")
    twilio_sid = os.environ.get("TWILIO_SID")
    if twilio_sid:
        try:
            from twilio.rest import Client
            Client(twilio_sid, os.environ.get("TWILIO_TOKEN")) \
                .messages.create(body=text, from_=AGRIAI_NUMBER, to=to)
        except Exception as e:
            print(f"Twilio error: {e}")

# ── FLASK APP ────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route('/sms', methods=['GET', 'POST'])
def handle_sms():
    """Entry point for Flask App (Render Webhook)."""
    load_artifacts()
    
    if request.method != "POST":
        return "Only POST allowed", 405

    # Webhook Authentication
    signature = request.headers.get("X-Signature", "")
    body_text = request.get_data(as_text=True)
    if not verify_signature(body_text, signature):
        return "Unauthorized", 401

    data = request.get_json(silent=True) or request.form.to_dict()
    sender = data.get("sender") or data.get("From")
    message = data.get("message") or data.get("Body")

    if not sender or not message:
        return "Missing sender/message", 400

    # Rate Limiting
    if not check_rate_limit(sender):
        send_sms_stub(sender, "Error: Rate limit exceeded. Try again in 1 hour.")
        return "Rate Limited", 429

    # 1. Handle Confirmation (YES/NO)
    msg_clean = message.strip().upper()
    if msg_clean in ["YES", "NO"]:
        pending_ref = db.collection("pending").document(sender)
        pending_doc = pending_ref.get()
        
        if pending_doc.exists:
            p = pending_doc.to_dict()
            db.collection("feedback").add({
                "timestamp": time.time(),
                "phone_hash": get_phone_hash(sender),
                "location": p.get("location"),
                "prediction": p.get("prediction"),
                "label_id": p.get("label_id"),
                "confidence": p.get("confidence"),
                "confirmed": 1 if msg_clean == "YES" else 0,
                "embedding": p.get("embedding")
            })
            pending_ref.delete()
            reply = "நன்றி! உங்கள் பதில் சேமிக்கப்பட்டது." if msg_clean == "YES" else "நன்றி. உங்கள் கருத்தை பதிவு செய்துள்ளோம்."
            send_sms_stub(sender, reply)
            return "Feedback logged", 200
        return "No pending request", 200

    # 2. Handle Image Embedding (LOC:NAME|VAL,VAL...)
    if "|" in message and ":" in message:
        try:
            head, vals_raw = message.split("|", 1)
            loc = head.split(":")[1] if ":" in head else "Unknown"
            vals = [float(v) for v in vals_raw.split(",")]
            
            # Apply PCA if 1024-dim
            if len(vals) == 1024 and pca:
                vals = pca.transform([vals])[0]
            
            # Predict
            dmatrix = xgb.DMatrix([vals])
            probs = model.predict(dmatrix)
            label_id = int(np.argmax(probs))
            conf = float(np.max(probs))
            disease = labels_list[label_id]
            
            # Firestore Store Pending
            db.collection("pending").document(sender).set({
                "timestamp": time.time(),
                "location": loc,
                "prediction": disease,
                "label_id": label_id,
                "confidence": conf,
                "embedding": vals_raw
            })

            reply = f"நுண்ணறிவு முடிவு: {disease} ({conf:.1%}). இது சரியா? (YES/NO என பதிலளிக்கவும்)"
            send_sms_stub(sender, reply)
            return "Classified", 200
            
        except Exception as e:
            print(f"Error: {e}")
            return f"Processing Error: {str(e)}", 500

    return "Invalid format", 400
