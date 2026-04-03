"""
AgriAI — File 2 of 4: FastAPI Cloud Server (Improved)
======================================================
Receives SMS from farmer → classifies disease → sends Tamil reply

Run:
    pip install -r requirements.txt
    uvicorn 2_server:app --host 0.0.0.0 --port 8000

SMS Gateway: MSG91 (India) or Twilio
    - Configure your gateway webhook to POST to /sms/receive
    - Set env vars in .env file

Improvements over v1:
    - SQLite persistence for pending confirmations + feedback (survives restarts)
    - HMAC webhook authentication (optional, set WEBHOOK_SECRET)
    - Per-phone rate limiting (max 10 SMS/hour)
    - PCA transform applied server-side to match training pipeline
"""

import os, json, pickle, re, hmac, hashlib, logging, sqlite3, time
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
log = logging.getLogger("agriai")

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH      = Path("agriai_model.pkl")
PCA_PATH        = Path("agriai_pca.pkl")
LABELS_PATH     = Path("agriai_labels.json")
DB_PATH         = Path("agriai_server.db")
CONF_HIGH       = 0.70   # above this → send diagnosis
CONF_MED        = 0.45   # above this → send with warning; below → ask re-photo
RATE_LIMIT      = 10     # max SMS per phone per hour
RATE_WINDOW     = 3600   # 1 hour in seconds

# SMS gateway sender number (set in .env)
AGRIAI_NUMBER   = os.getenv("AGRIAI_NUMBER", "1800AGRIAI")
WEBHOOK_SECRET  = os.getenv("WEBHOOK_SECRET", "")  # optional HMAC auth

# ── LOAD MODEL + PCA ─────────────────────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)
with open(LABELS_PATH, encoding="utf-8") as f:
    LABELS = json.load(f)

PCA_TRANSFORM = None
if PCA_PATH.exists():
    with open(PCA_PATH, "rb") as f:
        PCA_TRANSFORM = pickle.load(f)
    log.info(f"PCA loaded: {PCA_TRANSFORM.n_components_} components")

EMBED_DIM = LABELS.get("pca_dim", 64)
FULL_EMBED_DIM = LABELS.get("full_embed_dim", 1024)

log.info(f"Model loaded. Classes: {len(LABELS['id_to_info'])}, embed_dim: {EMBED_DIM}")

app = FastAPI(title="AgriAI SMS Server", version="2.0")


# ── SQLITE DATABASE ──────────────────────────────────────────────────────────
def init_db():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending (
            phone       TEXT PRIMARY KEY,
            embedding   TEXT NOT NULL,
            prediction  TEXT NOT NULL,
            location    TEXT,
            timestamp   TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            phone_hash  INTEGER NOT NULL,
            location    TEXT,
            prediction  TEXT NOT NULL,
            label_id    INTEGER NOT NULL,
            confidence  REAL NOT NULL,
            confirmed   INTEGER NOT NULL,
            embedding   TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rate_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            phone       TEXT NOT NULL,
            timestamp   REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rate_phone ON rate_log(phone, timestamp)")
    conn.commit()
    conn.close()

init_db()

@contextmanager
def get_db():
    """Thread-safe SQLite connection context manager."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── RATE LIMITING ─────────────────────────────────────────────────────────────
def check_rate_limit(phone: str) -> bool:
    """Returns True if the phone is under the rate limit."""
    now = time.time()
    with get_db() as db:
        # Clean old entries
        db.execute("DELETE FROM rate_log WHERE timestamp < ?", (now - RATE_WINDOW,))
        # Count recent
        row = db.execute(
            "SELECT COUNT(*) as cnt FROM rate_log WHERE phone = ? AND timestamp > ?",
            (phone, now - RATE_WINDOW)
        ).fetchone()
        if row["cnt"] >= RATE_LIMIT:
            return False
        # Log this request
        db.execute("INSERT INTO rate_log (phone, timestamp) VALUES (?, ?)", (phone, now))
    return True


# ── WEBHOOK AUTH ──────────────────────────────────────────────────────────────
def verify_webhook(request: Request, body: str) -> bool:
    """Verify HMAC signature if WEBHOOK_SECRET is configured."""
    if not WEBHOOK_SECRET:
        return True  # auth not configured → allow all
    sig = request.headers.get("X-Signature", "")
    expected = hmac.new(WEBHOOK_SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)


# ── SMS FORMAT ────────────────────────────────────────────────────────────────
# Farmer sends: LOC:TNJ|12,34,56,...(N ints scaled ×100)|CHK:789
# Server parses, classifies, replies in Tamil

def parse_sms(body: str) -> tuple[list[float] | None, str, int | None]:
    """
    Returns (embedding, location, checksum) or (None, '', None) on error.
    SMS format: LOC:TNJ|f0,f1,...fN|CHK:999
    """
    try:
        parts    = body.strip().split("|")
        loc      = parts[0].replace("LOC:", "").strip()
        raw_ints = [int(x) for x in parts[1].split(",")]
        chk_recv = int(parts[2].replace("CHK:", "").strip())

        # Accept both old (32-dim) and new (64-dim) formats
        if len(raw_ints) not in (32, 64, EMBED_DIM):
            log.warning(f"Unexpected embedding dim: {len(raw_ints)}")
            return None, loc, None

        # Verify checksum
        chk_calc = sum(raw_ints) % 999
        if chk_calc != chk_recv:
            log.warning(f"Checksum mismatch: got {chk_recv}, calc {chk_calc}")
            return None, loc, None

        embedding = [v / 100.0 for v in raw_ints]
        return embedding, loc, chk_recv
    except Exception as e:
        log.error(f"Parse error: {e}")
        return None, "", None


def classify(embedding: list[float]) -> dict:
    """Run PCA transform + XGBoost on embedding → disease prediction."""
    X = np.array(embedding).reshape(1, -1)

    # Apply PCA if available and input is full-dim
    if PCA_TRANSFORM is not None and X.shape[1] == FULL_EMBED_DIM:
        X = PCA_TRANSFORM.transform(X)
    elif X.shape[1] != EMBED_DIM:
        # Truncate or pad to expected dim
        if X.shape[1] > EMBED_DIM:
            X = X[:, :EMBED_DIM]
        else:
            X = np.pad(X, ((0,0), (0, EMBED_DIM - X.shape[1])))

    probs = MODEL.predict_proba(X)[0]
    pid   = int(np.argmax(probs))
    conf  = float(probs[pid])
    info  = LABELS["id_to_info"].get(str(pid), {
        "class": "unknown", "tamil": "தெரியவில்லை", "action": "விவசாய அலுவலரை அழைக்கவும்"
    })
    tier = "high" if conf >= CONF_HIGH else ("medium" if conf >= CONF_MED else "low")
    return {
        "class_en": info["class"],
        "tamil":    info["tamil"],
        "action":   info["action"],
        "conf":     conf,
        "tier":     tier,
        "label_id": pid,
    }


def build_reply(result: dict) -> str:
    """Format Tamil SMS reply. Stays under 320 chars (2 SMS units)."""
    if result["tier"] == "low":
        return (
            "AgriAI: படம் தெளிவற்றது.\n"
            "நல்ல வெளிச்சத்தில் மீண்டும் எடுத்து அனுப்பவும்.\n"
            "உதவி: 1800-XXX-XXXX"
        )
    pct  = int(result["conf"] * 100)
    warn = "\n[நடுத்தர நம்பிக்கை — விவசாயியை கலந்தாலோசிக்கவும்]" if result["tier"] == "medium" else ""
    return (
        f"AgriAI நோய்: {result['tamil']}\n"
        f"செய்க: {result['action']}\n"
        f"நம்பிக்கை: {pct}%{warn}\n"
        f"சரியா? YES அல்லது NO அனுப்பவும்."
    )


def send_sms(to: str, body: str):
    """
    Send SMS via MSG91 (India) or Twilio.
    Replace the stub below with your actual gateway SDK call.
    """
    # ── Twilio ──
    twilio_sid = os.getenv("TWILIO_SID")
    if twilio_sid:
        from twilio.rest import Client
        try:
            Client(twilio_sid, os.getenv("TWILIO_TOKEN")) \
                .messages.create(body=body, from_=AGRIAI_NUMBER, to=to)
        except Exception as e:
            log.error(f"Twilio error: {e}")
        return

    log.info(f"SMS → {to}: {body[:80]}...")


# ── PENDING STORE (SQLite) ────────────────────────────────────────────────────
def store_pending(phone: str, embedding: list, prediction: dict, location: str):
    with get_db() as db:
        db.execute(
            "INSERT OR REPLACE INTO pending (phone, embedding, prediction, location, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (phone, json.dumps(embedding), json.dumps(prediction), location,
             datetime.utcnow().isoformat())
        )

def pop_pending(phone: str) -> dict | None:
    with get_db() as db:
        row = db.execute("SELECT * FROM pending WHERE phone = ?", (phone,)).fetchone()
        if row:
            db.execute("DELETE FROM pending WHERE phone = ?", (phone,))
            return {
                "embedding":  json.loads(row["embedding"]),
                "prediction": json.loads(row["prediction"]),
                "location":   row["location"],
                "timestamp":  row["timestamp"],
            }
    return None


# ── FEEDBACK LOGGER (SQLite) ──────────────────────────────────────────────────
def log_feedback(phone: str, record: dict, confirmed: bool):
    with get_db() as db:
        db.execute(
            "INSERT INTO feedback (timestamp, phone_hash, location, prediction, "
            "label_id, confidence, confirmed, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record["timestamp"],
                hash(phone) % 99999,
                record["location"],
                record["prediction"]["class_en"],
                record["prediction"]["label_id"],
                round(record["prediction"]["conf"], 4),
                1 if confirmed else 0,
                "|".join(f"{v:.4f}" for v in record["embedding"]),
            )
        )
    log.info(f"Feedback logged: confirmed={confirmed}, class={record['prediction']['class_en']}")


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.post("/sms/receive")
async def receive_sms(
    request: Request,
    From: str = Form(...),
    Body: str = Form(...),
):
    """
    Webhook called by SMS gateway when farmer sends a message.
    Handles:
      - Feature vector SMS  →  classify + reply
      - YES confirmation    →  log correct label
      - NO confirmation     →  log wrong label
    """
    phone   = From.strip()
    body    = Body.strip()
    log.info(f"SMS from {phone}: {body[:80]}")

    # ── Webhook auth ──────────────────────────────────────────────────────────
    if not verify_webhook(request, body):
        log.warning(f"Auth failed for {phone}")
        raise HTTPException(status_code=403, detail="Invalid signature")

    # ── Rate limiting ─────────────────────────────────────────────────────────
    if not check_rate_limit(phone):
        log.warning(f"Rate limited: {phone}")
        send_sms(phone, "AgriAI: அதிக செய்திகள். தயவுசெய்து 1 மணி நேரம் காத்திருக்கவும்.")
        return PlainTextResponse("rate_limited")

    # ── Handle YES/NO feedback ────────────────────────────────────────────────
    if body.upper() in ("YES", "NO", "ஆம்", "இல்லை"):
        confirmed = body.upper() in ("YES", "ஆம்")
        record = pop_pending(phone)
        if record:
            log_feedback(phone, record, confirmed)
            reply = "நன்றி! உங்கள் கருத்து பதிவாகியது." if confirmed else \
                    "மன்னிக்கவும். மீண்டும் புகைப்படம் எடுத்து அனுப்பவும்."
        else:
            reply = "AgriAI: தாமதமான பதில். மீண்டும் புகைப்படம் அனுப்பவும்."
        send_sms(phone, reply)
        return PlainTextResponse("ok")

    # ── Handle feature vector SMS ─────────────────────────────────────────────
    embedding, loc, chk = parse_sms(body)

    if embedding is None:
        send_sms(phone,
            "AgriAI: SMS வடிவம் தவறு.\n"
            "AgriAI app மூலம் மீண்டும் அனுப்பவும்."
        )
        return PlainTextResponse("parse_error")

    result = classify(embedding)
    reply  = build_reply(result)

    # Store for YES/NO feedback (SQLite — survives restarts)
    store_pending(phone, embedding, result, loc)

    send_sms(phone, reply)
    log.info(f"Classified: {result['class_en']} ({result['conf']*100:.0f}%) → {phone}")
    return PlainTextResponse("ok")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "classes": len(LABELS["id_to_info"]),
        "embed_dim": EMBED_DIM,
        "pca": PCA_TRANSFORM is not None,
    }


@app.get("/stats")
async def stats():
    """Stats endpoint with live data from SQLite."""
    with get_db() as db:
        pending_count  = db.execute("SELECT COUNT(*) as c FROM pending").fetchone()["c"]
        feedback_count = db.execute("SELECT COUNT(*) as c FROM feedback").fetchone()["c"]
        confirmed      = db.execute("SELECT COUNT(*) as c FROM feedback WHERE confirmed=1").fetchone()["c"]
    return {
        "pending_confirmations": pending_count,
        "total_feedback":        feedback_count,
        "confirmed_correct":     confirmed,
        "model":                 str(MODEL_PATH),
        "server":                "running",
    }


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("2_server:app", host="0.0.0.0", port=8000, reload=False)
