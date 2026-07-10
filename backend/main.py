"""
NidrAlert Backend — FastAPI
Run:  uvicorn main:app --reload --port 8000
Install:  pip install fastapi uvicorn pymongo bcrypt pyjwt python-multipart
"""
from ai_process_router import router as ai_process_router

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
import jwt
import bcrypt
from pymongo import MongoClient
from bson import ObjectId
import os

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MONGO_URI        = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME          = "nidralert"
JWT_SECRET       = os.getenv("JWT_SECRET", "change_this_secret_in_production")
JWT_EXPIRE_HOURS = 24
ADMIN_ID         = "admin"
ADMIN_PASS       = "Admin@123"

# ─────────────────────────────────────────
# MONGODB
# ─────────────────────────────────────────
client       = MongoClient(MONGO_URI)
db           = client[DB_NAME]
users_col    = db["users"]
sessions_col = db["sessions"]
alerts_col   = db["alerts"]
settings_col = db["settings"]

# ─────────────────────────────────────────
# APP + CORS
# ─────────────────────────────────────────
app = FastAPI(title="NidrAlert API")
app.include_router(ai_process_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:3000"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def create_token(data: dict) -> str:
    payload = {**data, "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    return decode_token(credentials.credentials)

def get_admin_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    payload = decode_token(credentials.credentials)
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return payload

def serialize(doc: dict) -> dict:
    """Make MongoDB document JSON-safe."""
    doc["_id"] = str(doc["_id"])
    return doc

# ─────────────────────────────────────────
# PYDANTIC MODELS  (what the frontend sends)
# ─────────────────────────────────────────
class RegisterInput(BaseModel):
    name: str
    driverId: str
    password: str
    photo: Optional[str] = ""   # base64 from frontend

class LoginInput(BaseModel):
    driverId: str
    password: str

class AdminLoginInput(BaseModel):
    adminId: str
    password: str

class AlertInput(BaseModel):
    type: str            # "DROWSINESS" | "YAWNING" | "HEAD_DROP" | "DISTRACTION"
    ear: Optional[float] = None
    mar: Optional[float] = None

class SettingsInput(BaseModel):
    volume: Optional[int]  = 50
    alertType: Optional[str] = "sound"
    sound: Optional[str]   = None

class UserUpdateInput(BaseModel):
    name: Optional[str]     = None
    driverId: Optional[str] = None

class AddUserInput(BaseModel):
    name: str
    driverId: str
    password: str = "Temp@1234"
    photo: Optional[str] = ""


class AIAlertInput(BaseModel):
    """Logged by ai/main.py (local process, no JWT)."""
    driverId: str
    type: str
    ear: Optional[float] = None
    mar: Optional[float] = None


# ══════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════
@app.post("/api/auth/register")
def register(body: RegisterInput):
    if users_col.find_one({"driverId": body.driverId}):
        raise HTTPException(409, "Driver ID already exists")
    users_col.insert_one({
        "name":      body.name,
        "driverId":  body.driverId,
        "password":  hash_password(body.password),
        "photo":     body.photo,
        "createdAt": datetime.utcnow().isoformat(),
    })
    return {"message": "Registered successfully"}


@app.post("/api/auth/login")
def login(body: LoginInput):
    user = users_col.find_one({"driverId": body.driverId})
    if not user or not verify_password(body.password, user["password"]):
        raise HTTPException(401, "Invalid Driver ID or Password")
    token = create_token({"driverId": user["driverId"], "name": user["name"], "role": "driver"})
    return {
        "token": token,
        "driverData": {
            "name":     user["name"],
            "driverId": user["driverId"],
            "photo":    user.get("photo", ""),
        }
    }


@app.post("/api/auth/admin-login")
def admin_login(body: AdminLoginInput):
    if body.adminId != ADMIN_ID or body.password != ADMIN_PASS:
        raise HTTPException(401, "Invalid Admin Credentials")
    token = create_token({"adminId": ADMIN_ID, "role": "admin"})
    return {"token": token}


# ══════════════════════════════════════════
#  SESSIONS
# ══════════════════════════════════════════
@app.post("/api/sessions/start")
def start_session(user=Depends(get_current_user)):
    doc = {
        "driverId": user["driverId"],
        "start":    datetime.utcnow().isoformat(),
        "end":      None,
        "alerts":   0,
    }
    result = sessions_col.insert_one(doc)
    return {"sessionId": str(result.inserted_id), "start": doc["start"]}


@app.patch("/api/sessions/{session_id}/end")
def end_session(session_id: str, user=Depends(get_current_user)):
    r = sessions_col.update_one(
        {"_id": ObjectId(session_id), "driverId": user["driverId"]},
        {"$set": {"end": datetime.utcnow().isoformat()}}
    )
    if r.matched_count == 0:
        raise HTTPException(404, "Session not found")
    return {"message": "Session ended"}


@app.patch("/api/sessions/{session_id}/alert")
def increment_session_alert(session_id: str, user=Depends(get_current_user)):
    sessions_col.update_one(
        {"_id": ObjectId(session_id), "driverId": user["driverId"]},
        {"$inc": {"alerts": 1}}
    )
    return {"message": "Alert count updated"}


@app.get("/api/sessions/history")
def get_history(user=Depends(get_current_user)):
    did  = user["driverId"]
    docs = list(sessions_col.find({"driverId": did}, sort=[("start", -1)]))

    enriched = []
    for d in docs:
        ai = d.get("aiSummary") or {}

        # Resolve best available alert count — 4 fallback levels
        count = d.get("alerts")

        if not count and ai.get("total_alerts") is not None:
            count = int(ai["total_alerts"])

        if not count:
            count = (
                int(ai.get("drowsy_events")   or 0) +
                int(ai.get("yawn_count")       or 0) +
                int(ai.get("distraction_acts") or 0) +
                int(ai.get("head_drops")       or 0)
            )

        if not count:
            # Last resort: count from alerts_col between session start and end
            start_iso = d.get("start") or ""
            end_iso   = d.get("end")   or datetime.utcnow().isoformat()
            if start_iso:
                count = alerts_col.count_documents({
                    "driverId":  did,
                    "timestamp": {"$gte": start_iso, "$lte": end_iso},
                })

        d["alerts"] = int(count or 0)

        # Hoist aiSummary fields to top level so frontend reads them directly
        for field in ("drowsy_events", "yawn_count", "distraction_acts",
                      "head_drops", "duration_fmt", "baseline_ear",
                      "glasses_detected", "calibrated", "total_alerts"):
            if field not in d and field in ai:
                d[field] = ai[field]

        enriched.append(serialize(d))

    return enriched


# ── AI script posts its summary here (no auth — runs locally)
@app.post("/api/ai/session")
def save_ai_session(body: dict):
    """
    Merge AI summary into the existing session row (matched by sessionId + driverId).
    Fallback 1: match the most-recent open session for this driver (no sessionId).
    Fallback 2: insert a standalone doc so data is never lost.
    Always writes alerts count + hoisted breakdown fields so get_history sees them.
    """
    doc   = {**body, "savedAt": datetime.utcnow().isoformat()}
    sid   = doc.get("sessionId")
    did   = doc.get("driverId")
    total = int(doc.get("total_alerts") or 0)

    ai_summary = {
        k: v for k, v in doc.items()
        if k not in ("sessionId", "driverId", "savedAt")
    }

    update_payload = {
        "$set": {
            "end":      datetime.utcnow().isoformat(),
            "alerts":   total,          # top-level field — get_history reads this
            "aiSummary": ai_summary,    # full breakdown kept for detail views
            # Also hoist breakdown fields to top level for direct frontend access
            "drowsy_events":    doc.get("drowsy_events"),
            "yawn_count":       doc.get("yawn_count"),
            "distraction_acts": doc.get("distraction_acts"),
            "head_drops":       doc.get("head_drops"),
            "duration_fmt":     doc.get("duration_fmt"),
            "baseline_ear":     doc.get("baseline_ear"),
            "glasses_detected": doc.get("glasses_detected"),
            "calibrated":       doc.get("calibrated"),
        }
    }

    # ── Attempt 1: match by sessionId ────────────────────────────────────
    if sid and did:
        try:
            r = sessions_col.update_one(
                {"_id": ObjectId(sid), "driverId": did},
                update_payload,
            )
            if r.matched_count:
                return {"message": "Session updated with AI summary (by sessionId)"}
        except Exception:
            pass

    # ── Attempt 2: match the most-recent session for this driver ─────────
    if did:
        recent = sessions_col.find_one(
            {"driverId": did, "end": None},
            sort=[("start", -1)],
        )
        if recent:
            sessions_col.update_one({"_id": recent["_id"]}, update_payload)
            return {"message": "Session updated with AI summary (most-recent match)"}

    # ── Fallback: insert standalone doc (no session was open) ────────────
    sessions_col.insert_one(doc)
    return {"message": "AI session saved (standalone)"}


@app.post("/api/ai/alert")
def ai_script_log_alert(body: AIAlertInput):
    alerts_col.insert_one({
        "driverId":  body.driverId,
        "type":      body.type,
        "ear":       body.ear,
        "mar":       body.mar,
        "timestamp": datetime.utcnow().isoformat(),
    })
    return {"ok": True}


# ══════════════════════════════════════════
#  ALERTS
# ══════════════════════════════════════════
@app.post("/api/alerts")
def log_alert(body: AlertInput, user=Depends(get_current_user)):
    alerts_col.insert_one({
        "driverId":  user["driverId"],
        "type":      body.type,
        "ear":       body.ear,
        "mar":       body.mar,
        "timestamp": datetime.utcnow().isoformat(),
    })
    return {"message": "Alert logged"}


@app.get("/api/alerts")
def get_alerts(user=Depends(get_current_user)):
    docs = list(alerts_col.find({"driverId": user["driverId"]}, sort=[("timestamp", -1)]))
    return [serialize(d) for d in docs]


# ══════════════════════════════════════════
#  PROFILE
# ══════════════════════════════════════════
@app.get("/api/profile")
def get_profile(user=Depends(get_current_user)):
    doc = users_col.find_one({"driverId": user["driverId"]}, {"password": 0})
    if not doc:
        raise HTTPException(404, "User not found")
    return serialize(doc)


@app.patch("/api/profile")
def update_profile(body: dict, user=Depends(get_current_user)):
    body.pop("password", None)
    users_col.update_one({"driverId": user["driverId"]}, {"$set": body})
    return {"message": "Profile updated"}


# ══════════════════════════════════════════
#  DRIVER SETTINGS
# ══════════════════════════════════════════
@app.get("/api/settings")
def get_settings(user=Depends(get_current_user)):
    doc = settings_col.find_one({"driverId": user["driverId"]})
    if not doc:
        return {"volume": 50, "alertType": "sound", "sound": None}
    return serialize(doc)


@app.put("/api/settings")
def save_settings(body: SettingsInput, user=Depends(get_current_user)):
    settings_col.update_one(
        {"driverId": user["driverId"]},
        {"$set": body.dict()},
        upsert=True,
    )
    return {"message": "Settings saved"}


# ══════════════════════════════════════════
#  ADMIN
# ══════════════════════════════════════════
@app.get("/api/admin/stats")
def admin_stats(admin=Depends(get_admin_user)):
    return {
        "users": users_col.count_documents({}),
        "sessions": sessions_col.count_documents({}),
        "alerts": alerts_col.count_documents({}),
    }


@app.get("/api/admin/users")
def admin_get_users(admin=Depends(get_admin_user)):
    docs = list(users_col.find({}, {"password": 0}))
    return [serialize(d) for d in docs]


@app.get("/api/admin/users/{driver_id}")
def admin_search_user(driver_id: str, admin=Depends(get_admin_user)):
    doc = users_col.find_one({"driverId": driver_id}, {"password": 0})
    if not doc:
        raise HTTPException(404, "User not found")
    result = serialize(doc)
    result["totalAlerts"] = alerts_col.count_documents({"driverId": driver_id})
    return result


@app.post("/api/admin/users")
def admin_add_user(body: AddUserInput, admin=Depends(get_admin_user)):
    if users_col.find_one({"driverId": body.driverId}):
        raise HTTPException(409, "Driver ID already exists")
    users_col.insert_one({
        "name":      body.name,
        "driverId":  body.driverId,
        "password":  hash_password(body.password),
        "photo":     body.photo,
        "createdAt": datetime.utcnow().isoformat(),
    })
    return {"message": "User added"}


@app.patch("/api/admin/users/{driver_id}")
def admin_update_user(driver_id: str, body: UserUpdateInput, admin=Depends(get_admin_user)):
    users_col.update_one({"driverId": driver_id}, {"$set": body.dict(exclude_none=True)})
    return {"message": "User updated"}


@app.delete("/api/admin/users/{driver_id}")
def admin_delete_user(driver_id: str, admin=Depends(get_admin_user)):
    r = users_col.delete_one({"driverId": driver_id})
    if r.deleted_count == 0:
        raise HTTPException(404, "User not found")
    return {"message": "User deleted"}


@app.get("/api/admin/report/{driver_id}")
def admin_get_report(driver_id: str, admin=Depends(get_admin_user)):
    driver_sessions = list(sessions_col.find({"driverId": driver_id}, sort=[("start", 1)]))
    total_db = alerts_col.count_documents({"driverId": driver_id})

    def alerts_between(start_iso: str, end_iso: Optional[str]) -> int:
        if not start_iso:
            return 0
        q: dict = {"driverId": driver_id, "timestamp": {"$gte": start_iso}}
        if end_iso:
            q["timestamp"]["$lte"] = end_iso
        return alerts_col.count_documents(q)

    trend_sessions = driver_sessions[-15:]
    alert_trend: list[int] = []
    session_labels: list[str] = []
    for s in trend_sessions:
        st = s.get("start") or ""
        en = s.get("end")
        cnt = alerts_between(st, en)
        if cnt == 0:
            ai = s.get("aiSummary") or {}
            if isinstance(ai, dict) and ai.get("total_alerts") is not None:
                cnt = int(ai["total_alerts"])
            else:
                cnt = int(s.get("alerts") or 0)
        alert_trend.append(cnt)
        session_labels.append(st[:19].replace("T", " ") if st else "—")

    alerts_by_day: list[dict] = []
    for i in range(13, -1, -1):
        d = (datetime.utcnow() - timedelta(days=i)).date()
        ds = d.isoformat()
        nxt = d + timedelta(days=1)
        c = alerts_col.count_documents({
            "driverId": driver_id,
            "timestamp": {"$gte": f"{ds}T00:00:00", "$lt": f"{nxt.isoformat()}T00:00:00"},
        })
        alerts_by_day.append({"date": ds, "count": c})

    return {
        "driverId":       driver_id,
        "totalAlerts":    total_db,
        "alertTrend":     alert_trend,
        "sessionLabels":  session_labels,
        "alertsByDay":    alerts_by_day,
        "sessions":       [serialize(s) for s in driver_sessions],
    }


@app.get("/api/admin/settings")
def admin_get_settings(admin=Depends(get_admin_user)):
    doc = settings_col.find_one({"type": "global"})
    if not doc:
        return {"volume": 50, "alertType": "sound", "sound": None}
    return serialize(doc)


@app.put("/api/admin/settings")
def admin_save_settings(body: SettingsInput, admin=Depends(get_admin_user)):
    settings_col.update_one(
        {"type": "global"},
        {"$set": {**body.dict(), "type": "global"}},
        upsert=True,
    )
    return {"message": "Global settings saved"}