"""
ai_process_router.py  — lives next to FastAPI main.py (backend folder).

Endpoints:
    POST /api/ai/start    – spawn ai/main.py (OpenCV window + live /api/ai/frame)
    POST /api/ai/stop     – terminate the AI subprocess
    GET  /api/ai/status   – is it running?
    POST /api/ai/frame    – AI script pushes live data each frame
    GET  /api/ai/frame    – React polls this

The AI script runs in a separate process (and OpenCV always uses its own window).
On Windows we avoid CREATE_NEW_CONSOLE so you do not get an extra terminal window.
"""

import subprocess
import sys
import os
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

# Integrated script: posts frames to the backend and saves sessions via the API.
AI_SCRIPT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ai", "main.py")
)

# ── Subprocess singleton ────────────────────────────────────────────
_ai_process: subprocess.Popen | None = None

# ── Live frame state ────────────────────────────────────────────────
_latest_frame: dict = {
    "ear":        0.0,
    "mar":        0.0,
    "pitch":      0.0,
    "yaw":        0.0,
    "status":     "waiting",
    "alert":      None,
    "calibrated": False,
    "sessionId":  None,
}
_latest_preview: str | None = None


class StartAIRequest(BaseModel):
    """Forwarded to the AI process via environment (see NIDRALERT_* in ai/main.py)."""
    driverId: Optional[str] = None
    sessionId: Optional[str] = None


class PreviewInput(BaseModel):
    jpegBase64: str


class FrameInput(BaseModel):
    ear:        float
    mar:        float
    pitch:      float
    yaw:        float
    status:     str
    alert:      Optional[str] = None
    calibrated: bool          = False
    sessionId:  Optional[str] = None


def _is_running() -> bool:
    return _ai_process is not None and _ai_process.poll() is None


# ── Process control ─────────────────────────────────────────────────

def _win_subprocess_flags() -> int:
    """Hide the extra console on Windows; OpenCV still shows its own window."""
    if sys.platform != "win32":
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


@router.post("/api/ai/start")
def start_ai(raw: dict = Body(default={})):
    req = StartAIRequest.model_validate(raw)
    global _ai_process, _latest_frame, _latest_preview

    if _is_running():
        return {"ok": True, "message": "AI already running", "pid": _ai_process.pid}

    if not os.path.exists(AI_SCRIPT_PATH):
        return JSONResponse(
            status_code=404,
            content={
                "ok": False,
                "message": (
                    f"AI script not found at: {AI_SCRIPT_PATH}. "
                    "Expected integrated script at ai/main.py."
                )
            }
        )

    _latest_preview = None
    _latest_frame = {
        "ear": 0.0, "mar": 0.0, "pitch": 0.0, "yaw": 0.0,
        "status": "calibrating", "alert": None,
        "calibrated": False, "sessionId": req.sessionId,
    }

    child_env = os.environ.copy()
    child_env["NIDRALERT_HEADLESS"] = "1"
    if req.driverId:
        child_env["NIDRALERT_DRIVER_ID"] = req.driverId
    if req.sessionId:
        child_env["NIDRALERT_SESSION_ID"] = req.sessionId

    try:
        _ai_process = subprocess.Popen(
            [sys.executable, AI_SCRIPT_PATH],
            env=child_env,
            cwd=os.path.dirname(AI_SCRIPT_PATH),
            creationflags=_win_subprocess_flags(),
        )
        return {"ok": True, "message": "AI detection started", "pid": _ai_process.pid}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "message": str(e)})


@router.post("/api/ai/stop")
def stop_ai():
    global _ai_process, _latest_frame, _latest_preview

    if not _is_running():
        _latest_frame["status"] = "waiting"
        _latest_preview = None
        return {"ok": True, "message": "AI was not running"}

    try:
        _ai_process.terminate()
        _ai_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _ai_process.kill()
    finally:
        _ai_process = None

    _latest_preview = None
    _latest_frame = {
        "ear": 0.0, "mar": 0.0, "pitch": 0.0, "yaw": 0.0,
        "status": "waiting", "alert": None,
        "calibrated": False, "sessionId": None,
    }
    return {"ok": True, "message": "AI detection stopped"}


@router.get("/api/ai/status")
def ai_status():
    running = _is_running()
    return {"running": running, "pid": _ai_process.pid if running else None}


# ── Live frame endpoints ────────────────────────────────────────────

@router.post("/api/ai/preview")
def push_preview(body: PreviewInput):
    global _latest_preview
    _latest_preview = f"data:image/jpeg;base64,{body.jpegBase64}"
    return {"ok": True}


@router.post("/api/ai/frame")
def push_frame(body: FrameInput):
    """Called by ai/main.py every frame."""
    global _latest_frame
    _latest_frame = body.model_dump() if hasattr(body, "model_dump") else body.dict()
    return {"ok": True}


@router.get("/api/ai/frame")
def get_frame():
    """Polled by React; includes `preview` data URL when headless JPEG stream is active."""
    return {**_latest_frame, "preview": _latest_preview}
