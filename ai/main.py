
"""
NidrAlert AI detection — single entry script.

When started from the FastAPI app:
  - NIDRALERT_DRIVER_ID / NIDRALERT_SESSION_ID link sessions & alerts to the driver
  - NIDRALERT_HEADLESS=1 skips the OpenCV window; JPEG preview goes to the web UI

Run locally:  python main.py

── Detection improvements merged from nidra.py ───────────────────────────────
  1. Pitch sign fixed: head-DOWN now correctly raises pitch (was inverted).
  2. EAR smoothing removed from detection — raw EAR used for instant response.
     Smoothing kept only for the on-screen metric strip number.
  3. Smile-suppression guard removed — open mouth no longer blocks drowsiness.
  4. Yawning alert unconditional — not blocked by a prior drowsiness flag.
  5. Continuous throttled beep while alert is active, matching nidra.py.
  6. MediaPipe confidence thresholds raised to 0.5 for cleaner landmarks.
  7. draw_alert_banner upgraded to nidra.py centred pill style.
──────────────────────────────────────────────────────────────────────────────
"""
import base64
import cv2
import math
import os
import threading
import time
import urllib.request
from typing import Optional

import mediapipe as mp
import numpy as np
import requests
from datetime import datetime

import pygame
import pygame.mixer as _mixer
import pyttsx3

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── App integration (set by backend when spawning) ─────────────────
DRIVER_ID   = os.getenv("NIDRALERT_DRIVER_ID", "").strip()
SESSION_ID  = os.getenv("NIDRALERT_SESSION_ID", "").strip() or None
HEADLESS    = os.getenv("NIDRALERT_HEADLESS", "").strip() in ("1", "true", "yes")
BACKEND_URL = os.getenv("NIDRALERT_BACKEND_URL", "http://localhost:8000").rstrip("/")

# ── Preview throttle (JPEG to /api/ai/preview) ─────────────────────
_PREVIEW_INTERVAL_S = 0.22
_last_preview_t     = 0.0
_frame_counter      = 0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND COMMUNICATION
# ─────────────────────────────────────────────────────────────────────────────

def push_frame_to_backend(ear, mar, pitch, yaw, alert_text, calibrated, session_id=None):
    if alert_text == "CRITICAL: DROWSINESS":
        status = "drowsy"
    elif alert_text == "WARNING: YAWNING":
        status = "yawning"
    elif alert_text == "ALERT: FOCUS ON ROAD":
        status = "distracted"
    elif alert_text == "ALERT: HEAD DROP":
        status = "head_drop"
    elif not calibrated:
        status = "calibrating"
    else:
        status = "active"

    payload = {
        "ear":        round(ear, 4),
        "mar":        round(mar, 4),
        "pitch":      round(pitch, 4),
        "yaw":        round(yaw, 4),
        "status":     status,
        "alert":      alert_text,
        "calibrated": calibrated,
        "sessionId":  session_id,
    }
    try:
        requests.post(f"{BACKEND_URL}/api/ai/frame", json=payload, timeout=0.25)
    except Exception:
        pass


def push_preview_jpeg(bgr_frame):
    """Downscale + JPEG to backend for the React live view."""
    global _last_preview_t
    now = time.monotonic()
    if now - _last_preview_t < _PREVIEW_INTERVAL_S:
        return
    _last_preview_t = now
    try:
        h, w   = bgr_frame.shape[:2]
        max_w  = 480
        if w > max_w:
            scale     = max_w / float(w)
            bgr_frame = cv2.resize(bgr_frame, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 52])
        if not ok:
            return
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        requests.post(
            f"{BACKEND_URL}/api/ai/preview",
            json={"jpegBase64": b64},
            timeout=0.35,
        )
    except Exception:
        pass


def save_alert(alert_type: str, value: Optional[float] = None):
    """Persist driver-scoped alert for Report / Admin charts (via API)."""
    if not DRIVER_ID:
        return
    payload = {
        "driverId": DRIVER_ID,
        "type":     alert_type,
        "ear":      None,
        "mar":      None,
    }
    if value is not None:
        if alert_type == "DROWSINESS":
            payload["ear"] = round(float(value), 4)
        elif alert_type == "YAWNING":
            payload["mar"] = round(float(value), 4)
    try:
        requests.post(f"{BACKEND_URL}/api/ai/alert", json=payload, timeout=0.6)
    except Exception:
        pass


def save_session(doc: dict):
    doc = doc.copy()
    if DRIVER_ID:
        doc["driverId"] = DRIVER_ID
    if SESSION_ID:
        doc["sessionId"] = SESSION_ID

    api_ok = False
    try:
        resp = requests.post(f"{BACKEND_URL}/api/ai/session", json=doc, timeout=8)
        if resp.status_code == 200:
            api_ok = True
            print("✓ Session posted to backend API.")
        else:
            print(f"⚠  Backend API returned {resp.status_code}")
    except Exception as e:
        print(f"⚠  Could not reach backend: {e}")

    try:
        from pymongo import MongoClient
        _client  = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
        _client.server_info()
        db       = _client["nidralert"]
        sessions = db["sessions"]
    except Exception:
        sessions = None

    if not api_ok and sessions is not None:
        try:
            rid = sessions.insert_one(doc.copy())
            print(f"✓ Session saved to MongoDB (_id: {rid.inserted_id})")
        except Exception as e:
            print(f"⚠  MongoDB write failed: {e}")
    if not api_ok and sessions is None:
        with open(os.path.join(SCRIPT_DIR, "session_log.txt"), "a", encoding="utf-8") as f:
            f.write(str(doc) + "\n")
        print("✓ Session saved to session_log.txt (fallback).")

    try:
        requests.post(
            f"{BACKEND_URL}/api/ai/frame",
            json={
                "ear": 0.0, "mar": 0.0, "pitch": 0.0, "yaw": 0.0,
                "status": "waiting", "alert": None,
                "calibrated": False, "sessionId": None,
            },
            timeout=2,
        )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO
# ─────────────────────────────────────────────────────────────────────────────
_mixer.init(frequency=44100, size=-16, channels=1, buffer=512)


def _make_beep(freq: int, ms: int) -> pygame.mixer.Sound:
    sr          = 44100
    t           = np.linspace(0, ms / 1000, int(sr * ms / 1000), False)
    wave        = (np.sin(2 * np.pi * freq * t) * 28000).astype(np.int16)
    wave_stereo = np.repeat(wave.reshape(-1, 1), 2, axis=1)
    return pygame.sndarray.make_sound(wave_stereo)


_BEEP_DROWSY   = _make_beep(1500, 250)   # matches nidra.py's winsound.Beep(1500,250)
_BEEP_YAWN     = _make_beep(660,  300)
_BEEP_DISTRACT = _make_beep(980,  200)
_BEEP_HEAD     = _make_beep(1500, 300)

# Per-alert-type timestamp for throttling continuous beeps
_last_beep_t: dict = {}
_BEEP_INTERVAL_S   = 0.55     # minimum gap between repeated beeps of the same type


def _throttled_beep(key: str, sound: pygame.mixer.Sound):
    """Play `sound` at most once per _BEEP_INTERVAL_S for this alert type."""
    now = time.monotonic()
    if now - _last_beep_t.get(key, 0.0) >= _BEEP_INTERVAL_S:
        _last_beep_t[key] = now
        threading.Thread(target=sound.play, daemon=True).start()


def beep_drowsy():   _throttled_beep("drowsy",   _BEEP_DROWSY)
def beep_yawn():     _throttled_beep("yawn",     _BEEP_YAWN)
def beep_distract(): _throttled_beep("distract", _BEEP_DISTRACT)
def beep_head():     _throttled_beep("head",     _BEEP_HEAD)


# ─────────────────────────────────────────────────────────────────────────────
# TTS
# ─────────────────────────────────────────────────────────────────────────────
_tts      = pyttsx3.init()
_tts.setProperty("rate", 165)
_tts.setProperty("volume", 1.0)
_tts_lock = threading.Lock()
_tts_busy = False


def speak(text: str):
    global _tts_busy
    if _tts_busy:
        return

    def _run():
        global _tts_busy
        with _tts_lock:
            _tts_busy = True
            try:
                _tts.say(text)
                _tts.runAndWait()
            except Exception as e:
                print(f"[TTS] {e}")
            _tts_busy = False

    threading.Thread(target=_run, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DROWSY_PERCENTAGE        = 0.78
MAR_THRESHOLD            = 0.58
CONSECUTIVE_FRAMES_EYE   = 15
CONSECUTIVE_FRAMES_MOUTH = 20

# ── Head pose thresholds ──────────────────────────────────────────────────────
# FIX: In MediaPipe normalised coords nose.y INCREASES as head goes DOWN,
# so a downward nod produces a POSITIVE pitch value.
# Old main.py had -0.18/-0.08 (inverted) — head-drop never fired.
NOD_PITCH_THRESHOLD   =  0.20   # head DOWN trigger  (chin toward chest)
NOD_RECOVER_THRESHOLD =  0.08   # must recover above this to reset the state
NOD_DEBOUNCE_S        =  1.5    # min seconds between logged nod events

YAW_DISTRACTION_THR = 0.28   # lowered: nose-to-edge ratio is compressed, 0.35 was too strict
DISTRACTION_FRAMES  = 18    # lowered: ~0.65s at 30fps — was 40 (~1.3s), too slow to react
CALIBRATION_DURATION = 20

# ── EAR smoothing — display only, NOT detection ───────────────────────────────
EAR_SMOOTH_N       = 8
GLASSES_EAR_OFFSET = 0.03

# ── Frame enhancement ─────────────────────────────────────────────────────────
_clahe               = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
DARK_PIXEL_THRESHOLD = 80
DARK_GAMMA           = 1.8
_GAMMA_LUT           = np.array(
    [int(((i / 255.0) ** (1.0 / DARK_GAMMA)) * 255) for i in range(256)],
    dtype=np.uint8,
)


def enhance_frame(bgr: np.ndarray) -> np.ndarray:
    lab       = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b   = cv2.split(lab)
    mean_luma = float(np.mean(l))
    if mean_luma < DARK_PIXEL_THRESHOLD:
        l = _clahe.apply(l)
        if mean_luma < DARK_PIXEL_THRESHOLD * 0.5:
            l = cv2.LUT(l, _GAMMA_LUT)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return bgr


# ─────────────────────────────────────────────────────────────────────────────
# COLOURS / FONT
# ─────────────────────────────────────────────────────────────────────────────
C_RED    = (45,  45,  225)
C_AMBER  = (30,  180, 230)
C_WHITE  = (240, 240, 240)
C_ACCENT = (0,   180, 120)
C_DIM    = (80,  80,  80)
C_LABEL  = (160, 160, 160)
FONT     = cv2.FONT_HERSHEY_DUPLEX


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
session_running     = True
calibration_samples = []
baseline_ear        = None
is_calibrated       = False
_wearing_glasses    = False
_glasses_checks     = []

total_nods         = 0
total_yawns        = 0
total_distractions = 0
total_drowsy       = 0

yawn_flag     = False
nod_head_down = False        # state machine: True while head is down
last_nod_time = 0.0

f_eye = f_mouth = f_distract = 0

_ear_history = []            # display buffer — not used in detection
_yaw_history = []            # short smoother to stop noisy yaw resetting f_distract
YAW_SMOOTH_N = 4             # 4-frame average — quick but kills single-frame noise


# ─────────────────────────────────────────────────────────────────────────────
# CORE MATH
# ─────────────────────────────────────────────────────────────────────────────

def gdist(a, b):
    return math.dist((a.x, a.y), (b.x, b.y))


def calculate_ear(lm) -> float:
    """Raw un-smoothed EAR — used directly for detection (no lag)."""
    ear_l = (gdist(lm[385], lm[373]) + gdist(lm[387], lm[380])) / (2.0 * gdist(lm[362], lm[263]) or 1)
    ear_r = (gdist(lm[160], lm[144]) + gdist(lm[158], lm[153])) / (2.0 * gdist(lm[33],  lm[133]) or 1)
    raw   = (ear_l + ear_r) / 2.0
    return raw + (GLASSES_EAR_OFFSET if _wearing_glasses else 0.0)


def smooth_ear_display(raw: float) -> float:
    """Rolling average used ONLY for the on-screen number (keeps display stable)."""
    _ear_history.append(raw)
    if len(_ear_history) > EAR_SMOOTH_N:
        _ear_history.pop(0)
    return float(np.mean(_ear_history))


def smooth_yaw(raw: float) -> float:
    """Short rolling average so single noisy frames don't reset f_distract to 0."""
    _yaw_history.append(raw)
    if len(_yaw_history) > YAW_SMOOTH_N:
        _yaw_history.pop(0)
    return float(np.mean(_yaw_history))


def calculate_mar(lm) -> float:
    hz = gdist(lm[61], lm[291])
    return gdist(lm[13], lm[14]) / hz if hz else 0.0


def estimate_head_pose(lm):
    """
    pitch > 0  →  head tilting DOWN  (nose.y increases toward chin)
    pitch < 0  →  head tilting UP
    yaw   > 0  →  face turned LEFT
    yaw   < 0  →  face turned RIGHT
    """
    fh    = abs(lm[152].y - lm[10].y) or 1e-6
    pitch = (lm[1].y - (lm[10].y + lm[152].y) / 2.0) / fh
    dl    = gdist(lm[1], lm[234])
    dr    = gdist(lm[1], lm[454])
    yaw   = (dl - dr) / ((dl + dr) or 1e-6)
    return float(pitch), float(yaw)


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def alpha_rect(img, x1, y1, x2, y2, color, alpha):
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    block = np.full(roi.shape, color, dtype=np.uint8)
    cv2.addWeighted(block, alpha, roi, 1 - alpha, 0, roi)
    img[y1:y2, x1:x2] = roi


def draw_header(img, w, elapsed):
    alpha_rect(img, 0, 0, w, 62, (0, 0, 0), 0.65)
    cv2.line(img, (0, 62), (w, 62), C_ACCENT, 1)
    cv2.putText(img, "NIDRA", (22, 44),  FONT, 1.05, C_WHITE,  2, cv2.LINE_AA)
    cv2.putText(img, "ALERT", (148, 44), FONT, 1.05, C_ACCENT, 2, cv2.LINE_AA)
    timer = f"SESSION  {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
    tw    = cv2.getTextSize(timer, FONT, 0.55, 1)[0][0]
    cv2.putText(img, timer, (w - tw - 18, 40), FONT, 0.55, C_LABEL, 1, cv2.LINE_AA)
    if _wearing_glasses:
        cv2.putText(img, "SPECS MODE", (w - tw - 120, 40), FONT, 0.38, C_ACCENT, 1, cv2.LINE_AA)


def draw_calibration_bar(img, w, h, cal_rem):
    pct   = 1.0 - cal_rem / CALIBRATION_DURATION
    bar_w = int((w - 40) * pct)
    alpha_rect(img, 0, h - 38, w, h, (0, 0, 0), 0.55)
    cv2.rectangle(img, (20, h - 24), (w - 20, h - 12), C_DIM, -1)
    if bar_w > 4:
        cv2.rectangle(img, (20, h - 24), (20 + bar_w, h - 12), C_ACCENT, -1)
    label = f"CALIBRATING SENSORS — {int(cal_rem)}s remaining"
    cv2.putText(img, label, (22, h - 30), FONT, 0.42, (0, 200, 255), 1, cv2.LINE_AA)


def draw_alert_banner(img, w, h, text, color):
    """
    Centred translucent pill banner — nidra.py style.
    Larger text (scale 0.85), centred on screen, double border glow.
    """
    scale  = 0.85
    thick  = 2
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    pad_x, pad_y = 32, 18
    bw = tw + pad_x * 2
    bh = th + pad_y * 2
    bx = (w - bw) // 2
    by = h // 2 - bh // 2

    alpha_rect(img, bx, by, bx + bw, by + bh, (10, 10, 20), 0.72)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color, thick, cv2.LINE_AA)
    cv2.rectangle(img, (bx + 3, by + 3), (bx + bw - 3, by + bh - 3), color, 1, cv2.LINE_AA)
    tx = bx + pad_x
    ty = by + pad_y + th - 4
    cv2.putText(img, text, (tx, ty), FONT, scale, C_WHITE, thick, cv2.LINE_AA)
    cv2.circle(img, (bx + pad_x - 16, by + bh // 2), 6, color, -1)


def draw_end_button(img, w, h):
    bx1, by1, bx2, by2 = w - 172, h - 56, w - 16, h - 18
    alpha_rect(img, bx1, by1, bx2, by2, (20, 10, 40), 0.80)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), C_RED, 1, cv2.LINE_AA)
    label = "END SESSION"
    lw    = cv2.getTextSize(label, FONT, 0.50, 1)[0][0]
    cv2.putText(
        img, label,
        (bx1 + (bx2 - bx1 - lw) // 2, by1 + 26),
        FONT, 0.50, C_WHITE, 1, cv2.LINE_AA,
    )
    return (bx1, by1, bx2, by2)


def draw_metric_strip(img, w, h, ear_display, mar, pitch, yaw, baseline, calibrated):
    """Live readout (top-right). Uses display-smoothed EAR/yaw for stability."""
    if not calibrated or baseline is None:
        return
    lines = [
        (f"EAR  {ear_display:.2f} / {baseline * DROWSY_PERCENTAGE:.2f}", C_WHITE),
        (f"MAR  {mar:.2f}",                                               C_WHITE),
        (f"PITCH {pitch:+.2f}  YAW {yaw:+.2f}",                          C_LABEL),
        (f"DIST f={f_distract}/{DISTRACTION_FRAMES}",                     C_LABEL),
    ]
    sx = w - 210
    for i, (txt, col) in enumerate(lines):
        cv2.putText(img, txt, (sx, 88 + i * 20), FONT, 0.35, col, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# MOUSE CALLBACK
# ─────────────────────────────────────────────────────────────────────────────
_btn_rect_cache = (0, 0, 1, 1)


def on_click(event, x, y, flags, param):
    global session_running
    if event == cv2.EVENT_LBUTTONDOWN:
        bx1, by1, bx2, by2 = _btn_rect_cache
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            session_running = False


# ─────────────────────────────────────────────────────────────────────────────
# MODEL + CAMERA SETUP
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(SCRIPT_DIR, "face_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
if not os.path.exists(MODEL_PATH):
    print("Downloading face landmarker model (~30 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

# FIX: raised to 0.5 (nidra.py values) — reduces jittery/noisy landmarks
_face_options = mp_vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

video = None
for _cam_idx in range(3):
    cap = cv2.VideoCapture(_cam_idx, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if cap.isOpened():
        video = cap
        print(f"✓ Camera opened at index {_cam_idx}")
        break
    cap.release()

if video is None:
    raise RuntimeError("Cannot open webcam.")

video.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
video.set(cv2.CAP_PROP_GAIN, 0)

start_time = time.time()
WIN        = "NIDRALERT"

if not HEADLESS:
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_click)
    print("✓ OpenCV window mode — keep face visible for calibration.")
else:
    print("✓ Headless mode — live preview in the browser only.")

print("✓ Starting — keep face visible for 20s calibration.")
speak("NidrAlert started.")

cur_ear         = 0.0
cur_ear_display = 0.0   # smoothed — for metric strip only
cur_mar         = 0.0
cur_pitch       = 0.0
cur_yaw         = 0.0
_last_enhanced  = None   # cached enhanced frame — reused every 3 frames

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DETECTION LOOP
# ─────────────────────────────────────────────────────────────────────────────
with mp_vision.FaceLandmarker.create_from_options(_face_options) as detector:

    results = None   # seed so alternate-frame reuse is safe on frame 0

    while session_running:
        ret, frame = video.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        elapsed = time.time() - start_time
        ts_ms   = int(elapsed * 1000)

        threading.Thread(target=push_preview_jpeg, args=(frame.copy(),), daemon=True).start()
        # Re-enhance only every 3 frames — LAB+CLAHE is expensive, cached result is fine
        if _frame_counter % 3 == 0:
            _last_enhanced = enhance_frame(frame)
        enhanced = _last_enhanced if _last_enhanced is not None else frame

        try:
            if _frame_counter % 2 == 0:   # detect on every other frame — halves MP cost
                rgb     = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = detector.detect_for_video(mp_img, ts_ms)
            # else: reuse `results` from the previous frame
        except Exception:
            results = None

        alert_text  = None
        alert_color = C_RED

        if results and results.face_landmarks:
            lm = results.face_landmarks[0]

            try:
                # FIX: raw EAR goes straight to detection; display gets smoothed copy
                cur_ear         = calculate_ear(lm)
                cur_ear_display = smooth_ear_display(cur_ear)
                cur_mar         = calculate_mar(lm)
                cur_pitch, _raw_yaw = estimate_head_pose(lm)
                cur_yaw         = smooth_yaw(_raw_yaw)   # smoothed yaw for detection + display
            except Exception:
                pass

            # ── Calibration ──────────────────────────────────────────────
            if not is_calibrated:
                calibration_samples.append(cur_ear)

                if elapsed > CALIBRATION_DURATION * 0.5:
                    _glasses_checks.append(cur_ear)

                if elapsed >= CALIBRATION_DURATION:
                    baseline_ear  = float(np.mean(calibration_samples))
                    is_calibrated = True

                    if _glasses_checks:
                        late_mean = float(np.mean(_glasses_checks))
                        if baseline_ear > 0 and (late_mean / baseline_ear) < 0.93:
                            _wearing_glasses = True
                            baseline_ear    += GLASSES_EAR_OFFSET
                            print("✓ Glasses detected — EAR offset applied.")

                    print(f"✓ Calibrated. Baseline EAR = {baseline_ear:.4f}  glasses={_wearing_glasses}")
                    speak("Calibration complete. Stay alert.")

            # ── Detection ────────────────────────────────────────────────
            else:
                dyn_thr = baseline_ear * DROWSY_PERCENTAGE

                # 1. Drowsiness ──────────────────────────────────────────
                # FIX: smile-suppression removed. Open mouth (yawn) should not
                # block a closed-eye (drowsy) alert as it did in the old code.
                if cur_ear < dyn_thr:
                    f_eye += 1
                    if f_eye > CONSECUTIVE_FRAMES_EYE:
                        alert_text  = "CRITICAL: DROWSINESS"
                        alert_color = C_RED
                        if f_eye == CONSECUTIVE_FRAMES_EYE + 1:
                            total_drowsy += 1
                            speak("Wake up!")
                            save_alert("DROWSINESS", cur_ear)
                        # Continuous throttled beep (nidra.py behaviour)
                        beep_drowsy()
                else:
                    f_eye = 0

                # 2. Yawning ─────────────────────────────────────────────
                if cur_mar > MAR_THRESHOLD:
                    f_mouth += 1
                    if f_mouth > CONSECUTIVE_FRAMES_MOUTH:
                        # Only override if distraction isn't already showing
                        # (distraction evaluated next, so we park yawn and let
                        # distraction overwrite if needed in step 3)
                        alert_text  = "WARNING: YAWNING"
                        alert_color = C_AMBER
                        if not yawn_flag:
                            total_yawns += 1
                            yawn_flag    = True
                            speak("Yawning detected.")
                            save_alert("YAWNING", cur_mar)
                        beep_yawn()
                else:
                    f_mouth   = 0
                    yawn_flag = False

                # 3. Distraction (yaw) ───────────────────────────────────
                # Distraction always wins over yawning — looking away from the
                # road is more safety-critical than a yawn.
                if abs(cur_yaw) > YAW_DISTRACTION_THR:
                    f_distract += 1
                    if f_distract > DISTRACTION_FRAMES:
                        # Unconditionally overwrite — distraction > yawning priority
                        alert_text  = "ALERT: FOCUS ON ROAD"
                        alert_color = C_AMBER
                        if f_distract == DISTRACTION_FRAMES + 1:
                            total_distractions += 1
                            speak("Eyes on the road!")
                            save_alert("DISTRACTION", cur_yaw)
                        beep_distract()
                else:
                    f_distract = 0

                # 4. Head nod / drop (state machine) ─────────────────────
                # FIX: pitch sign corrected. head DOWN → pitch INCREASES (+ve).
                # Old code used `cur_pitch < -0.18` (never triggered downward).
                now = time.time()
                if not nod_head_down:
                    if cur_pitch > NOD_PITCH_THRESHOLD:      # head going DOWN
                        nod_head_down = True
                else:
                    if cur_pitch < NOD_RECOVER_THRESHOLD:    # head came back UP
                        nod_head_down = False
                        if now - last_nod_time > NOD_DEBOUNCE_S:
                            total_nods    += 1
                            last_nod_time  = now
                            alert_text     = "ALERT: HEAD DROP"
                            alert_color    = C_RED
                            speak("Wake up! Head drop detected.")
                            beep_head()
                            save_alert("HEAD_DROP", cur_pitch)

        else:
            # No face — reset counters
            f_eye = f_mouth = f_distract = 0
            yawn_flag = False

        # Push every frame (including calibration phase) to backend — non-blocking
        threading.Thread(
            target=push_frame_to_backend,
            args=(cur_ear, cur_mar, cur_pitch, cur_yaw, alert_text, is_calibrated, SESSION_ID),
            daemon=True,
        ).start()

        # ── Draw UI ──────────────────────────────────────────────────────
        if not HEADLESS:
            draw_header(img=frame, w=w, elapsed=elapsed)
            if not is_calibrated:
                draw_calibration_bar(frame, w, h, max(0.0, CALIBRATION_DURATION - elapsed))
            else:
                draw_metric_strip(frame, w, h, cur_ear_display, cur_mar,
                                  cur_pitch, cur_yaw, baseline_ear, is_calibrated)
            if alert_text:
                draw_alert_banner(frame, w, h, alert_text, alert_color)
            btn             = draw_end_button(frame, w, h)
            _btn_rect_cache = btn
            cv2.imshow(WIN, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
        else:
            time.sleep(0.02)

        _frame_counter += 1


# ─────────────────────────────────────────────────────────────────────────────
# SESSION SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
duration     = time.time() - start_time
total_alerts = total_drowsy + total_yawns + total_distractions + total_nods

summary = {
    "timestamp":        datetime.utcnow().isoformat() + "Z",
    "duration_seconds": round(duration, 1),
    "duration_fmt":     f"{int(duration // 60)}m {int(duration % 60)}s",
    "baseline_ear":     round(baseline_ear, 4) if baseline_ear else None,
    "glasses_detected": _wearing_glasses,
    "drowsy_events":    total_drowsy,
    "head_drops":       total_nods,
    "yawn_count":       total_yawns,
    "distraction_acts": total_distractions,
    "total_alerts":     total_alerts,
    "calibrated":       is_calibrated,
}

print(f"\n{'=' * 40}")
print("  NIDRALERT SESSION SUMMARY")
print(f"{'=' * 40}")
for k, v in summary.items():
    print(f"  {k:<22}: {v}")
print(f"{'=' * 40}\n")

save_session(summary)

_mixer.quit()
video.release()
if not HEADLESS:
    cv2.destroyAllWindows()