import requests
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import math, time, threading, os, urllib.request
import numpy as np
from datetime import datetime
import pygame
import pygame.mixer as _mixer
import pyttsx3

# ─────────────────────────────────────────────
# AUDIO SETUP
# ─────────────────────────────────────────────
_mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

def _make_beep(freq: int, ms: int) -> pygame.mixer.Sound:
    sr   = 44100
    t    = np.linspace(0, ms / 1000, int(sr * ms / 1000), False)
    wave = (np.sin(2 * np.pi * freq * t) * 28000).astype(np.int16)
    wave_stereo = np.repeat(wave.reshape(-1, 1), 2, axis=1)
    return pygame.sndarray.make_sound(wave_stereo)

_BEEP_YAWN     = _make_beep(660, 300)
_BEEP_DISTRACT = _make_beep(980, 200)

def _double_beep(sound: pygame.mixer.Sound):
    """Play the beep twice with a short gap — non-blocking."""
    def _play():
        sound.play()
        time.sleep(0.32)
        sound.play()
    threading.Thread(target=_play, daemon=True).start()

def beep_yawn():
    _double_beep(_BEEP_YAWN)

def beep_distract():
    _double_beep(_BEEP_DISTRACT)

# ─────────────────────────────────────────────
# TTS
# ─────────────────────────────────────────────
_tts      = pyttsx3.init()
_tts.setProperty('rate',   165)
_tts.setProperty('volume', 1.0)
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

# ─────────────────────────────────────────────
# MONGODB
# ─────────────────────────────────────────────
try:
    from pymongo import MongoClient
    _client  = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    _client.server_info()
    _db      = _client["nidralert"]
    sessions = _db["sessions"]
    alerts   = _db["alerts"]
    MONGO_OK = True
    print("✓ MongoDB connected.")
except Exception as e:
    MONGO_OK = False
    print(f"⚠  MongoDB unavailable ({e}). Falling back to alert_log.txt.")

def save_alert(alert_type: str, value: float = None):
    doc = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "alert":     alert_type,
        "value":     round(value, 4) if value is not None else None,
    }
    if MONGO_OK:
        try:
            alerts.insert_one(doc)
            print(f"  [DB] Alert saved: {alert_type}")
            return
        except Exception as e:
            print(f"  [DB] Write failed: {e}")
    with open("alert_log.txt", "a", encoding="utf-8") as f:
        f.write(str(doc) + "\n")

def save_session_summary(doc: dict):
    if MONGO_OK:
        try:
            sessions.insert_one(doc)
            print("✓ Session summary saved to MongoDB.")
            return
        except Exception as e:
            print(f"⚠  Session summary write failed: {e}")
    with open("alert_log.txt", "a", encoding="utf-8") as f:
        f.write("SESSION: " + str(doc) + "\n")
    print("✓ Session summary saved to alert_log.txt.")

# ─────────────────────────────────────────────
# BACKEND API  (push live data to FastAPI → React)
# ─────────────────────────────────────────────
BACKEND_URL = "http://localhost:8000"

def push_frame_to_backend(ear, mar, pitch, yaw, alert_text, calibrated, session_id=None):
    """Push the current frame's detection result to FastAPI backend (non-blocking)."""
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
        requests.post(f"{BACKEND_URL}/api/ai/frame", json=payload, timeout=0.3)
    except Exception:
        pass  # Never let a backend failure crash the detection loop

def save_session(doc: dict):
    """Persist session summary to MongoDB + backend API + fallback file."""
    # 1. MongoDB
    if MONGO_OK:
        try:
            sessions.insert_one(doc.copy())
            print(f"✓ Session saved to MongoDB.")
        except Exception as e:
            print(f"⚠  MongoDB write failed: {e}")

    # 2. POST to FastAPI so React History page sees it
    try:
        resp = requests.post(f"{BACKEND_URL}/api/ai/session", json=doc, timeout=5)
        if resp.status_code == 200:
            print("✓ Session posted to backend API.")
        else:
            print(f"⚠  Backend API returned {resp.status_code}")
    except Exception as e:
        print(f"⚠  Could not reach backend: {e}")
        with open("session_log.txt", "a", encoding="utf-8") as f:
            f.write(str(doc) + "\n")
        print("✓ Session saved to session_log.txt (fallback).")

    # 3. Clear live frame so React shows "waiting"
    try:
        requests.post(f"{BACKEND_URL}/api/ai/frame", json={
            "ear": 0.0, "mar": 0.0, "pitch": 0.0, "yaw": 0.0,
            "status": "waiting", "alert": None,
            "calibrated": False, "sessionId": None,
        }, timeout=2)
    except Exception:
        pass

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DROWSY_PERCENTAGE        = 0.72
MAR_THRESHOLD            = 0.45
CONSECUTIVE_FRAMES_EYE   = 20   # ~0.67s at 30fps — was 12 (~0.4s), reduces false triggers
CONSECUTIVE_FRAMES_MOUTH = 28   # ~0.93s at 30fps — was 20 (~0.67s)
NOD_PITCH_THRESHOLD      = -0.18
NOD_RECOVER_THRESHOLD    = -0.08
NOD_DEBOUNCE_S           = 2.0
YAW_DISTRACTION_THR      = 0.40
DISTRACTION_FRAMES       = 50
CALIBRATION_DURATION     = 20

# ── EAR smoothing ─────────────────────────────────────────────────────────────
# Glasses occlude eyelid-edge landmarks → artificially lower EAR.
# Longer history (8) smooths out single-frame noise from lens reflections.
# GLASSES_EAR_OFFSET compensates for systematic underestimation.
EAR_SMOOTH_N       = 8
GLASSES_EAR_OFFSET = 0.03   # added to raw EAR once glasses are detected

# ── Smile suppression ─────────────────────────────────────────────────────────
# When smiling, raised cheeks compress lower eyelid landmarks and drop EAR
# even though the eyes are open — identical signal to drowsiness.
# Fix: if MAR is above this threshold the person is smiling/talking,
# so we reset f_eye and refuse to increment the drowsy counter.
SMILE_MAR_SUPPRESS = 0.30   # MAR above this = mouth active → ignore EAR drop

# ── Low-light preprocessing ───────────────────────────────────────────────────
# CLAHE on the LAB luminance channel lifts local contrast without blowing out
# highlights — more robust than a flat brightness boost.
_clahe               = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
DARK_PIXEL_THRESHOLD = 80     # mean luma below this → apply CLAHE
DARK_GAMMA           = 1.8    # extra gamma lift in very dark scenes

_GAMMA_LUT = np.array(
    [int(((i / 255.0) ** (1.0 / DARK_GAMMA)) * 255) for i in range(256)],
    dtype=np.uint8
)

def enhance_frame(bgr: np.ndarray) -> np.ndarray:
    """
    Adaptive low-light enhancement — applied to the detection copy only.
    The original frame is shown to the user so the display looks natural.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mean_luma = float(np.mean(l))
    if mean_luma < DARK_PIXEL_THRESHOLD:
        l = _clahe.apply(l)
        if mean_luma < DARK_PIXEL_THRESHOLD * 0.5:   # very dark → also gamma lift
            l = cv2.LUT(l, _GAMMA_LUT)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return bgr

# ─────────────────────────────────────────────
# COLOURS / FONT
# ─────────────────────────────────────────────
C_RED    = (45,  45,  225)
C_AMBER  = (30,  180, 230)
C_WHITE  = (240, 240, 240)
C_ACCENT = (0,   180, 120)
C_DIM    = (80,  80,  80)
C_LABEL  = (160, 160, 160)
FONT     = cv2.FONT_HERSHEY_DUPLEX

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
session_running     = True
calibration_samples = []
baseline_ear        = None
is_calibrated       = False
_wearing_glasses    = False
total_nods          = 0
total_yawns         = 0
total_distractions  = 0
total_drowsy        = 0
yawn_flag           = False
nod_head_down       = False
last_nod_time       = 0.0
f_eye = f_mouth = f_distract = 0
_ear_history        = []
_glasses_checks     = []   # late-calibration EAR samples for glasses vote

# ─────────────────────────────────────────────
# CORE MATH
# ─────────────────────────────────────────────
def gdist(a, b):
    return math.dist((a.x, a.y), (b.x, b.y))

def calculate_ear(lm):
    ear_l = (gdist(lm[385], lm[373]) + gdist(lm[387], lm[380])) / (2.0 * gdist(lm[362], lm[263]) or 1)
    ear_r = (gdist(lm[160], lm[144]) + gdist(lm[158], lm[153])) / (2.0 * gdist(lm[33],  lm[133]) or 1)
    raw   = (ear_l + ear_r) / 2.0
    return raw + (GLASSES_EAR_OFFSET if _wearing_glasses else 0.0)

def smooth_ear(raw: float) -> float:
    _ear_history.append(raw)
    if len(_ear_history) > EAR_SMOOTH_N:
        _ear_history.pop(0)
    return float(np.mean(_ear_history))

def calculate_mar(lm):
    hz = gdist(lm[61], lm[291])
    return gdist(lm[13], lm[14]) / hz if hz else 0.0

def estimate_head_pose(lm):
    fh    = abs(lm[152].y - lm[10].y) or 1e-6
    pitch = (lm[1].y - (lm[10].y + lm[152].y) / 2.0) / fh
    dl    = gdist(lm[1], lm[234])
    dr    = gdist(lm[1], lm[454])
    yaw   = (dl - dr) / ((dl + dr) or 1e-6)
    return float(pitch), float(yaw)

# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
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
    cv2.putText(img, "NIDRA",  (22,  44), FONT, 1.05, C_WHITE,  2, cv2.LINE_AA)
    cv2.putText(img, "ALERT", (148, 44), FONT, 1.05, C_ACCENT, 2, cv2.LINE_AA)
    timer = f"SESSION  {int(elapsed//60):02d}:{int(elapsed%60):02d}"
    tw = cv2.getTextSize(timer, FONT, 0.55, 1)[0][0]
    cv2.putText(img, timer, (w - tw - 18, 40), FONT, 0.55, C_LABEL, 1, cv2.LINE_AA)
    if _wearing_glasses:
        cv2.putText(img, "SPECS MODE", (w - tw - 120, 40),
                    FONT, 0.38, C_ACCENT, 1, cv2.LINE_AA)

def draw_calibration_bar(img, w, h, cal_rem):
    pct   = 1.0 - cal_rem / CALIBRATION_DURATION
    bar_w = int((w - 40) * pct)
    alpha_rect(img, 0, h - 38, w, h, (0, 0, 0), 0.55)
    cv2.rectangle(img, (20, h-24), (w-20, h-12), C_DIM, -1)
    if bar_w > 4:
        cv2.rectangle(img, (20, h-24), (20+bar_w, h-12), C_ACCENT, -1)
    label = f"CALIBRATING SENSORS — {int(cal_rem)}s remaining"
    cv2.putText(img, label, (22, h-30), FONT, 0.42, (0, 200, 255), 1, cv2.LINE_AA)

def draw_alert_banner(img, w, h, text, color):
    """
    Lean pill banner — compact, bottom-left, with a coloured left accent bar.
    Much smaller footprint than the original centred box.
    """
    scale = 0.52
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)

    pad_x, pad_y = 12, 7
    dot_gap = 18                        # space for the status dot
    bw = tw + pad_x * 2 + dot_gap
    bh = th + pad_y * 2

    bx = 16
    by = h - 76 - bh                    # sits just above the END SESSION button

    # semi-transparent dark fill
    alpha_rect(img, bx, by, bx + bw, by + bh, (8, 8, 18), 0.78)

    # thick left accent bar
    cv2.rectangle(img, (bx, by), (bx + 3, by + bh), color, -1)

    # thin outer border
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color, 1, cv2.LINE_AA)

    # pulsing dot
    dot_x = bx + 11
    dot_y = by + bh // 2
    cv2.circle(img, (dot_x, dot_y), 4, color, -1, cv2.LINE_AA)

    # alert text
    cv2.putText(img, text,
                (dot_x + 10, by + pad_y + th - 1),
                FONT, scale, C_WHITE, thick, cv2.LINE_AA)

def draw_end_button(img, w, h):
    bx1, by1, bx2, by2 = w-172, h-56, w-16, h-18
    alpha_rect(img, bx1, by1, bx2, by2, (20, 10, 40), 0.80)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), C_RED, 1, cv2.LINE_AA)
    label = "END SESSION"
    lw = cv2.getTextSize(label, FONT, 0.50, 1)[0][0]
    cv2.putText(img, label, (bx1+(bx2-bx1-lw)//2, by1+26), FONT, 0.50, C_WHITE, 1, cv2.LINE_AA)
    return (bx1, by1, bx2, by2)

# ─────────────────────────────────────────────
# MOUSE CALLBACK
# ─────────────────────────────────────────────
_btn_rect_cache = (0, 0, 1, 1)

def on_click(event, x, y, flags, param):
    global session_running
    if event == cv2.EVENT_LBUTTONDOWN:
        bx1, by1, bx2, by2 = _btn_rect_cache
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            session_running = False

# ─────────────────────────────────────────────
# MODEL DOWNLOAD
# ─────────────────────────────────────────────
MODEL_PATH = "face_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("Downloading face landmarker model (~30 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

# ─────────────────────────────────────────────
# MEDIAPIPE SETUP
# Lower confidence thresholds for better tracking in dim light / with glasses.
# ─────────────────────────────────────────────
_face_options = mp_vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.4,   # was 0.5
    min_face_presence_confidence=0.4,    # was 0.5
    min_tracking_confidence=0.4,         # was 0.5
)

# ─────────────────────────────────────────────
# CAMERA + WINDOW
# ─────────────────────────────────────────────
video = cv2.VideoCapture(0)
if not video.isOpened():
    raise RuntimeError("Cannot open webcam.")

# Hint camera toward auto-exposure (helps in dark environments).
# Not all cameras honour these — no harm if ignored.
video.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
video.set(cv2.CAP_PROP_GAIN, 0)

start_time = time.time()
WIN = "NIDRALERT"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WIN, on_click)

print("✓ Starting — keep face visible for 20s calibration.")
speak("NidrAlert started.")

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
cur_ear = cur_mar = cur_pitch = cur_yaw = 0.0

with mp_vision.FaceLandmarker.create_from_options(_face_options) as detector:

    while session_running:
        ret, frame = video.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        elapsed = time.time() - start_time
        ts_ms   = int(elapsed * 1000)

        # ── Low-light enhancement (detection copy only) ───────────────
        # MediaPipe receives 'enhanced'; user sees the natural 'frame'.
        enhanced = enhance_frame(frame)

        try:
            rgb     = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = detector.detect_for_video(mp_img, ts_ms)
        except Exception:
            results = None

        alert_text  = None
        alert_color = C_RED

        if results and results.face_landmarks:
            lm = results.face_landmarks[0]

            try:
                cur_ear            = smooth_ear(calculate_ear(lm))
                cur_mar            = calculate_mar(lm)
                cur_pitch, cur_yaw = estimate_head_pose(lm)
            except Exception:
                pass

            # ── Calibration ───────────────────────────────────────────
            if not is_calibrated:
                calibration_samples.append(cur_ear)

                # Collect late-calibration samples for glasses detection.
                # We wait until halfway so the EAR history is fully warm.
                if elapsed > CALIBRATION_DURATION * 0.5:
                    _glasses_checks.append(cur_ear)

                if elapsed >= CALIBRATION_DURATION:
                    baseline_ear  = float(np.mean(calibration_samples))
                    is_calibrated = True

                    # ── glasses decision ──────────────────────────────
                    # Glasses suppress EAR ~8-18% even with eyes open.
                    # If the late-cal mean is <93% of the full baseline,
                    # flag glasses and shift both the offset and baseline.
                    if _glasses_checks:
                        late_mean = float(np.mean(_glasses_checks))
                        if baseline_ear > 0 and (late_mean / baseline_ear) < 0.93:
                            _wearing_glasses = True
                            baseline_ear    += GLASSES_EAR_OFFSET
                            print("✓ Glasses detected — EAR offset applied.")

                    print(f"✓ Calibrated. Baseline EAR = {baseline_ear:.4f}  "
                          f"glasses={_wearing_glasses}")
                    speak("Calibration complete. Stay alert.")

            # ── Alert detection ───────────────────────────────────────
            else:
                dyn_thr = baseline_ear * DROWSY_PERCENTAGE

                # 1. DROWSINESS → voice + double low beep
                # Smile suppression: raised cheeks during a smile push up
                # the lower eyelid and drop EAR — same signal as closed eyes.
                # If MAR is elevated (smiling/talking) reset counter instead.
                smile_active = cur_mar > SMILE_MAR_SUPPRESS
                if cur_ear < dyn_thr and not smile_active:
                    f_eye += 1
                    if f_eye > CONSECUTIVE_FRAMES_EYE:
                        alert_text  = "CRITICAL: DROWSINESS"
                        alert_color = C_RED
                        if f_eye == CONSECUTIVE_FRAMES_EYE + 1:
                            total_drowsy += 1
                            speak("Wake up!")
                            beep_yawn()
                            save_alert("DROWSINESS", cur_ear)
                else:
                    f_eye = 0  # reset on open eyes OR active mouth

                # 2. YAWNING → double low beep
                if cur_mar > MAR_THRESHOLD:
                    f_mouth += 1
                    if f_mouth > CONSECUTIVE_FRAMES_MOUTH:
                        if alert_text is None:
                            alert_text  = "WARNING: YAWNING"
                            alert_color = C_AMBER
                        if not yawn_flag:
                            total_yawns += 1
                            yawn_flag    = True
                            beep_yawn()
                            save_alert("YAWNING", cur_mar)
                else:
                    f_mouth   = 0
                    yawn_flag = False

                # 3. DISTRACTION → double high beep
                if abs(cur_yaw) > YAW_DISTRACTION_THR:
                    f_distract += 1
                    if f_distract > DISTRACTION_FRAMES:
                        if alert_text is None:
                            alert_text  = "ALERT: FOCUS ON ROAD"
                            alert_color = C_AMBER
                        if f_distract == DISTRACTION_FRAMES + 1:
                            total_distractions += 1
                            beep_distract()
                            save_alert("DISTRACTION", cur_yaw)
                else:
                    f_distract = 0

                # 4. HEAD DROP → voice + double low beep (always overrides banner)
                now = time.time()
                if not nod_head_down:
                    if cur_pitch < NOD_PITCH_THRESHOLD:
                        nod_head_down = True
                else:
                    if cur_pitch > NOD_RECOVER_THRESHOLD:
                        nod_head_down = False
                        if now - last_nod_time > NOD_DEBOUNCE_S:
                            total_nods    += 1
                            last_nod_time  = now
                            alert_text     = "ALERT: HEAD DROP"
                            alert_color    = C_RED
                            speak("Wake up! Head drop detected.")
                            beep_yawn()
                            save_alert("HEAD_DROP", cur_pitch)

        else:
            # Face lost — reset frame counters
            f_eye      = 0
            f_mouth    = 0
            f_distract = 0
            yawn_flag  = False

        # ── Push live data to FastAPI → React dashboard ──────────────
        push_frame_to_backend(
            ear=cur_ear,
            mar=cur_mar,
            pitch=cur_pitch,
            yaw=cur_yaw,
            alert_text=alert_text,
            calibrated=is_calibrated,
        )

        # ── Draw UI on original (unenhanced) frame ────────────────────
        draw_header(img=frame, w=w, elapsed=elapsed)
        if not is_calibrated:
            draw_calibration_bar(frame, w, h, max(0.0, CALIBRATION_DURATION - elapsed))
        if alert_text:
            draw_alert_banner(frame, w, h, alert_text, alert_color)
        btn = draw_end_button(frame, w, h)
        _btn_rect_cache = btn

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

# ─────────────────────────────────────────────
# SESSION SUMMARY
# ─────────────────────────────────────────────
duration     = time.time() - start_time
total_alerts = total_drowsy + total_yawns + total_distractions + total_nods

summary = {
    "timestamp":        datetime.utcnow().isoformat() + "Z",
    "duration_seconds": round(duration, 1),
    "duration_fmt":     f"{int(duration//60)}m {int(duration%60)}s",
    "baseline_ear":     round(baseline_ear, 4) if baseline_ear else None,
    "glasses_detected": _wearing_glasses,
    "drowsy_events":    total_drowsy,
    "head_drops":       total_nods,
    "yawn_count":       total_yawns,
    "distraction_acts": total_distractions,
    "total_alerts":     total_alerts,
    "calibrated":       is_calibrated,
}

print(f"\n{'='*40}")
print(f"  NIDRALERT SESSION SUMMARY")
print(f"{'='*40}")
for k, v in summary.items():
    print(f"  {k:<22}: {v}")
print(f"{'='*40}\n")

if total_alerts > 0:
    save_session(summary)   # saves to MongoDB + POSTs to /api/ai/session → React history
else:
    print("  No alerts fired — session summary not saved.")

_mixer.quit()
video.release()
cv2.destroyAllWindows()