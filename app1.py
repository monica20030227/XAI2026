import requests
import ast
import hashlib
import json
import os
import uuid
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import cv2
except Exception:
    cv2 = None

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

try:
    from deepface import DeepFace
    DEEPFACE_IMPORT_ERROR = ""
except Exception as e:
    DeepFace = None
    DEEPFACE_IMPORT_ERROR = repr(e)

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
except Exception:
    webrtc_streamer = None
    WebRtcMode = None
    VideoProcessorBase = object

try:
    import av
except Exception:
    av = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SNAP_DIR = BASE_DIR / "emotion_snapshots"
TASKS_FILE = BASE_DIR / "tasks.json"
RESPONSES_CSV = DATA_DIR / "xai_responses.csv"
EVENTS_CSV = DATA_DIR / "xai_events.csv"
EMOTION_CSV = DATA_DIR / "xai_emotion_records.csv"
DATA_DIR.mkdir(exist_ok=True)
SNAP_DIR.mkdir(exist_ok=True)

ADMIN_CODE = os.environ.get("XAI_ADMIN_CODE", "teacher-demo")
APPS_SCRIPT_URL = os.environ.get("APPS_SCRIPT_URL", "https://script.google.com/macros/s/AKfycbycczwACCLax8hs5Qej17F-yP7vqdXrdA-nFRYmdS8vAw7RpXayrAD6u49qKW2H1J2U/exec")
APPS_SCRIPT_TOKEN = os.environ.get("APPS_SCRIPT_TOKEN", "xai-2026")

EMOTION_RECORD_COLUMNS = [
    "timestamp",
    "session_id",
    "student_id",
    "task_id",
    "task_order",
    "condition_id",
    "raw_emotion",
    "stable_emotion",
    "confidence",
    "engagement_score",
]

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def append_csv(path: Path, row: dict):
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(path, mode="w", header=True, index=False, encoding="utf-8-sig")

def sync_to_apps_script(sheet_name: str, row: dict):
    """
    將單筆資料同步到 Google Apps Script。
    若同步失敗，不中斷本機 CSV 儲存，避免影響學生實驗流程。
    """
    if not APPS_SCRIPT_URL:
        return False

    payload = {
        "token": APPS_SCRIPT_TOKEN,
        "sheet": sheet_name,
        "data": row,
    }

    try:
        response = requests.post(
            APPS_SCRIPT_URL,
            json=payload,
            timeout=5,
        )

        if response.status_code == 200:
            result = response.json()
            return bool(result.get("ok"))

        return False

    except Exception as e:
        # 不要讓同步失敗影響平台作答
        print(f"[Apps Script sync failed] {sheet_name}: {e}")
        return False

def log_event(event_type: str, **payload):
    row = {
        "timestamp": utc_now_iso(),
        "session_id": st.session_state.get("session_id", ""),
        "student_id": st.session_state.get("student_id", ""),
        "week": st.session_state.get("week", ""),
        "condition_id": st.session_state.get("condition_id", ""),
        "phase": st.session_state.get("phase", ""),
        "event_type": event_type,
        "payload": json.dumps(payload, ensure_ascii=False),
    }

    append_csv(EVENTS_CSV, row)
    sync_to_apps_script("events", row)


def save_response(row: dict):
    append_csv(RESPONSES_CSV, row)
    sync_to_apps_script("responses", row)



def save_emotion_record(row: dict):
    normalized = {col: row.get(col, "") for col in EMOTION_RECORD_COLUMNS}
    df_new = pd.DataFrame([normalized], columns=EMOTION_RECORD_COLUMNS)

    if EMOTION_CSV.exists():
        try:
            df_old = pd.read_csv(EMOTION_CSV, on_bad_lines="skip")
        except Exception:
            df_old = pd.DataFrame(columns=EMOTION_RECORD_COLUMNS)

        for col in EMOTION_RECORD_COLUMNS:
            if col not in df_old.columns:
                df_old[col] = ""

        df_old = df_old[EMOTION_RECORD_COLUMNS]
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all.to_csv(EMOTION_CSV, index=False, encoding="utf-8-sig")
    else:
        df_new.to_csv(EMOTION_CSV, index=False, encoding="utf-8-sig")
    sync_to_apps_script("emotion", normalized)

def load_config():
    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def get_week_cfg(config):
    return config["weeks"][str(st.session_state.get("week", "7"))]

def assign_condition(student_id: str, conditions):
    digest = hashlib.md5(student_id.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(conditions)
    return conditions[idx]


def get_current_condition(config):
    week_cfg = get_week_cfg(config)
    conditions = week_cfg.get("conditions", [])
    current_id = st.session_state.get("condition_id", "")

    condition = next((c for c in conditions if c.get("id") == current_id), None)

    if condition is None:
        student_id = st.session_state.get("student_id", "").strip()

        if student_id and conditions:
            condition = assign_condition(student_id, conditions)
        elif conditions:
            condition = conditions[0]
        else:
            raise ValueError(f"week {st.session_state.get('week')} 沒有可用的 conditions 設定")

        st.session_state["condition_id"] = condition["id"]
        st.session_state["condition_name"] = condition["name"]

    return condition


def ensure_student_state(config):
    role = st.session_state.get("role", "student")

    if role == "admin":
        return True

    student_id = st.session_state.get("student_id", "").strip()
    if not student_id:
        st.session_state["phase"] = "login"
        return False

    # 本版刪除「每位學生固定一個實驗條件」的顯示與派派。
    # 條件 A/B/C 會依每題任務的平衡順序動態設定。
    return True

def init_session_defaults():
    defaults = {
        "session_id": str(uuid.uuid4()),
        "student_id": "",
        "role": "student",
        "phase": "login",
        "week": "7",
        "condition_id": "",
        "condition_name": "",
        "student_code": "",
        "task_started_at": None,
        "verify_attempts": 0,
        "hint_requests": 0,
        "proactive_hint_count": 0,
        "verified": False,
        "last_result": "",
        "final_saved": False,
        "pretest_score": None,
        "posttest_score": None,
        "baseline_confidence": 3,
        "baseline_emotion_self": "中性/平穩",
        "task_emotion_self": "中性/平穩",
        "post_emotion_self": "中性/平穩",
        "prior_xai": "是",
        "chosen_explanation_faithful": "",
        "chosen_explanation_actionable": "",
        "emotion_detecting": False,
        "emotion_recent_window": [],
        "latest_raw_emotion": "",
        "latest_stable_emotion": "",
        "latest_confidence": 0.0,
        "latest_engagement_score": 0.0,
        "emotion_uploaded_video_path": "",
        "last_emotion_capture_at": 0.0,
        "webrtc_enabled": False,
        "current_task_index": 0,
        "task_sequence": [],
        "task_records": [],
        "overall_preference_understand": "",
        "overall_preference_trust": "",
        "overall_preference_next_step": "",
        "overall_reflection": "",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def inject_custom_css():
    st.markdown("""
    <style>
    .stApp {background: #f7f8fb;}
    .main .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1320px;}
    [data-testid="stSidebar"] {background: #ffffff; border-right: 1px solid #e6e8ef;}
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1f3b75 55%, #2855b8 100%);
        color: white; padding: 1rem 1.2rem; border-radius: 18px; margin-bottom: 1rem;
        box-shadow: 0 10px 25px rgba(15,23,42,0.12);
    }
    .hero h2 {margin: 0 0 0.25rem 0; font-size: 1.55rem;}
    .hero p {margin: 0; opacity: 0.95; font-size: 0.98rem;}
    .soft-card {
        background: white; border: 1px solid #e8ebf3; border-radius: 16px;
        padding: 0.95rem 1rem; margin-bottom: 0.75rem;
        box-shadow: 0 6px 18px rgba(15,23,42,0.05);
    }
    .soft-card h4 {margin: 0 0 0.45rem 0; color:#0f172a;}
    .soft-kpi {
        background: #eef4ff; border: 1px solid #dbe7ff; border-radius: 14px;
        padding: 0.8rem 0.9rem; margin-bottom: 0.6rem;
    }
    .muted {color:#475569; font-size:0.95rem;}
    .section-label{
        display:inline-block; padding:0.25rem 0.65rem; border-radius:999px;
        background:#e8f0ff; color:#214caa; font-weight:600; font-size:0.82rem; margin-bottom:0.45rem;
    }
    .step-chip{
        display:inline-block; padding:0.18rem 0.55rem; border-radius:999px;
        background:#f1f5f9; color:#334155; font-size:0.8rem; margin:0 0.3rem 0.35rem 0;
        border:1px solid #e2e8f0;
    }
    .note-box{
        background:#fff9e8; border:1px solid #f6e6a8; color:#6b5300;
        border-radius:14px; padding:0.8rem 0.9rem; margin-bottom:0.8rem;
    }
    div[data-testid="stCodeBlock"] {border-radius: 14px; overflow:hidden;}
    textarea {font-family: Consolas, 'Courier New', monospace !important;}
    </style>
    """, unsafe_allow_html=True)

def render_header(config):
    st.set_page_config(page_title=config["course_title"], page_icon="🧠", layout="wide")
    inject_custom_css()
    st.markdown(f"""<div class="hero"><h2>🧠 {config["course_title"]}</h2><p>{config["course_caption"]}</p></div>""", unsafe_allow_html=True)


class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.latest_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_frame = img.copy()
        return frame

def save_uploaded_image(upload, prefix: str) -> str:
    if upload is None:
        return ""
    ext = "jpg"
    name = f"{prefix}_{st.session_state.get('session_id','')}.{ext}"
    path = SNAP_DIR / name
    with open(path, "wb") as f:
        f.write(upload.getbuffer())
    return str(path)


def save_uploaded_video(upload, prefix: str) -> str:
    if upload is None:
        return ""
    suffix = Path(upload.name).suffix or ".mp4"
    name = f"{prefix}_{st.session_state.get('session_id','')}{suffix}"
    path = SNAP_DIR / name
    with open(path, "wb") as f:
        f.write(upload.getbuffer())
    return str(path)

def emotion_to_engagement_score(emotion: str) -> float:
    mapping = {
        "happy": 0.80,
        "neutral": 0.70,
        "surprise": 0.50,
        "sad": 0.30,
        "angry": 0.20,
        "fear": 0.20,
        "disgust": 0.10,
        "unknown": 0.50,
        "-": 0.50,
    }
    return float(mapping.get(str(emotion).lower(), 0.50))

def build_stable_emotion(raw_emotion: str, window_size: int = 5) -> str:
    recent = list(st.session_state.get("emotion_recent_window", []))
    recent.append(raw_emotion or "unknown")
    recent = recent[-window_size:]
    st.session_state["emotion_recent_window"] = recent
    counter = Counter(recent)
    return counter.most_common(1)[0][0] if counter else "unknown"

def append_task_emotion_row(raw_emotion: str, stable_emotion: str, confidence: float):
    row = {
        "timestamp": utc_now_iso(),
        "session_id": st.session_state.get("session_id", ""),
        "student_id": st.session_state.get("student_id", ""),
        "task_id": st.session_state.get("current_task_id", ""),
        "task_order": st.session_state.get("current_task_index", ""),
        "condition_id": st.session_state.get("condition_id", ""),
        "raw_emotion": raw_emotion,
        "stable_emotion": stable_emotion,
        "confidence": float(confidence),
        "engagement_score": float(emotion_to_engagement_score(stable_emotion)),
    }
    st.session_state["latest_raw_emotion"] = row["raw_emotion"]
    st.session_state["latest_stable_emotion"] = row["stable_emotion"]
    st.session_state["latest_confidence"] = row["confidence"]
    st.session_state["latest_engagement_score"] = row["engagement_score"]
    save_emotion_record(row)
    return row


def find_available_camera_index(max_index: int = 3):
    if cv2 is None:
        return None
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, _ = cap.read()
            cap.release()
            if ok:
                return idx
        else:
            cap.release()
    return None

def analyze_frame_emotion(frame):
    if cv2 is None:
        raise RuntimeError("OpenCV 未安裝，無法進行情緒偵測。")
    if DeepFace is None:
        raise RuntimeError("DeepFace 未安裝，無法進行情緒偵測。")

    result = DeepFace.analyze(
        img_path=frame,
        actions=["emotion"],
        enforce_detection=False,
        detector_backend="opencv",
        silent=True,
    )
    if isinstance(result, list):
        result = result[0]

    emotion_scores = result.get("emotion", {}) or {}
    raw_emotion = result.get("dominant_emotion", "unknown") or "unknown"
    confidence = float(emotion_scores.get(raw_emotion, 0.0))
    stable_emotion = build_stable_emotion(raw_emotion)
    return append_task_emotion_row(raw_emotion, stable_emotion, confidence)

def process_uploaded_video(video_path: str, sample_every_seconds: int = 3, max_duration_seconds: int = 120):
    if cv2 is None:
        raise RuntimeError("OpenCV 未安裝，無法分析影片。")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("無法開啟上傳影片。")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if frame_count is None or frame_count <= 0:
        duration_seconds = max_duration_seconds
    else:
        duration_seconds = frame_count / fps

    analysis_duration = min(float(duration_seconds), float(max_duration_seconds))
    sample_points = list(range(0, int(analysis_duration), int(sample_every_seconds)))

    rows = []

    try:
        for sec in sample_points:
            target_msec = float(sec) * 1000.0
            cap.set(cv2.CAP_PROP_POS_MSEC, target_msec)
            ok, frame = cap.read()

            if not ok:
                target_frame = int(sec * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ok, frame = cap.read()

            if ok:
                try:
                    row = analyze_frame_emotion(frame)
                except Exception:
                    row = append_task_emotion_row("unknown", build_stable_emotion("unknown"), 0.0)
            else:
                row = append_task_emotion_row("unknown", build_stable_emotion("unknown"), 0.0)

            rows.append(row)
    finally:
        cap.release()

    return rows

def get_recent_emotion_records(limit: int = 20) -> pd.DataFrame:
    if not EMOTION_CSV.exists():
        return pd.DataFrame(columns=EMOTION_RECORD_COLUMNS)
    try:
        df = pd.read_csv(EMOTION_CSV, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame(columns=EMOTION_RECORD_COLUMNS)

    for c in EMOTION_RECORD_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    df = df[EMOTION_RECORD_COLUMNS]
    sid = str(st.session_state.get("session_id", "") or "")
    if sid:
        df = df[df["session_id"].astype(str) == sid]
    return df.tail(limit)

@st.fragment(run_every="3s")
def render_emotion_detection_live_panel(webrtc_ctx):
    info_cols = st.columns(4)
    info_cols[0].metric("raw_emotion", st.session_state.get("latest_raw_emotion", "-") or "-")
    info_cols[1].metric("stable_emotion", st.session_state.get("latest_stable_emotion", "-") or "-")
    info_cols[2].metric("confidence", f"{float(st.session_state.get('latest_confidence', 0.0)):.2f}")
    info_cols[3].metric("engagement", f"{float(st.session_state.get('latest_engagement_score', 0.0)):.2f}")

    if st.session_state.get("emotion_detecting", False):
        if DeepFace is None:
            st.error(f"目前環境缺少 DeepFace，無法開始情緒偵測。錯誤原因：{DEEPFACE_IMPORT_ERROR}")
        elif webrtc_ctx is None:
            st.error("webcam 模組不可用。請確認已安裝 streamlit-webrtc 與 av。")
        elif not getattr(webrtc_ctx.state, "playing", False):
            st.info("請先允許瀏覽器使用內建相機，啟動 webcam 後才會開始記錄。")
        else:
            processor = getattr(webrtc_ctx, "video_processor", None)
            frame = getattr(processor, "latest_frame", None) if processor else None
            if frame is not None:
                now_ts = time.time()
                last_ts = float(st.session_state.get("last_emotion_capture_at", 0.0) or 0.0)
                if now_ts - last_ts >= 3:
                    try:
                        row = analyze_frame_emotion(frame)
                        st.session_state["last_emotion_capture_at"] = now_ts
                        log_event(
                            "emotion_detected",
                            raw_emotion=row["raw_emotion"],
                            stable_emotion=row["stable_emotion"],
                            confidence=row["confidence"],
                            engagement_score=row["engagement_score"],
                        )
                        st.success("已記錄 1 筆任務操作情緒。")
                    except Exception as e:
                        st.error(f"情緒偵測失敗：{e}")
            else:
                st.info("正在等待 webcam 畫面...")

    st.markdown("#### 即時情緒紀錄表")
    st.dataframe(get_recent_emotion_records(20), use_container_width=True)

def render_task_video_emotion_monitor():
    st.markdown("### 任務操作情緒記錄")
    st.caption("請開啟 webcam 進行拍攝與偵測；按開始偵測後進入任務區。")

    btn1, btn2 = st.columns(2)
    with btn1:
        if st.button("開始偵測", key="start_task_emotion_detection"):
            st.session_state["emotion_detecting"] = True
            st.session_state["last_emotion_capture_at"] = 0.0
            log_event("emotion_detection_started", source="webcam")
            st.rerun()
    with btn2:
        if st.button("停止偵測", key="stop_task_emotion_detection"):
            st.session_state["emotion_detecting"] = False
            log_event("emotion_detection_stopped", source="webcam")
            st.rerun()

    st.markdown("#### webcam 拍攝框")
    webrtc_ctx = None
    if webrtc_streamer is not None and WebRtcMode is not None and av is not None:
        webrtc_ctx = webrtc_streamer(
            key=f"task_emotion_webrtc_{st.session_state.get('week')}",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]}
                ]
            },
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
            video_processor_factory=EmotionVideoProcessor,
            async_processing=True,
        )
    else:
        st.warning("目前環境未安裝 streamlit-webrtc / av，無法直接使用瀏覽器內建 webcam。可改用下方影片上傳。")

    uploaded_video = st.file_uploader(
        "若無攝影機，可上傳 任務操作情緒記錄 影片",
        type=["mp4", "mov", "avi", "mkv", "mpeg"],
        key="task_video_uploader",
    )
    if uploaded_video is not None:
        video_path = save_uploaded_video(uploaded_video, f"task_video_w{st.session_state.get('week')}")
        st.session_state["emotion_uploaded_video_path"] = video_path
        st.video(video_path)

        if st.button("分析上傳影片並保存紀錄", key="analyze_uploaded_task_video"):
            try:
                rows = process_uploaded_video(video_path)
                log_event("emotion_video_processed", source="uploaded_video", n_rows=len(rows))
                st.success(f"影片分析完成，已保存 {len(rows)} 筆情緒紀錄。")
                st.rerun()
            except Exception as e:
                st.error(f"影片分析失敗：{e}")

    render_emotion_detection_live_panel(webrtc_ctx)

def render_emotion_capture(title: str, store_prefix: str, week_cfg: dict):
    st.markdown(f"### {title}")
    options = week_cfg["emotion_module"]["self_report_options"]
    current = st.session_state.get(f"{store_prefix}_emotion_self", options[1])
    idx = options.index(current) if current in options else 1
    st.session_state[f"{store_prefix}_emotion_self"] = st.selectbox(
        "主觀情緒自評", options, index=idx, key=f"{store_prefix}_emotion_selectbox"
    )
    upload = st.camera_input(f"可選：拍攝 {title} 快照", key=f"{store_prefix}_cam")
    if upload is None:
        upload = st.file_uploader(f"若無攝影機，可上傳 {title} 照片", type=["jpg","jpeg","png"], key=f"{store_prefix}_file")
    if st.button(f"保存 {title} 紀錄", key=f"save_{store_prefix}"):
        img_path = save_uploaded_image(upload, f"{store_prefix}_w{st.session_state.get('week')}")
        st.session_state[f"{store_prefix}_snapshot_path"] = img_path
        save_emotion_record(
            {
                "timestamp": utc_now_iso(),
                "session_id": st.session_state.get("session_id",""),
                "student_id": st.session_state.get("student_id",""),
                "week": st.session_state.get("week",""),
                "phase_label": store_prefix,
                "snapshot_path": img_path,
                "self_report": st.session_state[f"{store_prefix}_emotion_self"],
                "backend": "self_report_only",
            }
        )
        log_event("emotion_saved", phase_label=store_prefix, self_report=st.session_state[f"{store_prefix}_emotion_self"])
        st.success("情緒紀錄已保存。")

def grade_mcq(items, prefix: str):
    score = 0
    answers = {}
    for i, item in enumerate(items, start=1):
        key = f"{prefix}_q{i}"
        choice = st.radio(f"Q{i}. {item['question']}", item["options"], key=key)
        answers[key] = choice
        if choice == item["answer"]:
            score += 1
    return score, answers

def reset_task_counters():
    for k in ["verify_attempts", "hint_requests", "proactive_hint_count"]:
        st.session_state[k] = 0
    st.session_state["verified"] = False
    st.session_state["last_result"] = ""
    st.session_state["final_saved"] = False


def get_task_sequence(config):
    """Return balanced task-condition sequence for the current student.

    G1: T1+A, T2+B, T3+C
    G2: T1+B, T2+C, T3+A
    G3: T1+C, T2+A, T3+B
    """
    week_cfg = get_week_cfg(config)
    tasks = week_cfg.get("tasks", [])
    if len(tasks) < 3:
        raise ValueError("tasks.json 需要至少 3 個 tasks。")

    student_id = st.session_state.get("student_id", "") or "demo"
    group_idx = int(hashlib.md5(student_id.encode("utf-8")).hexdigest(), 16) % 3
    condition_orders = [
        ["A", "B", "C"],
        ["B", "C", "A"],
        ["C", "A", "B"],
    ]
    cond_order = condition_orders[group_idx]
    return [
        {"task_index": i, "task_id": tasks[i]["id"], "condition_id": cond_order[i]}
        for i in range(3)
    ]


def get_current_task_and_condition(config):
    week_cfg = get_week_cfg(config)
    tasks = week_cfg.get("tasks", [])
    conditions = {c["id"]: c for c in week_cfg.get("conditions", [])}
    seq = st.session_state.get("task_sequence") or get_task_sequence(config)
    st.session_state["task_sequence"] = seq
    idx = int(st.session_state.get("current_task_index", 0))
    idx = min(max(idx, 0), len(seq) - 1)
    seq_item = seq[idx]
    task = tasks[seq_item["task_index"]]
    condition = conditions[seq_item["condition_id"]]
    st.session_state["current_task_id"] = task["id"]
    st.session_state["condition_id"] = condition["id"]
    st.session_state["condition_name"] = condition["name"]
    return task, condition, idx, seq


def grade_task_answer(task, agree_choice, emotion_choice, next_step_choice):
    score = 0
    if agree_choice == task["answers"]["agree"]:
        score += 1
    if emotion_choice == task["answers"]["emotion"]:
        score += 1
    if next_step_choice == task["answers"]["next_step"]:
        score += 1
    return score

def render_login(config):
    st.subheader("登入")
    c1, c2 = st.columns([2, 1])
    with c1:
        student_id = st.text_input("請輸入學號 / student_id", value=st.session_state.get("student_id", ""))
    with c2:
        role = st.selectbox("身份", ["student", "admin"], index=0 if st.session_state.get("role") == "student" else 1)

    admin_code = ""
    if role == "admin":
        admin_code = st.text_input("管理者代碼", type="password")

    if st.button("進入平台", type="primary"):
        if not student_id.strip():
            st.error("請先輸入學號。")
            return
        if role == "admin" and admin_code != ADMIN_CODE:
            st.error("管理者代碼錯誤。")
            return

        st.session_state["student_id"] = student_id.strip()
        st.session_state["role"] = role
        st.session_state["week"] = "7"
        st.session_state["task_sequence"] = get_task_sequence(config) if role == "student" else []
        st.session_state["current_task_index"] = 0
        st.session_state["task_records"] = []
        reset_task_counters()

        if role == "student":
            st.session_state["phase"] = "intro"
        else:
            st.session_state["phase"] = "admin"
        log_event("login", role=role, selected_week="7")
        st.rerun()

def render_admin():
    st.success("管理端")
    if RESPONSES_CSV.exists():
        df = pd.read_csv(RESPONSES_CSV)
        st.metric("回應筆數", len(df))
        st.dataframe(df.tail(20), use_container_width=True)
        st.download_button("下載 responses.csv", RESPONSES_CSV.read_bytes(), "responses.csv", "text/csv")
    if EVENTS_CSV.exists():
        df = pd.read_csv(EVENTS_CSV)
        st.metric("事件筆數", len(df))
        st.dataframe(df.tail(20), use_container_width=True)
        st.download_button("下載 events.csv", EVENTS_CSV.read_bytes(), "events.csv", "text/csv")
    if EMOTION_CSV.exists():
        df = pd.read_csv(EMOTION_CSV, on_bad_lines="skip")
        for c in EMOTION_RECORD_COLUMNS:
            if c not in df.columns:
                df[c] = ""
        df = df[EMOTION_RECORD_COLUMNS]
        st.metric("情緒紀錄筆數", len(df))
        st.dataframe(df.tail(20), use_container_width=True)
        st.download_button("下載 emotion_records.csv", df.to_csv(index=False, encoding="utf-8-sig"), "emotion_records.csv", "text/csv")
    if st.button("測試同步到 Google Sheet"):
        test_row = {
        "timestamp": utc_now_iso(),
        "student_id": "admin_test",
        "session_id": st.session_state.get("session_id", ""),
        "message": "這是 Streamlit 管理端同步測試",
    }

    ok = sync_to_apps_script("responses", test_row)

    if ok:
        st.success("同步成功，請到 Google Sheet 的 responses 工作表確認。")
    else:
        st.error("同步失敗，請檢查 Apps Script URL、Token 或部署權限。")

def render_intro(config):
    week_cfg = get_week_cfg(config)

    st.markdown(f'<div class="section-label">{week_cfg["week_label"]}</div>', unsafe_allow_html=True)
    st.markdown('<div class="soft-card"><h4>AI 情緒判斷案例題</h4><div class="muted">本次任務將以三個等值案例，比較 A/B/C 三種 XAI 解釋呈現方式。每位學生都會完成三題，但題目與解釋條件採平衡順序，不固定 T1-A、T2-B、T3-C。</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="note-box">{week_cfg["intro_banner"]}</div>', unsafe_allow_html=True)

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("### 任務目標")
        for item in week_cfg["intro_items"]:
            st.write(f"- {item}")
        st.markdown("### XAI 觀察重點")
        for item in week_cfg["xai_focus"]:
            st.write(f"- {item}")
    with right:
        with st.container(border=True):
            st.markdown("### 本次任務安排")
            st.write("- T1、T2、T3 均為 AI 情緒判斷案例題。")
            st.write("- 任務操作維持三區塊：任務情境、XAI 解釋、學生作答。")
            st.write("- 任務區作答期間可連續紀錄 webcam 情緒。")
            st.write("- A/B/C 解釋條件依平衡順序出現，不顯示為個人固定實驗條件。")

    st.markdown("### 任務前短量表（基線）")
    st.session_state["baseline_confidence"] = st.slider("我現在對完成情緒判斷任務的信心", 1, 5, st.session_state.get("baseline_confidence", 3))
    st.session_state["prior_xai"] = st.radio("你是否修習過或聽過 xAI / AI 解釋相關概念？", ["是", "否"], horizontal=True, index=0 if st.session_state.get("prior_xai") == "是" else 1)

    st.markdown("### 前測（Pre-test）")
    pre_score, pre_answers = grade_mcq(week_cfg["pretest_items"], "pretest_xai_emotion")
    confirm = st.checkbox("我了解本任務會閱讀 AI 情緒判斷案例，並完成三個任務與短量表。")
    if st.button("開始正式任務", type="primary"):
        if not confirm:
            st.error("請先勾選確認。")
            return
        st.session_state["pretest_score"] = pre_score
        st.session_state["pretest_answers"] = pre_answers
        st.session_state["phase"] = "task"
        st.session_state["current_task_index"] = 0
        st.session_state["task_sequence"] = get_task_sequence(config)
        st.session_state["task_records"] = []
        st.session_state["task_started_at"] = utc_now_iso()
        reset_task_counters()
        log_event("task_sequence_started", pretest_score=pre_score, sequence=st.session_state["task_sequence"])
        st.rerun()

def verify_code(task):
    st.session_state["verify_attempts"] += 1
    try:
        safe_globals = {}
        safe_locals = {}
        exec(st.session_state["student_code"], safe_globals, safe_locals)
        fn = safe_locals.get(task["function_name"]) or safe_globals.get(task["function_name"])
        if not callable(fn):
            raise ValueError(f"找不到 {task['function_name']} 函式。")
        result = fn(ast.literal_eval(task["test_input"]))
        ok = str(result) == str(task["expected_output"])
        st.session_state["last_result"] = str(result)
        st.session_state["verified"] = ok
        if ok:
            st.success(f"驗證成功，輸出為 {result}。你已完成本題操作，請繼續完成量表與後測。")
        else:
            st.error(f"目前輸出為 {result}，尚未達到預期 {task['expected_output']}。目前未通過驗證。請回頭檢關鍵條件是否與題目要求一致，再重新驗證。")
        log_event("verify", task_id=task["id"], ok=ok, result=str(result))
    except Exception as e:
        st.session_state["last_result"] = f"ERROR: {e}"
        st.session_state["verified"] = False
        st.error(f"執行失敗：{e}")
        log_event("verify", task_id=task["id"], ok=False, result=str(e))

def render_explanation_panel(task, condition):
    condition_id = condition["id"]
    explanation = task["explanations"][condition_id]
    st.markdown(f"#### XAI 解釋｜{condition['name']}")
    st.caption(condition.get("student_facing_desc", ""))

    with st.container(border=True):
        st.markdown("**AI 解釋內容**")
        st.write(explanation["text"])

    if condition_id == "C":
        st.markdown("#### 分層 Hint")
        hints = explanation.get("hints", [])
        for i, hint in enumerate(hints, start=1):
            if st.button(f"Show Hint {i}", key=f"hint_{task['id']}_{i}_{st.session_state.get('current_task_index')}"):
                st.session_state["hint_requests"] += 1
                log_event("hint_requested", task_id=task["id"], hint_level=i, condition_id=condition_id)
            if st.session_state.get("hint_requests", 0) >= i:
                with st.container(border=True):
                    st.markdown(f"**Hint {i}**")
                    st.write(hint)
    else:
        st.info("本條件不提供額外 Hint，請依目前 AI 解釋完成判斷。")

def render_task(config):
    week_cfg = get_week_cfg(config)
    task, condition, idx, seq = get_current_task_and_condition(config)

    st.markdown(f'<div class="section-label">{week_cfg["week_label"]}｜任務 {idx + 1} / {len(seq)}｜{task["title"]}</div>', unsafe_allow_html=True)
    st.markdown('<div class="soft-card"><h4>任務操作區</h4><div class="muted">請開啟情緒偵測後再作答。作答期間系統會每 3 秒嘗試記錄一次情緒。任務操作維持三區塊：任務情境、XAI 解釋、學生作答。</div></div>', unsafe_allow_html=True)

    render_task_video_emotion_monitor()

    if not st.session_state.get("emotion_detecting", False):
        st.warning("請先按上方「開始偵測」，再進行任務作答；若使用上傳影片，請先完成影片分析。")

    left, center, right = st.columns([1.05, 1.25, 1.1])
    with left:
        with st.container(border=True):
            st.markdown("#### 任務情境")
            st.write(task["case_description"])
            st.markdown("**學習反思文字**")
            st.info(task["reflection_text"])
            st.markdown("**AI 判斷**")
            st.write(task["ai_judgment"])
            st.markdown("**成功條件**")
            st.write(task["success_condition"])
            st.markdown("**流程提醒**")
            for item in task["steps"]:
                st.write(f"- {item}")

    with center:
        render_explanation_panel(task, condition)

    with right:
        with st.container(border=True):
            st.markdown("#### 學生作答")
            agree_choice = st.radio(
                "Q1. 你是否同意 AI 的情緒判斷？",
                task["answer_options"]["agree"],
                key=f"agree_{task['id']}_{condition['id']}"
            )
            emotion_choice = st.radio(
                "Q2. 你認為這段文字最主要的情緒是什麼？",
                task["answer_options"]["emotion"],
                key=f"emotion_{task['id']}_{condition['id']}"
            )
            next_step_choice = st.radio(
                "Q3. 你認為最適合的下一步協助方式是什麼？",
                task["answer_options"]["next_step"],
                key=f"next_step_{task['id']}_{condition['id']}"
            )

            st.markdown("#### 任務後短量表")
            scale_keys = {
                "explanation_clarity": "我能理解 AI 為什麼做出這個情緒判斷。",
                "key_factor_understanding": "AI 解釋讓我知道哪些語句是判斷依據。",
                "next_step_known": "AI 解釋讓我知道下一步應如何協助學生。",
                "trust_in_explanation": "我信任 AI 對此案例的情緒判斷。",
                "uncertainty_reduction": "AI 解釋降低了我判斷案例時的不確定感。",
                "explanation_useful": "這個解釋對我完成判斷任務有幫助。",
            }
            scale_values = {}
            for k, label in scale_keys.items():
                scale_values[k] = st.slider(label, 1, 5, 3, key=f"scale_{k}_{task['id']}_{condition['id']}")

            if st.button("提交本題並進入下一步", type="primary", key=f"submit_task_{task['id']}_{condition['id']}"):
                if not st.session_state.get("emotion_detecting", False) and not st.session_state.get("emotion_uploaded_video_path"):
                    st.error("請先開始情緒偵測，或上傳並分析任務操作影片，才能提交本題。")
                    return
                score = grade_task_answer(task, agree_choice, emotion_choice, next_step_choice)
                record = {
                    "task_order": idx + 1,
                    "task_id": task["id"],
                    "task_title": task["title"],
                    "condition_id": condition["id"],
                    "condition_name": condition["name"],
                    "ai_emotion": task["target_emotion"],
                    "student_agree": agree_choice,
                    "student_emotion_answer": emotion_choice,
                    "next_step_answer": next_step_choice,
                    "task_score": score,
                    "hint_requests": st.session_state.get("hint_requests", 0),
                    **scale_values,
                }
                st.session_state.setdefault("task_records", []).append(record)
                log_event("task_answer_submitted", **record)
                reset_task_counters()
                if idx + 1 < len(seq):
                    st.session_state["current_task_index"] = idx + 1
                    st.rerun()
                else:
                    st.session_state["phase"] = "posttest"
                    log_event("all_tasks_finished", n_tasks=len(seq))
                    st.rerun()

def render_posttest(config):
    week_cfg = get_week_cfg(config)
    st.markdown(f'<div class="section-label">{week_cfg["week_label"]}</div>', unsafe_allow_html=True)
    st.markdown('<div class="soft-card"><h4>完成三題任務後：後測與整體反思</h4><div class="muted">請依照剛剛三種 XAI 解釋方式的實際體驗作答。</div></div>', unsafe_allow_html=True)

    st.markdown("### 後測（Post-test）")
    post_score, post_answers = grade_mcq(week_cfg["posttest_items"], "posttest_xai_emotion")

    st.markdown("### 整體問卷與反思")
    options = ["A：General Explanation", "B：Actionable Explanation", "C：Actionable Explanation + Hint"]
    st.session_state["overall_preference_understand"] = st.radio("三種解釋方式中，你覺得哪一種最容易理解？", options, key="overall_understand")
    st.session_state["overall_preference_trust"] = st.radio("三種解釋方式中，你最信任哪一種？", options, key="overall_trust")
    st.session_state["overall_preference_next_step"] = st.radio("三種解釋方式中，哪一種最能幫助你知道下一步？", options, key="overall_next_step")
    st.session_state["overall_reflection"] = st.text_area("整體反思：你最喜歡哪一種 AI 解釋方式？為什麼？", value=st.session_state.get("overall_reflection", ""))

    if st.button("完成任務並提交"):
        if st.session_state.get("final_saved"):
            st.info("本任務已提交。")
            return
        duration_sec = ""
        started_at = st.session_state.get("task_started_at")
        if started_at:
            try:
                duration_sec = int((datetime.now(timezone.utc) - datetime.fromisoformat(started_at)).total_seconds())
            except Exception:
                duration_sec = ""

        task_records = st.session_state.get("task_records", [])
        if len(task_records) < 3:
            st.error("尚未完成三題任務，請回任務區完成。")
            return

        for rec in task_records:
            row = {
                "timestamp": utc_now_iso(),
                "session_id": st.session_state.get("session_id", ""),
                "student_id": st.session_state.get("student_id", ""),
                "week": st.session_state.get("week", ""),
                "mode": week_cfg["mode"],
                "prior_xai": st.session_state.get("prior_xai", ""),
                "pretest_score": st.session_state.get("pretest_score"),
                "posttest_score": post_score,
                "learning_gain": post_score - int(st.session_state.get("pretest_score") or 0),
                "pretest_answers_json": json.dumps(st.session_state.get("pretest_answers", {}), ensure_ascii=False),
                "posttest_answers_json": json.dumps(post_answers, ensure_ascii=False),
                "baseline_confidence": st.session_state.get("baseline_confidence", 3),
                "overall_preference_understand": st.session_state.get("overall_preference_understand", ""),
                "overall_preference_trust": st.session_state.get("overall_preference_trust", ""),
                "overall_preference_next_step": st.session_state.get("overall_preference_next_step", ""),
                "overall_reflection": st.session_state.get("overall_reflection", ""),
                "started_at": started_at,
                "submitted_at": utc_now_iso(),
                "duration_sec": duration_sec,
                "task_emotion_video_path": st.session_state.get("emotion_uploaded_video_path", ""),
                **rec,
            }
            save_response(row)

        st.session_state["posttest_score"] = post_score
        st.session_state["final_saved"] = True
        st.session_state["phase"] = "done"
        log_event("task_completed", n_task_records=len(task_records), duration_sec=duration_sec, posttest_score=post_score)
        st.rerun()

def render_done(config):
    week_cfg = get_week_cfg(config)
    st.success("全部步驟完成。")
    st.write(f"你已完成 {week_cfg['week_label']} 的前測、三個 AI 情緒判斷任務、短量表、後測與整體反思。")
    c1, c2, c3 = st.columns(3)
    c1.metric("前測分數", st.session_state.get("pretest_score", "-"))
    c2.metric("後測分數", st.session_state.get("posttest_score", "-"))
    try:
        gain = int(st.session_state.get("posttest_score") or 0) - int(st.session_state.get("pretest_score") or 0)
    except Exception:
        gain = "-"
    c3.metric("Learning gain", gain)
    if st.button("返回首頁"):
        sid = st.session_state.get("student_id", "")
        st.session_state.clear()
        init_session_defaults()
        st.session_state["student_id"] = sid
        st.session_state["phase"] = "login"
        st.rerun()

def main():
    init_session_defaults()
    config = load_config()
    render_header(config)

    if st.session_state.get("role") == "admin":
        if not st.session_state.get("student_id"):
            render_login(config)
            st.stop()
        render_admin()
        st.stop()

    if not ensure_student_state(config):
        render_login(config)
        st.stop()

    phase = st.session_state.get("phase", "login")

    if phase == "login":
        render_login(config)
    elif phase == "intro":
        render_intro(config)
    elif phase == "task":
        render_task(config)
    elif phase == "posttest":
        render_posttest(config)
    elif phase == "done":
        render_done(config)
    else:
        st.session_state["phase"] = "login"
        st.rerun()

if __name__ == "__main__":
    main()
