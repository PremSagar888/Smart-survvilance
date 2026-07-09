import os
from pathlib import Path
os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")

import cv2
import queue
import threading
import time
import argparse
import numpy as np
import psutil
import torch
import json
import base64
import asyncio
import websockets

from pathlib import Path
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
sys.path.insert(0, str(ROOT / "ultralytics-20260605T103000Z-3-001"))

from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, TclError
import webbrowser
import socket

def kill_stale_processes(ports=(8765, 8000)):
    import subprocess
    for port in ports:
        try:
            pids = subprocess.check_output(["lsof", "-t", f"-i:{port}"]).decode().strip().split()
            for pid_str in pids:
                try:
                    pid = int(pid_str)
                    if pid != os.getpid():
                        subprocess.run(["kill", "-9", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        print(f"[Setup] Killed process {pid} using port {port}")
                except Exception:
                    pass
        except Exception:
            pass

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def start_local_http_server(port=8000):
    import http.server
    import socketserver
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(ROOT), **kwargs)
        def log_message(self, format, *args):
            pass
    def run_server():
        socketserver.TCPServer.allow_reuse_address = True
        try:
            with socketserver.TCPServer(("", port), Handler) as httpd:
                httpd.serve_forever()
        except Exception as e:
            print(f"[HTTP Error] Server failed: {e}")
    t = threading.Thread(target=run_server, daemon=True)
    t.start()

class WebSocketBroadcaster:
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.send_queue = queue.Queue()
        self.loop = None
        self.thread = None
        self.paused = False
        self.stopped = False

    def start(self):
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._start_server())
        self.loop.create_task(self._queue_worker())
        self.loop.run_forever()

    async def _start_server(self):
        return await websockets.serve(self._handler, self.host, self.port)

    async def _handler(self, websocket, *args, **kwargs):
        self.clients.add(websocket)
        print(f"[WebSocket] Client connected: {websocket.remote_address}")
        try:
            async for message in websocket:
                msg = message.strip()
                if msg == "PAUSE":
                    self.paused = True
                    print("[WebSocket] Pause signal received from client.")
                elif msg == "RESUME":
                    self.paused = False
                    print("[WebSocket] Resume signal received from client.")
                elif msg == "STOP":
                    self.stopped = True
                    print("[WebSocket] Stop signal received from client.")
        except Exception as e:
            print(f"[WebSocket Error] Handler error: {e}")
        finally:
            self.clients.discard(websocket)
            print(f"[WebSocket] Client disconnected: {websocket.remote_address}")

    async def _queue_worker(self):
        while True:
            while not self.send_queue.empty():
                msg = self.send_queue.get_nowait()
                if self.clients:
                    await asyncio.gather(
                        *[client.send(msg) for client in self.clients],
                        return_exceptions=True
                    )
            await asyncio.sleep(0.05)

    def send(self, message):
        self.send_queue.put(message)

def image_to_base64(img_numpy):
    if img_numpy is None or img_numpy.size == 0:
        return ""
    try:
        h, w = img_numpy.shape[:2]
        max_dim = 480
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img_numpy = cv2.resize(img_numpy, (int(w * scale), int(h * scale)))
        _, buffer = cv2.imencode('.jpg', img_numpy, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{jpg_as_text}"
    except Exception as e:
        print(f"[Base64 Error] {e}")
        return ""

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

# ── Detection colours per COCO class ─────────────────────────────────────────
CUSTOM_COLORS = {
    0:  (0, 255, 0),     # person        — green
    39: (255, 0, 0),     # bottle        — blue
    56: (255, 165, 0),   # chair         — orange
    62: (255, 255, 0),   # tv / monitor  — yellow
    63: (0, 0, 255),     # laptop        — red
    64: (255, 0, 255),   # mouse         — magenta
    65: (0, 255, 255),   # remote        — cyan
    66: (128, 255, 0),   # keyboard      — lime
    67: (255, 128, 0),   # cell phone    — amber
}

# ── Lab assets the system monitors for theft ─────────────────────────────────
LAB_ASSETS = {
    "laptop"  : "laptop",
    "tv"      : "monitor",
    "keyboard": "keyboard",
    "mouse"   : "mouse",
}

# ── Keyframe similarity thresholds ───────────────────────────────────────────
CORR_HIGH = 0.95
CORR_LOW  = 0.80

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def crop_bbox(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()

def normalized_correlation(img_a: np.ndarray, img_b: np.ndarray) -> float:
    size = (64, 64)
    a = cv2.resize(img_a, size).astype(np.float32).flatten()
    b = cv2.resize(img_b, size).astype(np.float32).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-6:
        return 1.0
    return float(np.dot(a, b) / denom)

def asset_centre_in_person_bbox(
    px1: int, py1: int, px2: int, py2: int,
    ax1: int, ay1: int, ax2: int, ay2: int
) -> bool:
    """Return True when the asset bbox centre falls inside the person bbox."""
    acx = (ax1 + ax2) / 2
    acy = (ay1 + ay2) / 2
    return px1 <= acx <= px2 and py1 <= acy <= py2

# ═══════════════════════════════════════════════════════════════════════════════
# QWEN HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_qwen():
    print("[Qwen] Loading Qwen2.5-0.5B-Instruct …")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
    )
    qwen_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    qwen_model.eval()
    print("[Qwen] Loaded.")
    return tokenizer, qwen_model

def build_keyframe_log(
    keyframes_dir: Path,
    all_time_ids: set,
    total_frames: int,
    fps_estimate: float,
    event_log: list,
):
    """
    Build a clean, factual event log from the session.
    Returns (log_str, has_theft: bool, has_pickups: bool).
    Only confirmed, explicitly logged events are included — no inference.
    """
    duration_sec = total_frames / max(fps_estimate, 1)

    theft_lines  = []
    pickup_lines = []
    exit_lines   = []
    assets_seen  = set()

    for evt in event_log:
        el = evt.lower()
        if "theft confirmed" in el:
            theft_lines.append(evt)
        elif "picked up" in el or "confirmed carrying" in el:
            pickup_lines.append(evt)
        elif "exited" in el or "exiting via" in el:
            exit_lines.append(evt)
        # silently track asset presence for the header
        for k in LAB_ASSETS.values():
            if k in el:
                assets_seen.add(k)

    lines = [
        f"Session duration : {duration_sec:.0f} seconds",
        f"Assets present   : {', '.join(sorted(assets_seen)) if assets_seen else 'none detected'}",
        "",
        "=== SECURITY EVENTS ===",
        "",
        "[CONFIRMED THEFT]",
    ]
    lines += [f"  {e}" for e in theft_lines] or ["  None recorded."]

    lines += ["", "[ASSET PICKUPS / CARRY CONFIRMED]"]
    lines += [f"  {e}" for e in pickup_lines] or ["  None recorded."]

    lines += ["", "[EXIT EVENTS WITH ASSETS]"]
    lines += [f"  {e}" for e in exit_lines] or ["  None recorded."]

    if theft_lines:
        risk = "HIGH — Confirmed theft event(s) detected."
    elif pickup_lines:
        risk = "MEDIUM — Asset(s) carried; verify exit."
    else:
        risk = "LOW — No suspicious activity detected."
    lines += ["", f"Risk Level : {risk}"]

    return "\n".join(lines), bool(theft_lines), bool(pickup_lines)

def generate_lab_summary(
    tokenizer,
    qwen_model,
    keyframe_log: str,
    device: str,
    has_theft: bool,
    has_pickups: bool,
) -> str:
    """
    Generate Qwen summary with hard anti-hallucination gates.

    Anti-hallucination strategy:
      1. If no theft AND no pickups → SKIP Qwen entirely and return a
         hardcoded STATUS: NORMAL sentence. The 0.5B model cannot be
         reliably constrained from inventing events; bypassing it for
         clean sessions eliminates false theft reports completely.
      2. When Qwen IS called, the system prompt forbids inferring or
         fabricating any event not explicitly listed in the data.
      3. max_new_tokens is reduced to 150 — the smaller generation window
         gives the model less room to drift into hallucinated content.
    """
    # ── Hard gate for clean sessions (no theft, no pickups) ──────────────────
    if not has_theft and not has_pickups:
        return (
            "No theft or suspicious activity was detected during this monitoring session. "
            "All monitored lab assets remained in their designated locations throughout. "
            "No individuals were observed confirmed-carrying assets toward or through exit zones. "
            "STATUS: NORMAL"
        )

    # ── Qwen only runs when there IS something suspicious to report ───────────
    system_prompt = (
        "You are a lab security analyst writing a theft-detection report. "
        "You MUST report ONLY events that are EXPLICITLY listed in the security data. "
        "RULES — each rule violation is a failure:\n"
        "  1. If [CONFIRMED THEFT] says 'None recorded' → you MUST output STATUS: NORMAL.\n"
        "  2. If [ASSET PICKUPS] says 'None recorded' → do NOT mention any pickups.\n"
        "  3. NEVER infer, assume, or fabricate any event not explicitly in the data.\n"
        "  4. Write exactly 3 sentences.\n"
        "  5. End with exactly one of: STATUS: THEFT CONFIRMED | STATUS: SUSPICIOUS | STATUS: NORMAL"
    )

    user_prompt = (
        f"Security data:\n{keyframe_log}\n\n"
        "Write a 3-sentence theft-detection report based ONLY on the above data. "
        "Do not add anything not listed. End with the correct STATUS."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    qwen_model.to(device)

    with torch.no_grad():
        generated_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=150,     # short window = less hallucination room
            do_sample=False,        # greedy decoding → deterministic output
            temperature=None,
            top_p=None,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated_ids[0][inputs.input_ids.shape[1]:]
    summary    = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    qwen_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run(
    source      = "rtsp://admin:Eelab@2020@10.23.12.23:554/stream1",         # if you want to use another source use another source stream links or source 0 , source 1 
    yolo_weights= "best.pt",
    imgsz       = 640,
    conf_thres  = 0.15,
    iou_thres   = 0.5,
    classes     = [0, 62, 63, 64, 66],   # person, tv, laptop, mouse, keyboard
    save_crop   = True,
    save_vid    = True,
    show_vid    = True,
    project     = str(ROOT / "outputs_intern"),
    name        = "tracking_results",
    frame_skip  = 1,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Setup] Device: {device}")

    model = YOLO(yolo_weights)
    names = model.names

    # ── Resolve the "door" class from the weights (replaces manual door coords) ──
    name_to_id  = {v: k for k, v in names.items()}
    door_cls_id = name_to_id.get("door")
    if door_cls_id is not None:
        classes = list(classes) + [door_cls_id]
        print(f"[Setup] Door class detected: id={door_cls_id} — dynamic door zones enabled.")
    else:
        print("[WARN] No 'door' class found in model.names — door-zone tracking disabled; "
              "entry/exit/theft logic that depends on door zones will not trigger.")

    qwen_tokenizer, qwen_model = load_qwen()

    # ── Output directories ────────────────────────────────────────────────────
    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=True)

    existing = [
        int(f.name.split("_")[1])
        for f in save_dir.iterdir()
        if f.is_dir() and f.name.startswith("results_") and f.name.split("_")[1].isdigit()
    ]
    run_dir = save_dir / f"results_{max(existing, default=0) + 1}"
    run_dir.mkdir(parents=True, exist_ok=True)

    keyframes_dir = run_dir / "keyframes";  keyframes_dir.mkdir(parents=True, exist_ok=True)
    id_photos_dir = run_dir / "id_photos";  id_photos_dir.mkdir(parents=True, exist_ok=True)
    suspects_dir  = run_dir / "suspects";   suspects_dir.mkdir(parents=True,  exist_ok=True)

    timestamp   = time.strftime("%Y%m%d_%H%M%S")
    coords_txt  = run_dir / "person_coordinates.txt"
    summary_txt = run_dir / "final_summary.txt"

    with open(coords_txt, "w") as f:
        f.write("frame,id,x,y,w,h,conf,-1,-1,-1\n")

    # ── Video source ──────────────────────────────────────────────────────────
    print(f"[Setup] Opening source: {source}")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("[Error] Unable to open video source")
        return

    input_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[Setup] Input: {input_width}×{input_height} @ {input_fps:.1f} FPS")

    # Scale live window display to fit standard monitors without cropping (keep aspect ratio)
    max_disp_w = 1280
    max_disp_h = 720
    scale = min(max_disp_w / input_width, max_disp_h / input_height, 1.0)
    video_width  = int(input_width * scale)
    video_height = int(input_height * scale)

    # ── GUI setup displaying only the video feed ─────────────────────────────
    root = None
    video_label = None
    if show_vid:
        try:
            root = tk.Tk()
            root.title("Lab Surveillance Video Feed")
            root.geometry(f"{video_width}x{video_height}")
            root.configure(bg="#000000")

            video_label = ttk.Label(root)
            video_label.pack(fill="both", expand=True)
        except Exception as e:
            print(f"[GUI] Failed to initialize Tkinter: {e}. Reverting to headless mode.")
            show_vid = False


    # ── Tracking state ────────────────────────────────────────────────────────
    track_history   = {}
    ref_crop        = {}
    saved_id_photos = set()
    person_last_crop = {}
    event_log       = []
    previous_person_centers = {}

    # asset_id → {home_center, center, class, bbox_xyxy}
    tracked_assets  = {}

    # Inventory tracking (from theft_new.py — who entered/exited with what)
    person_entry_time = {}   # pid → "HH:MM:SS"
    person_exit_time  = {}   # pid → "HH:MM:SS"
    entry_inventory   = {}   # pid → set of asset names seen near them at first detection
    exit_inventory    = {}   # pid → set of asset names confirmed carried on exit
    person_inventory = {}
    person_activity_log = {}
    logged_pickups = {}

    carry_counter      = {}   # {pid: {asset_id: int}}
    confirmed_carrying = {}   # {pid: set(asset_ids)}
    asset_owner = {}
    ownership_counter = {}

    # ---------------- PERSON ENTRY STATES ----------------

    person_state = {}
    person_exit_door = {}
    person_entry_door = {}

    # ---------------- Person FSM ----------------

    STATE_OUTSIDE = 0
    STATE_ENTRY_ZONE = 1
    STATE_INSIDE = 2
    STATE_EXIT_ZONE = 3
    STATE_EXIT_CROSSED = 4
    STATE_LEFT = 5
    suspect_saved      = set()
    theft_events       = set()

    CARRY_CONFIRM_FRAMES = 8    # frames of bbox overlap → "carrying"
    
    CARRY_DROP_RATE      = 1    # counter decrement per frame without overlap

    # ── Dynamic door zones (built live from YOLO door-class detections) ──────────
    # Replaces the old hardcoded EXIT_ZONES/ENTRY_ZONES/EXIT_LINES/ENTRY_LINES.
    # door_zones: { "Door 1": {"raw_bbox":(x1,y1,x2,y2), "zone":(x1,y1,x2,y2),
    #                          "type":"vertical"/"horizontal", ... line params} }
    door_zones        = {}
    DOOR_ZONE_MARGIN  = 0      # px padding around the detected door bbox
    DOOR_IOU_MATCH    = 0.3    # min IoU to treat a new detection as an existing door

    def _bbox_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / float(area_a + area_b - inter)

    def register_or_update_door_zone(bbox):
        """
        Match a detected door bbox to an existing zone (by IoU) and refresh it,
        or register a brand-new zone (named "Door N" in order of first sighting).
        Returns the zone name.
        """
        x1, y1, x2, y2 = bbox
        for zname, zdata in door_zones.items():
            if _bbox_iou(bbox, zdata["raw_bbox"]) >= DOOR_IOU_MATCH:
                zdata["raw_bbox"] = bbox
                zx1 = x1 - DOOR_ZONE_MARGIN
                zy1 = y1 - DOOR_ZONE_MARGIN
                zx2 = x2 + DOOR_ZONE_MARGIN
                zy2 = y2 + DOOR_ZONE_MARGIN
                zdata["zone"] = (zx1, zy1, zx2, zy2)
                w, h = x2 - x1, y2 - y1
                if h >= w:
                    zdata.update(type="vertical", x=(x1 + x2) / 2, top=zy1, bottom=zy2)
                else:
                    zdata.update(type="horizontal", y=(y1 + y2) / 2, left=zx1, right=zx2)
                return zname
        zname = f"Door {len(door_zones) + 1}"
        zx1 = x1 - DOOR_ZONE_MARGIN
        zy1 = y1 - DOOR_ZONE_MARGIN
        zx2 = x2 + DOOR_ZONE_MARGIN
        zy2 = y2 + DOOR_ZONE_MARGIN
        zdata = {"raw_bbox": bbox, "zone": (zx1, zy1, zx2, zy2)}
        w, h = x2 - x1, y2 - y1
        if h >= w:
            zdata.update(type="vertical", x=(x1 + x2) / 2, top=zy1, bottom=zy2)
        else:
            zdata.update(type="horizontal", y=(y1 + y2) / 2, left=zx1, right=zx2)
        door_zones[zname] = zdata
        print(f"[DOOR] Registered new zone: {zname} @ {zdata['zone']}")
        return zname

    def get_exit_zone_bbox(px1: float, py1: float, px2: float, py2: float):
        """Return the zone name if the person's bbox overlaps with any detected door zone, else None."""
        for zname, zdata in door_zones.items():
            zx1, zy1, zx2, zy2 = zdata["zone"]
            ix1 = max(px1, zx1)
            iy1 = max(py1, zy1)
            ix2 = min(px2, zx2)
            iy2 = min(py2, zy2)
            if ix1 < ix2 and iy1 < iy2:
                return zname
        return None

    def get_exit_zone(cx: float, cy: float):
        """Return the zone name if (cx,cy) falls inside any detected door zone, else None."""
        for zname, zdata in door_zones.items():
            zx1, zy1, zx2, zy2 = zdata["zone"]
            if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                return zname
        return None

    # Entry and exit use the same live door zones.
    get_entry_zone = get_exit_zone
    def crossed_entry_line(previous_center, current_center, door):

        px_prev, py_prev = previous_center
        px_curr, py_curr = current_center

    # -----------------------------
    # Vertical Door
    # -----------------------------
        if door["type"] == "vertical":
            return (
                (
                    px_prev > door["x"] >= px_curr
                    or
                    px_prev < door["x"] <= px_curr
                )
                and
                door["top"] <= py_curr <= door["bottom"]
            )

    # -----------------------------
    # Horizontal Door
    # -----------------------------
        elif door["type"] == "horizontal":
            return (
                (
                    py_prev > door["y"] >= py_curr
                    or
                     py_prev < door["y"] <= py_curr
                )
                and
                door["left"] <= px_curr <= door["right"]
            )
        return False
        
    def crossed_exit_line(previous_center, current_center, door):

        px_prev, py_prev = previous_center
        px_curr, py_curr = current_center
        if door["type"] == "vertical":
            return (
                (
                    px_prev < door["x"] <= px_curr
                    or
                    px_prev > door["x"] >= px_curr
                )
                and
                door["top"] <= py_curr <= door["bottom"]
            )
        elif door["type"] == "horizontal":
            return (
                (
                    py_prev < door["y"] <= py_curr
                    or
                    py_prev > door["y"] >= py_curr
                )
                and
                door["left"] <= px_curr <= door["right"]
            )
        return False   

    # ── Performance / timing state ────────────────────────────────────────────
    MAX_HISTORY          = 30
    ID_TIMEOUT           = 15.0   # seconds before expired track is removed
    GUI_INTERVAL         = 3      # update GUI every N processed frames
    MEM_INTERVAL         = 30
    COORD_FLUSH_INTERVAL = 30

    prev_time    = time.time()
    total_frames = 0
    all_time_ids = set()
    frame_idx    = 0
    fps_running  = 30.0
    memory_usage = 0.0
    coord_buffer = []

    kill_stale_processes()
    local_ip = get_local_ip()

    ws_server = WebSocketBroadcaster()
    ws_server.start()
    print("[WebSocket] Background server started at ws://localhost:8765")
    start_local_http_server(8000)

    # Save and push an initial session.json at startup to let dashboards discover the live WS server
    initial_session_data = {
        "status": "NORMAL",
        "unique_persons": 0,
        "total_frames": 0,
        "avg_confidence": "0.0%",
        "summary": "Session started. Awaiting video feed...",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "suspects": [],
        "id_photos": [],
        "chart_data": {
            "labels": [],
            "persons": [],
            "objects": []
        },
        "events": [],
        "ws_url": f"ws://{local_ip}:8765"
    }

    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "session.json", "w") as f:
            json.dump(initial_session_data, f, indent=2)
    except Exception:
        pass

    git_repo_path = ROOT
    if (git_repo_path / ".git").exists():
        try:
            with open(git_repo_path / "dashboard" / "session.json", "w") as f:
                json.dump(initial_session_data, f, indent=2)
            
            try:
                import subprocess
                token = "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"
                repo_url = f"https://{token}@github.com/PremSagar888/Smart-survvilance.git"
                clean_url = "https://github.com/PremSagar888/Smart-survvilance.git"
                
                print("[Git] Deploying initial session.json to GitHub Pages...")
                subprocess.run(["git", "-C", str(git_repo_path), "remote", "set-url", "origin", repo_url], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["git", "-C", str(git_repo_path), "add", "-f", "dashboard/session.json"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["git", "-C", str(git_repo_path), "commit", "-m", "Auto-update initial session logs from python script"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["git", "-C", str(git_repo_path), "push", "origin", "main"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["git", "-C", str(git_repo_path), "remote", "set-url", "origin", clean_url], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("[Git] Successfully deployed initial session to GitHub Pages!")
            except Exception as git_err:
                print(f"[Git Error] Could not auto-deploy initial session to GitHub Pages: {git_err}")
                try:
                    subprocess.run(["git", "-C", str(git_repo_path), "remote", "set-url", "origin", "https://github.com/PremSagar888/Smart-survvilance.git"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass
        except Exception:
            pass

    # Automatically pop up the local website dashboard in the browser using the local HTTP server
    try:
        local_uri = f"http://127.0.0.1:8000/dashboard/index.html?ip=127.0.0.1"
        webbrowser.open(local_uri)
        print(f"[Browser] Opened local website dashboard: {local_uri}")
    except Exception as e:
        print(f"[Browser Error] Could not open browser: {e}")

    print(f"\n" + "="*70)
    print(f"📡  LIVE SURVEILLANCE DASHBOARD CHANNELS ACTIVE")
    print(f"----------------------------------------------------------------------")
    print(f"💻  Local PC Browser (Auto-opened):")
    print(f"    {local_uri}")
    print(f"📱  Other Devices (Mobile/Tablet/PC on same network):")
    print(f"    👉  http://{local_ip}:8000/dashboard/index.html?ip={local_ip}  (Recommended — avoids HTTPS Mixed Content blocks)")
    print(f"    👉  https://PremSagar888.github.io/Smart-survvilance/dashboard/?ip={local_ip}")
    print(f"="*70 + "\n")

    persons_history = []
    objects_history = []
    labels_history  = []

    clahe_obj    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ── Async video writer ────────────────────────────────────────────────────
    write_queue  = queue.Queue(maxsize=32)
    vid_writer   = None
    raw_writer   = None

    raw_path = str(run_dir / "raw.mp4")
    tracked_path = str(run_dir / "tracked.mp4")

    def _writer_thread():
        while True:
            item = write_queue.get()
            if item is None:
                break
            writer, frame = item
            writer.write(frame)
            write_queue.task_done()

    writer_thread = threading.Thread(target=_writer_thread, daemon=True)
    writer_thread.start()

    # ═════════════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ═════════════════════════════════════════════════════════════════════════
    while True:
        # Handle WebSocket Pause / Resume control
        if ws_server.paused:
            time.sleep(0.1)
            if show_vid and root is not None:
                try:
                    root.update_idletasks()
                    root.update()
                except Exception:
                    pass
            continue

        # Handle WebSocket Stop control
        if ws_server.stopped:
            print("[WebSocket] Stop signal acknowledged. Exiting frame loop...")
            break

        ret, im0 = cap.read()
        if not ret or im0 is None:
            break
        raw_frame     = im0.copy()
        frame_idx    += 1
        total_frames += 1
        # CLAHE contrast enhancement — every 3rd frame (reuse pre-built object)
        if frame_idx % 3 == 0:
            lab = cv2.cvtColor(im0, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l   = clahe_obj.apply(l)
            im0 = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        # YOLO + ByteTrack (every frame keeps Kalman filters consistent)
        results = model.track(
            im0,
            persist=True,
            tracker="bytetrack.yaml",
            classes=classes,
            conf=conf_thres,
            iou=iou_thres,
            imgsz=imgsz,
            agnostic_nms=False,
            half=True,
            max_det=100,
            verbose=False,
        )
        # frame_skip gates heavy per-frame processing but NOT the tracker itself
        if frame_idx % max(1, frame_skip) != 0:
            continue
        # ── Parse this frame's detections ────────────────────────────────────
        tracks              = []
        person_count        = 0
        object_count        = 0
        person_conf_sum     = 0.0
        person_conf_cnt     = 0
        person_bboxes       = {}   # pid  → (x1,y1,x2,y2)
        person_centers      = {}   # pid  → (cx,cy)
        current_frame_assets = set()   # asset track_ids visible this frame
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids   = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            clss  = results[0].boxes.cls.cpu().numpy().astype(int)
            for box, track_id, conf, cls in zip(boxes, ids, confs, clss):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w  = x2 - x1
                h  = y2 - y1
                tracks.append({
                    "id": track_id, "bbox": [cx, cy, w, h],
                    "bbox_xyxy": (x1, y1, x2, y2),
                    "class": cls, "conf": conf,
                })
                # ── PERSON ──────────────────────────────────────────────────
                if cls == 0:
                    person_centers[track_id] = (cx, cy)
                    person_bboxes[track_id]  = (x1, y1, x2, y2)
                    person_count  += 1
                    all_time_ids.add(track_id)
                    person_conf_sum += float(conf)
                    person_conf_cnt += 1
                    coord_buffer.append(
                        f"{frame_idx},{track_id},{x1},{y1},{w},{h},{conf:.4f},-1,-1,-1\n"
                    )
                    # Track history for motion trail
                    th = track_history.setdefault(track_id, [])
                    th.append((int(cx), int(cy)))
                    if len(th) > MAX_HISTORY:
                        track_history[track_id] = th[-MAX_HISTORY:]
                    if track_id not in person_state:
                        cx, cy = person_centers[track_id]
                        if get_entry_zone(cx, cy) is not None:
                            person_state[track_id] = STATE_OUTSIDE
                        else:
                            person_state[track_id] = STATE_INSIDE
                            person_entry_time[track_id] = time.strftime("%H:%M:%S")
                            entry_inventory.setdefault(track_id, set())
                            person_inventory.setdefault(track_id, set())
                            person_activity_log.setdefault(track_id, [])
                            print(f"[ENTRY] Person {track_id} already inside laboratory")
                    if track_id in previous_person_centers:
                        entered = False
                        entry_door = None
                        cx, cy = person_centers[track_id]
                        prev_cx, prev_cy = previous_person_centers[track_id]
                        entry_zone_now = get_entry_zone(cx, cy)
                        entry_zone_prev = get_entry_zone(prev_cx, prev_cy)
                        if (
                            person_state[track_id] == STATE_OUTSIDE
                            and
                            entry_zone_now is not None
                        ):
                            door = door_zones[entry_zone_now]
                            if (
                                entry_zone_prev != entry_zone_now
                                and
                                crossed_entry_line(
                                    previous_person_centers[track_id],
                                    person_centers[track_id],
                                    door
                                )
                            ):
                                entered = True
                                entry_door = entry_zone_now
                        if entered and person_state[track_id] != STATE_INSIDE:
                            person_state[track_id] = STATE_INSIDE
                            person_entry_time[track_id] = time.strftime("%H:%M:%S")
                            person_entry_door[track_id] = entry_door
                            entry_inventory.setdefault(track_id, set())
                            person_inventory.setdefault(track_id, set())
                            person_activity_log.setdefault(
                                track_id,
                                []
                            )
                            print(f"[ENTRY] Person {track_id} entered through {entry_door}")
                            event_log.append(
                                f"Person {track_id} entered through {entry_door}"
                            )
                            for asset_id, asset in tracked_assets.items():
                                if asset_id not in current_frame_assets:
                                    continue
                                ax1, ay1, ax2, ay2 = asset["bbox_xyxy"]
                                px1, py1, px2, py2 = person_bboxes[track_id]
                                if asset_centre_in_person_bbox(
                                    px1,
                                    py1,
                                    px2,
                                    py2,
                                    ax1,
                                    ay1,
                                    ax2,
                                    ay2,
                                ):
                                    entry_inventory[track_id].add(asset["class"])
                    # Keyframe extraction
                    crop = crop_bbox(raw_frame, x1, y1, x2, y2)
                    if crop is not None:
                        person_last_crop[track_id] = crop.copy()
                    if crop is not None:
                        if track_id not in ref_crop:
                            ref_crop[track_id] = crop.copy()
                            if track_id not in saved_id_photos:
                                d = id_photos_dir / f"id_{track_id}"
                                d.mkdir(parents=True, exist_ok=True)
                                cv2.imwrite(str(d / f"id_{track_id}.jpg"), crop)
                                saved_id_photos.add(track_id)
                                crop_b64 = image_to_base64(crop)
                                if crop_b64:
                                    ws_server.send(f"ID_PHOTO|{track_id}|{crop_b64}")

                        elif frame_idx % 20 == 0:
                            corr = normalized_correlation(ref_crop[track_id], crop)
                            if corr > CORR_HIGH:
                                ref_crop[track_id] = crop.copy()
                            elif corr < CORR_LOW:
                                if save_crop:
                                    d = keyframes_dir / f"id_{track_id}"
                                    d.mkdir(parents=True, exist_ok=True)
                                    cv2.imwrite(str(d / f"frame_{frame_idx}.jpg"), crop)
                                ref_crop[track_id] = crop.copy()
                            else:
                                ref_crop[track_id] = crop.copy()
                # ── DOOR (feeds the dynamic zone registry) ────────────────────
                elif door_cls_id is not None and cls == door_cls_id:
                    _door_zone_name = register_or_update_door_zone((x1, y1, x2, y2))
                # ── OBJECT / ASSET ──────────────────────────────────────────
                else:
                    object_count += 1
                    asset_display_name = LAB_ASSETS.get(names[cls])
                    if asset_display_name is not None:
                        current_frame_assets.add(track_id)
                        if track_id not in tracked_assets:
                            tracked_assets[track_id] = {
                                "home_center": (cx, cy),
                                "center":      (cx, cy),
                                "class":       asset_display_name,
                                "bbox_xyxy":   (x1, y1, x2, y2),
                            }
                        else:
                            tracked_assets[track_id]["center"]    = (cx, cy)
                            tracked_assets[track_id]["bbox_xyxy"] = (x1, y1, x2, y2)
                # ── Draw bbox + label ────────────────────────────────────────
                color = CUSTOM_COLORS.get(cls, (0, 255, 0))
                if cls == 0:
                    label = f"Person {track_id} {conf:.2f}"
                elif door_cls_id is not None and cls == door_cls_id:
                    label = f"{_door_zone_name} {conf:.2f}"
                else:
                    _aname = LAB_ASSETS.get(names[cls], names[cls])
                    label  = f"{_aname} {track_id} {conf:.2f}"
                cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
                cv2.putText(im0, label, (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        # ─────────────────────────────────────────────────────────────────────
        # THEFT DETECTION — NEW TWO-STAGE LOGIC
        # ─────────────────────────────────────────────────────────────────────
        # ── STAGE 1 : CARRY CONFIRMATION ─────────────────────────────────────
        for asset_id, asset in tracked_assets.items():
            if asset_id not in current_frame_assets:
                # The asset is missing from the video feed (e.g. picked up and occluded).
                # Keep it in confirmed_carrying if they were already carrying it.
                continue
            ax1, ay1, ax2, ay2 = asset["bbox_xyxy"]
            # Find the single person whose bbox contains this asset's centre
            overlapping_pid = None
            for pid, (px1, py1, px2, py2) in person_bboxes.items():
                if asset_centre_in_person_bbox(px1, py1, px2, py2, ax1, ay1, ax2, ay2):
                    overlapping_pid = pid
                    break
            if overlapping_pid is not None:
                pid_cc = carry_counter.setdefault(overlapping_pid, {})
                pid_cc[asset_id] = pid_cc.get(asset_id, 0) + 1
                if pid_cc[asset_id] == CARRY_CONFIRM_FRAMES:
                    confirmed_carrying.setdefault(overlapping_pid, set()).add(asset_id)
                    logged_pickups.setdefault(overlapping_pid, set())
                    carry_msg = (
                        f"Person {overlapping_pid} picked up "
                        f"{asset['class']} (confirmed frame {frame_idx})"
                    )
                    event_log.append(carry_msg)
                    person_inventory.setdefault(overlapping_pid, set()).add(asset["class"])
                    if asset_id not in logged_pickups[overlapping_pid]:
                        activity_time = time.strftime("%H:%M:%S")
                        activity = (
                            f"{activity_time} - Picked {asset['class']}"
                        )
                        person_activity_log.setdefault(
                            overlapping_pid,
                            []
                        ).append(
                            activity
                        )
                        logged_pickups[overlapping_pid].add(asset_id)
                    print(f"[CARRY] {carry_msg}")
            else:
                # The asset is detected, but not inside any person's bbox.
                # Only decay counter and discard carrying status if it is sitting freely.
                for pid in list(carry_counter.keys()):
                    if asset_id in carry_counter[pid]:
                        carry_counter[pid][asset_id] = max(
                            0, carry_counter[pid][asset_id] - CARRY_DROP_RATE
                        )
                        if carry_counter[pid][asset_id] == 0:
                            if pid in confirmed_carrying:
                                confirmed_carrying[pid].discard(asset_id)
        # ── STAGE 2 : EXIT CONFIRMATION ───────────────────────────────────────
        # Snapshot person-in-exit-zone state from the PREVIOUS frame
        # before we update it for the current frame.
        for pid, carried_set in list(confirmed_carrying.items()):
            if not carried_set:
                continue
            if pid not in person_centers:
                continue
            # Check if person is approaching/inside exit zone
            if pid in person_bboxes:
                px1, py1, px2, py2 = person_bboxes[pid]
                zone_name = get_exit_zone_bbox(px1, py1, px2, py2)
            else:
                zone_name = None
            if zone_name is not None:
                person_exit_door[pid] = zone_name
                if person_state.get(pid, STATE_INSIDE) == STATE_INSIDE:
                    person_state[pid] = STATE_EXIT_ZONE
                    print(f"[EXIT] Person {pid} approaching exit zone {zone_name}")
                    print(f"[STATE] Person {pid} -> STATE_EXIT_ZONE")
                    # Broadcast suspicious status to WebSocket
                    asset_names = [tracked_assets[aid]["class"] for aid in carried_set if aid in tracked_assets]
                    if asset_names:
                        susp_msg = f"WARNING | Person {pid} carrying {', '.join(asset_names)} approaching exit {zone_name}"
                        if susp_msg not in event_log:
                            event_log.append(susp_msg)
                            ws_server.send(f"STATUS|SUSPICIOUS")
                            print(f"[SUSPICIOUS] {susp_msg}")
                # Draw approaching exit indicator on video feed
                cv2.putText(
                    im0,
                    f"APPROACHING EXIT : Person {pid}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2,
                )
        # Door zones are tracked internally, not drawn on output video
        # ── Flush coordinate buffer ───────────────────────────────────────────
        if frame_idx % COORD_FLUSH_INTERVAL == 0 and coord_buffer:
            with open(coords_txt, "a") as f:
                f.writelines(coord_buffer)
            coord_buffer.clear()
        # ── FPS + memory ──────────────────────────────────────────────────────
        curr_time   = time.time()
        fps         = 1.0 / max(curr_time - prev_time, 1e-6)
        fps_running = fps
        prev_time   = curr_time
        if frame_idx % MEM_INTERVAL == 0:
            memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
        avg_conf = person_conf_sum / person_conf_cnt * 100 if person_conf_cnt > 0 else 0.0
        # Accumulate chart data once per second (or every N frames based on FPS)
        if frame_idx % max(1, int(input_fps)) == 0:
            time_str = f"{len(labels_history)}s"
            labels_history.append(time_str)
            persons_history.append(person_count)
            objects_history.append(object_count)
        # ── Dashboard update & WebSocket broadcast ───────────────────────────
        if frame_idx % GUI_INTERVAL == 0:
            # Get threat status
            status = "NORMAL"
            if theft_events:
                status = "THEFT"
            elif any(len(assets) > 0 for assets in confirmed_carrying.values()):
                status = "SUSPICIOUS"
            # Broadcast stats and status to WebSocket clients
            ws_server.send(f"STATS|{total_frames}|{person_count}|{object_count}|{avg_conf:.1f}%")
            ws_server.send(f"STATUS|{status}")

        # ── Video display (Tkinter window) ────────────────────────────────────
        if show_vid:
            try:
                im0_resized = cv2.resize(im0, (video_width, video_height))
                im0_rgb     = cv2.cvtColor(im0_resized, cv2.COLOR_BGR2RGB)
                img         = Image.fromarray(im0_rgb)
                imgtk       = ImageTk.PhotoImage(image=img)
                video_label.imgtk = imgtk
                video_label.configure(image=imgtk)
                root.update_idletasks()
                root.update()
            except TclError:
                print("[GUI] Window closed safely.")
                break
            except Exception as e:
                print(f"[WARN] GUI display error: {e}. Reverting to headless mode.")
                show_vid = False
        # ── Video save (async, native resolution) ─────────────────────────────
        if save_vid:
            if vid_writer is None:
                # Initialize tracked writer
                vid_writer = cv2.VideoWriter(
                    tracked_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    input_fps,
                    (input_width, input_height),
                )
                print(f"[VideoWriter] Tracked video → {tracked_path}")
                
                # Initialize raw writer
                raw_writer = cv2.VideoWriter(
                    raw_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    input_fps,
                    (input_width, input_height),
                )
                print(f"[VideoWriter] Raw video → {raw_path}")

            frame_to_write = (
                im0 if (im0.shape[1] == input_width and im0.shape[0] == input_height)
                else cv2.resize(im0, (input_width, input_height))
            )
            raw_to_write = (
                raw_frame if (raw_frame.shape[1] == input_width and raw_frame.shape[0] == input_height)
                else cv2.resize(raw_frame, (input_width, input_height))
            )
            
            if not write_queue.full():
                write_queue.put((vid_writer, frame_to_write))
            if not write_queue.full():
                write_queue.put((raw_writer, raw_to_write))
        current_person_ids = set(person_centers.keys())
        for pid in list(person_state.keys()):
            if (
                person_state[pid] == STATE_EXIT_ZONE
                and
                pid not in current_person_ids
            ):
                person_state[pid] = STATE_LEFT
                print(f"[STATE] Person {pid} -> STATE_LEFT")
                person_exit_time[pid] = time.strftime("%H:%M:%S")
                carried_assets = confirmed_carrying.get(pid, set())
                if carried_assets:
                    for asset_id in carried_assets:
                        if asset_id in tracked_assets:
                            exit_inventory.setdefault(pid, set()).add(tracked_assets[asset_id]["class"])
                print(f"[EXIT] Person {pid} left laboratory (disappeared while carrying assets near exit)")
                for asset_id in carried_assets:
                    if asset_id not in tracked_assets:
                        continue
                    asset_name = tracked_assets[asset_id]["class"]
                    event = (
                        f"THEFT CONFIRMED | "
                        f"Person {pid} | "
                        f"{asset_name} | "
                        f"Exited through {person_exit_door.get(pid, 'Unknown')}"
                    )
                    if event not in theft_events:
                        theft_events.add(event)
                        event_log.append(event)
                        # Save suspect image only once
                        if pid not in suspect_saved:
                            if pid in person_last_crop:
                                d = suspects_dir / f"person_{pid}"
                                d.mkdir(parents=True, exist_ok=True)
                                cv2.imwrite(
                                    str(d / "suspect.jpg"),
                                    person_last_crop[pid]
                                )
                                suspect_saved.add(pid)
                                suspect_b64 = image_to_base64(person_last_crop[pid])
                                timestamp_now = time.strftime("%H:%M:%S")
                                ws_server.send(f"SUSPECT|{pid}|{suspect_b64}|{timestamp_now}|{event}")
                                ws_server.send(f"STATUS|THEFT")

                        print(f"[THEFT] {event}")
        previous_person_centers = person_centers.copy()
        if show_vid and root is not None and frame_idx % GUI_INTERVAL == 0:
            try:
                root.update_idletasks()
                root.update()
            except TclError:
                break

    # ═════════════════════════════════════════════════════════════════════════
    # END OF SESSION — flush buffers, generate Qwen report
    # ═════════════════════════════════════════════════════════════════════════
    if coord_buffer:
        with open(coords_txt, "a") as f:
            f.writelines(coord_buffer)

    write_queue.put(None)   # sentinel → stop writer thread
    writer_thread.join()

    print("\n=== Generating LAB SECURITY REPORT with Qwen ===")

    keyframe_log, has_theft, has_pickups = build_keyframe_log(
        keyframes_dir, all_time_ids, total_frames, fps_running, event_log
    )
    print("\n[Event Log]:\n" + "\n".join(event_log))
    print("\n[Keyframe Log]:\n" + keyframe_log)

    summary = generate_lab_summary(
        qwen_tokenizer, qwen_model,
        keyframe_log, device,
        has_theft, has_pickups,
    )

    # Person inventory report (from theft_new.py — who entered/exited with what)
    inventory_report = "\n\nPERSON INVENTORY REPORT\n" + "=" * 50 + "\n"
    for pid in sorted(all_time_ids):
        entry_items = ", ".join(entry_inventory.get(pid, set())) or "none"
        exit_items  = ", ".join(exit_inventory.get(pid, set()))  or "none"
        stolen_items = (
            exit_inventory.get(pid, set())
            -
            entry_inventory.get(pid, set())
        )
        stolen_text = (
            ", ".join(sorted(stolen_items))
            if stolen_items
            else "none"
        ) 
        entry_door = person_entry_door.get(pid, "Unknown")
        exit_door = person_exit_door.get(pid, "Still inside")
        activities = person_activity_log.get(pid, [])
        if activities:
            activity_text = "\n".join(
                f"    {a}" for a in activities
            )
        else:
            activity_text = "    None"
        inventory_report += (
            f"\n{'='*60}\n"
            f"Person ID : {pid}\n\n"
            f"Entry Door : {entry_door}\n"
            f"Entry Time : {person_entry_time.get(pid, 'unknown')}\n\n"
            f"Carried In : {entry_items}\n\n"
            f"Activities\n"
            f"----------\n"
            f"{activity_text}\n\n"
            f"Exit Door  : {exit_door}\n"
            f"Exit Time  : {person_exit_time.get(pid, 'still in lab')}\n\n"
            f"Carried Out : {exit_items}\n"
            f"\n"
            f"Stolen Items : {stolen_text}\n"
        )
    with open(summary_txt, "w") as f:
        f.write("LAB SECURITY THEFT-DETECTION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Session   : {timestamp}\n")
        if door_zones:
            for _dname, _dzone in door_zones.items():
                f.write(f"{_dname:9s}: zone={_dzone['zone']} type={_dzone['type']}\n")
        else:
            f.write("Doors     : none detected this session\n")
        f.write("=" * 60 + "\n\n")
        f.write(summary + "\n")
        f.write(inventory_report + "\n")
        if theft_events:
            f.write("\nTHEFT EVENT LOG:\n")
            for e in sorted(theft_events):
                f.write(f"  {e}\n")

    print(f"\n[Qwen] Summary:\n{summary}")
    print(f"[Done] Report → {summary_txt}")

    # ── Generate session.json for dashboard ──────────────────────────────────
    status_final = "NORMAL"
    if theft_events:
        status_final = "THEFT"
    elif any(len(assets) > 0 for assets in confirmed_carrying.values()):
        status_final = "SUSPICIOUS"

    # Build suspects list with base64 photos
    suspects_list = []
    for pid in sorted(suspect_saved):
        suspect_file_dir = suspects_dir / f"person_{pid}"
        suspect_file = suspect_file_dir / "suspect.jpg"
        if suspect_file.exists():
            img_suspect = cv2.imread(str(suspect_file))
            photo_b64 = image_to_base64(img_suspect)
        else:
            photo_b64 = ""
        details_msg = next((e for e in event_log if f"Person {pid}" in e and "THEFT" in e), "Theft suspected")
        suspects_list.append({
            "id": int(pid),
            "photo": photo_b64,
            "timestamp": person_exit_time.get(pid, time.strftime("%H:%M:%S")),
            "details": details_msg
        })

    # Build id_photos list with base64 photos
    id_photos_list = []
    for pid in sorted(saved_id_photos):
        id_photo_file = id_photos_dir / f"id_{pid}" / f"id_{pid}.jpg"
        if id_photo_file.exists():
            img_id = cv2.imread(str(id_photo_file))
            photo_b64 = image_to_base64(img_id)
        else:
            photo_b64 = ""
        id_photos_list.append({
            "id": int(pid),
            "photo": photo_b64
        })

    # Build full JSON data
    session_json_data = {
        "status": status_final,
        "unique_persons": len(all_time_ids),
        "total_frames": total_frames,
        "avg_confidence": f"{avg_conf:.1f}%",
        "summary": summary,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "suspects": suspects_list,
        "id_photos": id_photos_list,
        "chart_data": {
            "labels": labels_history,
            "persons": persons_history,
            "objects": objects_history
        },
        "events": event_log,
        "ws_url": f"ws://{local_ip}:8765"
    }

    # Save locally to run_dir
    session_json_path = run_dir / "session.json"
    try:
        with open(session_json_path, "w") as f:
            json.dump(session_json_data, f, indent=2)
        print(f"[Done] Dashboard JSON saved → {session_json_path}")
    except Exception as e:
        print(f"[Error saving local JSON] {e}")

    # Save to git repository directory for hosting
    git_repo_path = ROOT
    if (git_repo_path / ".git").exists():
        git_json_path = git_repo_path / "dashboard" / "session.json"
        try:
            with open(git_json_path, "w") as f:
                json.dump(session_json_data, f, indent=2)
            print(f"[Done] Git repo JSON saved → {git_json_path}")
            
            # Send FINISHED and FINISHED_DATA messages over WebSocket to bypass CORS fetches
            ws_server.send(f"FINISHED_DATA|{json.dumps(session_json_data)}")
            ws_server.send(f"FINISHED|session.json")
            
            # Automatically commit and push the updated session.json to GitHub Pages
            try:
                import subprocess
                token = "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"
                repo_url = f"https://{token}@github.com/PremSagar888/Smart-survvilance.git"
                clean_url = "https://github.com/PremSagar888/Smart-survvilance.git"
                
                print("[Git] Deploying updated session.json to GitHub Pages...")
                # Temporarily configure token
                subprocess.run(["git", "-C", str(git_repo_path), "remote", "set-url", "origin", repo_url], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Commit and push
                subprocess.run(["git", "-C", str(git_repo_path), "add", "-f", "dashboard/session.json"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["git", "-C", str(git_repo_path), "commit", "-m", "Auto-update session logs from python script"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["git", "-C", str(git_repo_path), "push", "origin", "main"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Clean up remote URL
                subprocess.run(["git", "-C", str(git_repo_path), "remote", "set-url", "origin", clean_url], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("[Git] Successfully deployed to GitHub Pages!")
            except Exception as git_err:
                print(f"[Git Error] Could not auto-deploy to GitHub Pages: {git_err}")
                # Ensure clean URL is set in case of error
                try:
                    subprocess.run(["git", "-C", str(git_repo_path), "remote", "set-url", "origin", "https://github.com/PremSagar888/Smart-survvilance.git"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass
        except Exception as e:
            print(f"[Error saving git JSON] {e}")

    # Give the WebSocket background thread a moment to flush the final report to the dashboard
    print("[WebSocket] Flushing final report to dashboard...")
    time.sleep(2.0)

    if vid_writer is not None:
        vid_writer.release()
    if raw_writer is not None:
        raw_writer.release()
    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    if show_vid and root is not None:
        try:
            root.destroy()
            print("[GUI] Video window closed successfully.")
        except Exception:
            pass

# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_opt():
    p = argparse.ArgumentParser(description="Lab Security — YOLO + ByteTrack + Qwen")
    p.add_argument("--source",        type=str,   default="rtsp://admin:Eelab@2020@10.23.12.23:554/stream1")
    p.add_argument("--yolo-weights",  type=str,   default="best.pt")
    p.add_argument("--imgsz",         type=int,   default=640)
    p.add_argument("--conf-thres",    type=float, default=0.15)
    p.add_argument("--iou-thres",     type=float, default=0.5)
    p.add_argument("--frame-skip",    type=int,   default=1)
    p.add_argument("--show-vid",      type=str,   default="True")
    return p.parse_args()

def main(opt):
    source = int(opt.source) if str(opt.source).isdigit() else str(opt.source)
    show_vid = opt.show_vid.lower() == "true"
    run(
        source       = source,
        yolo_weights = opt.yolo_weights,
        imgsz        = opt.imgsz,
        conf_thres   = opt.conf_thres,
        iou_thres    = opt.iou_thres,
        frame_skip   = opt.frame_skip,
        show_vid     = show_vid,
    )

if __name__ == "__main__":
    main(parse_opt())
