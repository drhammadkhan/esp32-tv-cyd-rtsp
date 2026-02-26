import os
import subprocess
import threading
import time
import json
import secrets
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request
from video_server.video_preprocessor import process_videos

app = Flask(__name__)

FRAME_SIZE = (320, 240)
VIDEO_SERVER_PORT = int(os.getenv("VIDEO_SERVER_PORT", "8123"))

settings_lock = threading.Lock()
settings = {
    "rtsp_url": os.getenv("RTSP_URL", "").strip(),
    "streams": [],
    "active_stream_index": 0,
    "jpeg_quality": int(os.getenv("JPEG_QUALITY", "82")),
    "swap_rb": os.getenv("SWAP_RB", "0") == "1",
    "contrast": float(os.getenv("CONTRAST", "1.12")),
    "brightness": int(os.getenv("BRIGHTNESS", "-4")),
    "saturation": float(os.getenv("SATURATION", "1.18")),
    "preset": os.getenv("STREAM_PRESET", "balanced"),
}
if settings["rtsp_url"]:
    settings["streams"] = [{"name": "Stream 1", "url": settings["rtsp_url"]}]
    settings["active_stream_index"] = 0

capture_state_lock = threading.Lock()
capture_state = {
    "source": "",
    "status": "starting",
    "last_error": "",
    "last_frame_ms": 0,
}

active_stream_lock = threading.Lock()
active_stream_index = 0
last_requested_channel_lock = threading.Lock()
last_requested_channel_index = -1

video_data = []
latest_jpeg = None
latest_lock = threading.Lock()
running = False

PLAYER_DIR = (Path(__file__).resolve().parent.parent / "player").resolve()
LOCAL_OVERRIDES_PATH = PLAYER_DIR / "src" / "LocalOverrides.h"
SETTINGS_PATH = (Path(__file__).resolve().parent / "cache" / "settings.json").resolve()
STATIC_FIRMWARE_DIR = (Path(__file__).resolve().parent / "static" / "firmware").resolve()

WEBFLASH_SSID_TOKEN = b"CFG_WIFI_SSID_PLACEHOLDER_XXXXXXXXXXXXXXXX"
WEBFLASH_PASSWORD_TOKEN = b"CFG_WIFI_PASSWORD_PLACEHOLDER_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
WEBFLASH_HOST_TOKEN = b"CFG_VIDEO_SERVER_HOST_PLACEHOLDER_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
WEBFLASH_PORT_TOKEN = b"8124P"

webflash_lock = threading.Lock()
webflash_payloads = {}

flash_lock = threading.Lock()
flash_state = {
    "running": False,
    "status": "idle",
    "last_error": "",
    "last_exit_code": None,
    "last_started_ms": 0,
    "last_finished_ms": 0,
    "log_lines": [],
    "active_job": {},
}


def _now_ms():
    return int(time.time() * 1000)


def _get_settings():
    with settings_lock:
        return dict(settings)


def _normalize_streams(raw_streams):
    streams = []
    if not isinstance(raw_streams, list):
        return streams
    for idx, item in enumerate(raw_streams):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        if not name:
            name = f"Stream {len(streams) + 1}"
        streams.append({"name": name[:64], "url": url[:2048]})
        if len(streams) >= 16:
            break
    return streams


def _sync_rtsp_url_locked():
    streams = settings.get("streams", [])
    idx = int(settings.get("active_stream_index", 0))
    if len(streams) == 0:
        settings["active_stream_index"] = 0
        settings["rtsp_url"] = ""
        return
    if idx < 0 or idx >= len(streams):
        idx = 0
    settings["active_stream_index"] = idx
    settings["rtsp_url"] = streams[idx]["url"]


def _settings_for_response():
    cfg = _get_settings()
    streams = _normalize_streams(cfg.get("streams", []))
    idx = int(cfg.get("active_stream_index", 0))
    if len(streams) == 0 and cfg.get("rtsp_url", "").strip():
        streams = [{"name": "Stream 1", "url": cfg["rtsp_url"].strip()}]
        idx = 0
    if len(streams) == 0:
        idx = 0
    elif idx < 0 or idx >= len(streams):
        idx = 0
    cfg["streams"] = streams
    cfg["active_stream_index"] = idx
    cfg["rtsp_url"] = streams[idx]["url"] if streams else ""
    return cfg


def _persist_settings():
    data = _settings_for_response()
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_settings():
    if not SETTINGS_PATH.exists():
        return
    try:
        loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception as ex:
        print(f"warning: failed to read settings file: {ex}")
        return
    with settings_lock:
        if "streams" in loaded:
            settings["streams"] = _normalize_streams(loaded.get("streams", []))
        if "active_stream_index" in loaded:
            try:
                settings["active_stream_index"] = int(loaded.get("active_stream_index", 0))
            except Exception:
                settings["active_stream_index"] = 0
        if "rtsp_url" in loaded:
            settings["rtsp_url"] = str(loaded.get("rtsp_url", "")).strip()
        if "jpeg_quality" in loaded:
            try:
                settings["jpeg_quality"] = int(loaded["jpeg_quality"])
            except Exception:
                pass
        if "swap_rb" in loaded:
            settings["swap_rb"] = bool(loaded["swap_rb"])
        if "contrast" in loaded:
            try:
                settings["contrast"] = float(loaded["contrast"])
            except Exception:
                pass
        if "brightness" in loaded:
            try:
                settings["brightness"] = int(loaded["brightness"])
            except Exception:
                pass
        if "saturation" in loaded:
            try:
                settings["saturation"] = float(loaded["saturation"])
            except Exception:
                pass
        if "preset" in loaded:
            settings["preset"] = str(loaded["preset"])
        if len(settings.get("streams", [])) == 0 and settings["rtsp_url"]:
            settings["streams"] = [{"name": "Stream 1", "url": settings["rtsp_url"]}]
            settings["active_stream_index"] = 0
        _sync_rtsp_url_locked()


def _get_active_stream_index():
    with active_stream_lock:
        return active_stream_index


def _set_active_stream_index(idx):
    global active_stream_index
    with active_stream_lock:
        active_stream_index = idx


def _reset_last_requested_channel():
    global last_requested_channel_index
    with last_requested_channel_lock:
        last_requested_channel_index = -1


def _set_active_stream_from_channel(channel_index):
    cfg = _settings_for_response()
    streams = cfg.get("streams", [])
    if len(streams) == 0:
        _set_active_stream_index(0)
        return 0
    # Only switch stream when channel number changes; repeated /frame requests
    # for the same channel should not fight manual selection in the web UI.
    global last_requested_channel_index
    with last_requested_channel_lock:
        if channel_index == last_requested_channel_index:
            return _get_active_stream_index() % len(streams)
        last_requested_channel_index = channel_index
    idx = channel_index % len(streams)
    _set_active_stream_index(idx)
    return idx


def _get_active_stream_url():
    cfg = _settings_for_response()
    streams = cfg.get("streams", [])
    if len(streams) == 0:
        return "", 0
    idx = _get_active_stream_index() % len(streams)
    return streams[idx]["url"], idx


def _is_rtsp_mode():
    return len(_settings_for_response().get("streams", [])) > 0


def _set_capture_state(status=None, source=None, last_error=None):
    with capture_state_lock:
        if status is not None:
            capture_state["status"] = status
        if source is not None:
            capture_state["source"] = source
        if last_error is not None:
            capture_state["last_error"] = last_error


def _flash_supported():
    return PLAYER_DIR.exists()


def _append_flash_log(line):
    with flash_lock:
        flash_state["log_lines"].append(line.rstrip())
        if len(flash_state["log_lines"]) > 500:
            flash_state["log_lines"] = flash_state["log_lines"][-500:]


def _escape_c_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _write_local_overrides(ssid, password, host, port):
    content = f"""#pragma once
// Auto-generated by /admin firmware flasher UI.
// Update values via web UI instead of editing manually.

#define WIFI_SSID "{_escape_c_string(ssid)}"
#define WIFI_PASSWORD "{_escape_c_string(password)}"
#define VIDEO_SERVER_HOST "{_escape_c_string(host)}"
#define VIDEO_SERVER_PORT {int(port)}
#define VIDEO_SERVER_PORT_STR "{int(port)}"
"""
    LOCAL_OVERRIDES_PATH.write_text(content, encoding="utf-8")


def _run_flash_job(job):
    if not _flash_supported():
        with flash_lock:
            flash_state["running"] = False
            flash_state["status"] = "failed"
            flash_state["last_error"] = "Firmware source tree not available in this runtime"
            flash_state["last_exit_code"] = -1
            flash_state["last_finished_ms"] = _now_ms()
        _append_flash_log("ERROR: firmware flashing is disabled in this deployment")
        return

    env_name = "cheap-yellow-display-no-audio" if job["flavor"] == "no_audio" else "cheap-yellow-display"
    cmd = ["python3", "-m", "platformio", "run", "-e", env_name, "-t", "upload"]
    if job.get("upload_port"):
        cmd.extend(["--upload-port", job["upload_port"]])

    with flash_lock:
        flash_state["running"] = True
        flash_state["status"] = "preparing"
        flash_state["last_error"] = ""
        flash_state["last_exit_code"] = None
        flash_state["last_started_ms"] = _now_ms()
        flash_state["last_finished_ms"] = 0
        flash_state["log_lines"] = []
        flash_state["active_job"] = {
            "flavor": job["flavor"],
            "ssid": job["ssid"],
            "server_host": job["server_host"],
            "server_port": job["server_port"],
            "upload_port": job.get("upload_port", ""),
        }
    _append_flash_log(f"Preparing firmware job: {env_name}")
    _append_flash_log(f"Writing overrides to: {LOCAL_OVERRIDES_PATH}")
    try:
        _write_local_overrides(job["ssid"], job["password"], job["server_host"], job["server_port"])
    except Exception as ex:
        with flash_lock:
            flash_state["running"] = False
            flash_state["status"] = "failed"
            flash_state["last_error"] = f"Failed writing overrides: {ex}"
            flash_state["last_exit_code"] = -1
            flash_state["last_finished_ms"] = _now_ms()
        _append_flash_log(f"ERROR: {flash_state['last_error']}")
        return

    with flash_lock:
        flash_state["status"] = "flashing"
    _append_flash_log("Starting: " + " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        cwd=str(PLAYER_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        _append_flash_log(line)
    rc = proc.wait()
    with flash_lock:
        flash_state["running"] = False
        flash_state["last_exit_code"] = rc
        flash_state["last_finished_ms"] = _now_ms()
        if rc == 0:
            flash_state["status"] = "success"
            flash_state["last_error"] = ""
        else:
            flash_state["status"] = "failed"
            flash_state["last_error"] = f"PlatformIO exited with code {rc}"
    _append_flash_log(f"Finished with exit code: {rc}")


def _list_serial_ports():
    ports = []
    for pattern in ("cu.usb*", "cu.wch*", "cu.SLAB*", "ttyUSB*"):
        for path in sorted(Path("/dev").glob(pattern)):
            ports.append("/dev/" + path.name)
    return ports


def _encode_black_frame():
    frame = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    return buf.tobytes() if ok else b""


def _webflash_templates_exist():
    required = [
        STATIC_FIRMWARE_DIR / "bootloader-audio.bin",
        STATIC_FIRMWARE_DIR / "partitions-audio.bin",
        STATIC_FIRMWARE_DIR / "template-firmware-audio.bin",
        STATIC_FIRMWARE_DIR / "bootloader-no-audio.bin",
        STATIC_FIRMWARE_DIR / "partitions-no-audio.bin",
        STATIC_FIRMWARE_DIR / "template-firmware-no-audio.bin",
    ]
    return all(p.exists() for p in required)


def _cleanup_webflash_payloads():
    cutoff = _now_ms() - (30 * 60 * 1000)
    with webflash_lock:
        stale = [k for k, v in webflash_payloads.items() if v.get("created_ms", 0) < cutoff]
        for k in stale:
            del webflash_payloads[k]


def _patch_token(blob: bytes, token: bytes, value: str, max_len: int) -> bytes:
    raw = value.encode("utf-8")
    if len(raw) > max_len:
        raise ValueError(f"value too long for token (max {max_len})")
    if token not in blob:
        raise ValueError("firmware token not found")
    replacement = raw + b"\x00" + (b"\x00" * (len(token) - len(raw) - 1))
    return blob.replace(token, replacement, 1)


def _build_custom_webflash_firmware(flavor: str, ssid: str, password: str, server_host: str, server_port: int) -> bytes:
    template_name = "template-firmware-no-audio.bin" if flavor == "no_audio" else "template-firmware-audio.bin"
    blob = (STATIC_FIRMWARE_DIR / template_name).read_bytes()
    blob = _patch_token(blob, WEBFLASH_SSID_TOKEN, ssid, 32)
    blob = _patch_token(blob, WEBFLASH_PASSWORD_TOKEN, password, 63)
    blob = _patch_token(blob, WEBFLASH_HOST_TOKEN, server_host, 63)
    blob = _patch_token(blob, WEBFLASH_PORT_TOKEN, str(server_port), 4)
    return blob


def _fit_frame(frame):
    target_w, target_h = FRAME_SIZE
    src_h, src_w = frame.shape[:2]
    if src_w == 0 or src_h == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = max(target_w / src_w, target_h / src_h)
    resized_w = int(src_w * scale)
    resized_h = int(src_h * scale)
    # Keep downscaling smooth while preserving detail on any upscaled input.
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=interpolation)
    x0 = (resized_w - target_w) // 2
    y0 = (resized_h - target_h) // 2
    return resized[y0:y0 + target_h, x0:x0 + target_w]


def _enhance_frame(frame, contrast, brightness, saturation, preset):
    # Mild contrast/brightness correction plus saturation lift for CYD TFT.
    frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    if saturation != 1.0:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if preset == "best_quality":
        # Gentle denoise + unsharp mask to recover detail on low-cost TFT.
        frame = cv2.bilateralFilter(frame, d=5, sigmaColor=22, sigmaSpace=22)
        blurred = cv2.GaussianBlur(frame, (0, 0), 1.2)
        frame = cv2.addWeighted(frame, 1.2, blurred, -0.2, 0)
        # Slight gamma lift for punchier midtones.
        gamma = 1.08
        lut = np.array([pow(i / 255.0, 1.0 / gamma) * 255 for i in range(256)]).astype("uint8")
        frame = cv2.LUT(frame, lut)
    return frame


def _rtsp_capture_loop():
    global latest_jpeg
    while running:
        rtsp_url, selected_index = _get_active_stream_url()
        if not rtsp_url:
            _set_capture_state(status="idle", source="", last_error="")
            with latest_lock:
                latest_jpeg = _encode_black_frame()
            time.sleep(0.3)
            continue
        _set_capture_state(status="connecting", source=rtsp_url, last_error="")
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            _set_capture_state(status="error", source=rtsp_url, last_error="failed to open RTSP stream")
            time.sleep(1.0)
            continue
        _set_capture_state(status="streaming", source=rtsp_url, last_error="")
        last_emit = 0.0
        while running:
            cfg = _get_settings()
            streams = _settings_for_response().get("streams", [])
            if len(streams) == 0:
                break
            current_idx = _get_active_stream_index() % len(streams)
            current_url = streams[current_idx]["url"]
            if current_idx != selected_index or current_url != rtsp_url:
                # Active stream changed in UI or by channel select; reconnect.
                break
            ok, frame = cap.read()
            if not ok:
                _set_capture_state(status="reconnecting", source=rtsp_url, last_error="stream read failed")
                break
            frame = _fit_frame(frame)
            preset = cfg.get("preset", "balanced")
            frame = _enhance_frame(frame, cfg["contrast"], cfg["brightness"], cfg["saturation"], preset)
            if cfg["swap_rb"]:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), cfg["jpeg_quality"]])
            if ok:
                # Simple output FPS limiter for quality mode, gives CPU headroom to process better.
                now = time.time()
                target_fps = 12 if preset == "best_quality" else 0
                if target_fps > 0 and last_emit > 0 and (now - last_emit) < (1.0 / target_fps):
                    continue
                with latest_lock:
                    latest_jpeg = buf.tobytes()
                last_emit = now
                with capture_state_lock:
                    capture_state["last_frame_ms"] = _now_ms()
                    capture_state["status"] = "streaming"
        cap.release()
        time.sleep(0.5)


def init_video_source():
    global video_data, latest_jpeg, running
    _load_settings()
    _set_active_stream_index(_settings_for_response().get("active_stream_index", 0))
    _reset_last_requested_channel()
    if _is_rtsp_mode():
        latest_jpeg = _encode_black_frame()
        running = True
        threading.Thread(target=_rtsp_capture_loop, daemon=True).start()
        print("RTSP mode enabled")
    else:
        video_data = process_videos("movies", FRAME_SIZE)
        print(f"Movie mode: loaded {len(video_data)} channels from movies/")


def _ensure_movies_loaded():
    global video_data
    if len(video_data) == 0:
        video_data = process_videos("movies", FRAME_SIZE)


@app.route('/channel_info')
def get_channel_lengths():
    if _is_rtsp_mode():
        count = len(_settings_for_response().get("streams", []))
        if count <= 0:
            count = 1
        return jsonify([2_147_483_647] * count)
    _ensure_movies_loaded()
    lengths = [len(audio) for audio, frames in video_data]
    return jsonify(lengths)


@app.route('/audio/<int:channel_index>/<int:start>/<int:length>')
def get_audio(channel_index, start, length):
    if _is_rtsp_mode():
        _set_active_stream_from_channel(channel_index)
        if length <= 0:
            return Response(b'', mimetype='audio/x-raw')
        return Response(bytes([128]) * length, mimetype='audio/x-raw')
    _ensure_movies_loaded()
    audio, frames = video_data[channel_index % len(video_data)]
    end = start + length
    if end > len(audio):
        end = len(audio)
    if start > end:
        start = end
    if start == end:
        # return nothing - we've got no more data to give
        return Response(b'', mimetype='audio/x-raw')
    slice = audio[start:end]
    return Response(slice, mimetype='audio/x-raw')


@app.route('/frame/<int:channel_index>/<int:ms>')
def get_frame(channel_index, ms):
    if _is_rtsp_mode():
        _set_active_stream_from_channel(channel_index)
        with latest_lock:
            data = latest_jpeg if latest_jpeg is not None else _encode_black_frame()
        return Response(data, mimetype='image/jpeg')
    _ensure_movies_loaded()
    audio, frames = video_data[channel_index % len(video_data)]
    # use binary search to find the closest frame
    start = 0
    end = len(frames) - 1
    while start <= end:
        mid = (start + end) // 2
        if frames[mid][0] == ms:
            return Response(frames[mid][1], mimetype='image/jpeg')
        elif frames[mid][0] < ms:
            start = mid + 1
        else:
            end = mid - 1
    # we may not find the exact frame, so return the closest frame
    if end < 0:
        end = 0
    elif start >= len(frames):
        start = len(frames) - 1
    return Response(frames[start][1], mimetype='image/jpeg')


@app.route("/preview.mjpg")
def preview_mjpg():
    def _generate():
        boundary = b"--frame\r\n"
        while True:
            with latest_lock:
                frame = latest_jpeg if latest_jpeg is not None else _encode_black_frame()
            yield boundary
            yield b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.08)

    return Response(_generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    cfg = _settings_for_response()
    with capture_state_lock:
        state = dict(capture_state)
    streams = cfg.get("streams", [])
    if len(streams) > 0:
        idx = _get_active_stream_index() % len(streams)
        state["active_stream_index"] = idx
        state["active_stream_name"] = streams[idx]["name"]
    return jsonify({"settings": cfg, "state": state})


@app.route("/api/settings", methods=["POST"])
def api_set_settings():
    global running, latest_jpeg, video_data
    payload = request.get_json(silent=True) or {}
    errors = []
    updates = {}

    if "rtsp_url" in payload:
        updates["rtsp_url"] = str(payload["rtsp_url"]).strip()
    if "streams" in payload:
        streams = _normalize_streams(payload.get("streams", []))
        updates["streams"] = streams
        if len(streams) == 0:
            updates["active_stream_index"] = 0
    if "active_stream_index" in payload:
        try:
            updates["active_stream_index"] = int(payload["active_stream_index"])
        except Exception:
            errors.append("active_stream_index must be an integer")
    if "jpeg_quality" in payload:
        try:
            jpeg_quality = int(payload["jpeg_quality"])
            if jpeg_quality < 40 or jpeg_quality > 95:
                raise ValueError
            updates["jpeg_quality"] = jpeg_quality
        except Exception:
            errors.append("jpeg_quality must be an integer between 40 and 95")
    if "swap_rb" in payload:
        updates["swap_rb"] = bool(payload["swap_rb"])
    if "contrast" in payload:
        try:
            contrast = float(payload["contrast"])
            if contrast < 0.6 or contrast > 1.8:
                raise ValueError
            updates["contrast"] = contrast
        except Exception:
            errors.append("contrast must be between 0.6 and 1.8")
    if "brightness" in payload:
        try:
            brightness = int(payload["brightness"])
            if brightness < -40 or brightness > 40:
                raise ValueError
            updates["brightness"] = brightness
        except Exception:
            errors.append("brightness must be between -40 and 40")
    if "saturation" in payload:
        try:
            saturation = float(payload["saturation"])
            if saturation < 0.6 or saturation > 1.8:
                raise ValueError
            updates["saturation"] = saturation
        except Exception:
            errors.append("saturation must be between 0.6 and 1.8")
    if "preset" in payload:
        preset = str(payload["preset"]).strip()
        if preset not in ("max_fps", "balanced", "best_quality"):
            errors.append("preset must be max_fps, balanced, or best_quality")
        else:
            updates["preset"] = preset

    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    with settings_lock:
        settings.update(updates)
        if "rtsp_url" in updates and "streams" not in updates:
            if updates["rtsp_url"]:
                settings["streams"] = [{"name": "Stream 1", "url": updates["rtsp_url"]}]
                settings["active_stream_index"] = 0
            else:
                settings["streams"] = []
                settings["active_stream_index"] = 0
        _sync_rtsp_url_locked()
        applied = dict(settings)
    _set_active_stream_index(applied.get("active_stream_index", 0))
    persist_warning = ""
    try:
        _persist_settings()
    except Exception as ex:
        persist_warning = f"Settings applied in memory, but failed to save to disk: {ex}"
        print("warning:", persist_warning)
    # Allow live switching between movie mode and RTSP mode without restarting the server.
    now_rtsp_mode = len(applied.get("streams", [])) > 0
    if now_rtsp_mode and not running:
        latest_jpeg = _encode_black_frame()
        running = True
        threading.Thread(target=_rtsp_capture_loop, daemon=True).start()
        _set_capture_state(status="starting", source="", last_error="")
    elif (not now_rtsp_mode) and running:
        running = False
        _set_capture_state(status="idle", source="", last_error="")
        if len(video_data) == 0:
            video_data = process_videos("movies", FRAME_SIZE)
    return jsonify({"ok": True, "settings": applied, "warning": persist_warning})


@app.route("/api/flash/status", methods=["GET"])
def api_flash_status():
    with flash_lock:
        data = dict(flash_state)
        data["active_job"] = dict(flash_state["active_job"])
        data["log_lines"] = list(flash_state["log_lines"])
    data["ports"] = _list_serial_ports()
    data["supported"] = _flash_supported()
    if not data["supported"] and not data["last_error"]:
        data["last_error"] = "Firmware flasher disabled: player source tree not present (typical in container runtime)."
    return jsonify(data)


@app.route("/api/flash", methods=["POST"])
def api_flash_start():
    if not _flash_supported():
        return jsonify({"ok": False, "error": "Firmware flasher disabled in this deployment. Use local checkout for USB flashing."}), 501

    payload = request.get_json(silent=True) or {}
    flavor = str(payload.get("flavor", "audio_on")).strip()
    ssid = str(payload.get("ssid", "")).strip()
    password = str(payload.get("password", ""))
    server_host = str(payload.get("server_host", "")).strip()
    upload_port = str(payload.get("upload_port", "")).strip()
    try:
        server_port = int(payload.get("server_port", 8124))
    except Exception:
        return jsonify({"ok": False, "error": "server_port must be an integer"}), 400

    if flavor not in ("audio_on", "no_audio"):
        return jsonify({"ok": False, "error": "flavor must be audio_on or no_audio"}), 400
    if not ssid:
        return jsonify({"ok": False, "error": "SSID is required"}), 400
    if not server_host:
        return jsonify({"ok": False, "error": "Server host is required"}), 400
    if server_port < 1 or server_port > 65535:
        return jsonify({"ok": False, "error": "Server port must be 1-65535"}), 400

    with flash_lock:
        if flash_state["running"]:
            return jsonify({"ok": False, "error": "A flash job is already running"}), 409

    job = {
        "flavor": flavor,
        "ssid": ssid,
        "password": password,
        "server_host": server_host,
        "server_port": server_port,
        "upload_port": upload_port,
    }
    threading.Thread(target=_run_flash_job, args=(job,), daemon=True).start()
    return jsonify({"ok": True, "message": "Flash job started"})


@app.route("/api/webflash/prepare", methods=["POST"])
def api_webflash_prepare():
    if not _webflash_templates_exist():
        return jsonify({"ok": False, "error": "Web flash templates are not available in this deployment"}), 501

    payload = request.get_json(silent=True) or {}
    flavor = str(payload.get("flavor", "no_audio")).strip()
    ssid = str(payload.get("ssid", "")).strip()
    password = str(payload.get("password", ""))
    server_host = str(payload.get("server_host", "")).strip()
    try:
        server_port = int(payload.get("server_port", 8124))
    except Exception:
        return jsonify({"ok": False, "error": "server_port must be an integer"}), 400

    if flavor not in ("audio_on", "no_audio"):
        return jsonify({"ok": False, "error": "flavor must be audio_on or no_audio"}), 400
    if not ssid:
        return jsonify({"ok": False, "error": "SSID is required"}), 400
    if not server_host:
        return jsonify({"ok": False, "error": "Server host is required"}), 400
    if server_port < 1 or server_port > 65535:
        return jsonify({"ok": False, "error": "Server port must be 1-65535"}), 400

    try:
        firmware = _build_custom_webflash_firmware(flavor, ssid, password, server_host, server_port)
    except Exception as ex:
        return jsonify({"ok": False, "error": f"Failed to build custom firmware: {ex}"}), 500

    payload_id = secrets.token_hex(12)
    _cleanup_webflash_payloads()
    with webflash_lock:
        webflash_payloads[payload_id] = {
            "created_ms": _now_ms(),
            "flavor": flavor,
            "firmware": firmware,
        }
    return jsonify({
        "ok": True,
        "manifest_url": f"/api/webflash/manifest/{payload_id}.json",
        "expires_minutes": 30,
    })


@app.route("/api/webflash/manifest/<payload_id>.json", methods=["GET"])
def api_webflash_manifest(payload_id):
    with webflash_lock:
        entry = webflash_payloads.get(payload_id)
    if not entry:
        return jsonify({"ok": False, "error": "Manifest not found or expired"}), 404

    flavor = entry["flavor"]
    suffix = "no-audio" if flavor == "no_audio" else "audio"
    manifest = {
        "name": "ESP32 TV CYD Custom",
        "version": "1.0.0",
        "new_install_prompt_erase": True,
        "builds": [{
            "chipFamily": "ESP32",
            "parts": [
                {"path": f"/static/firmware/bootloader-{suffix}.bin", "offset": 4096},
                {"path": f"/static/firmware/partitions-{suffix}.bin", "offset": 32768},
                {"path": f"/api/webflash/bin/{payload_id}.bin", "offset": 65536},
            ],
        }],
    }
    return jsonify(manifest)


@app.route("/api/webflash/bin/<payload_id>.bin", methods=["GET"])
def api_webflash_bin(payload_id):
    with webflash_lock:
        entry = webflash_payloads.get(payload_id)
    if not entry:
        return Response("Not found", status=404, mimetype="text/plain")
    return Response(entry["firmware"], mimetype="application/octet-stream")


@app.route("/admin")
def admin_ui():
    return render_template_string(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CYD Stream Control</title>
  <style>
    :root {
      --bg-1: #0b1320;
      --bg-2: #16243a;
      --panel: rgba(255,255,255,0.88);
      --ink: #132033;
      --muted: #516176;
      --accent: #0d8fdf;
      --ok: #15803d;
      --warn: #b45309;
      --error: #b91c1c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(1000px 600px at 90% -10%, #265685, transparent 70%),
        radial-gradient(1000px 700px at -10% 110%, #1c3a59, transparent 70%),
        linear-gradient(160deg, var(--bg-1), var(--bg-2));
      min-height: 100vh;
      padding: 22px;
    }
    .wrap {
      max-width: 1100px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      gap: 18px;
    }
    .card {
      background: var(--panel);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 18px 40px rgba(0,0,0,0.25);
      backdrop-filter: blur(6px);
    }
    h1 {
      margin: 0 0 6px;
      font-size: 26px;
      letter-spacing: .2px;
    }
    p { margin: 0 0 14px; color: var(--muted); }
    .row { margin: 10px 0; }
    label { display:block; font-weight: 700; margin-bottom: 6px; font-size: 14px; }
    input[type="text"], input[type="number"], input[type="range"], select {
      width: 100%;
    }
    input[type="text"], input[type="number"], select {
      border: 1px solid #d6deea;
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 14px;
      background: #fff;
    }
    .inline {
      display: grid;
      grid-template-columns: 1fr 90px;
      gap: 8px;
      align-items: center;
    }
    input[type="range"] { accent-color: var(--accent); }
    .toggle { display: flex; align-items: center; gap: 10px; font-weight: 700; }
    .actions {
      display: flex;
      gap: 8px;
      margin-top: 14px;
    }
    button {
      border: 0;
      border-radius: 10px;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
    }
    .primary { background: var(--accent); color: #fff; }
    .secondary { background: #e2e8f0; color: #1f2937; }
    .status {
      margin-top: 10px;
      padding: 10px;
      border-radius: 10px;
      background: #eef6ff;
      font-size: 13px;
    }
    .status.ok { color: var(--ok); }
    .status.warn { color: var(--warn); }
    .status.err { color: var(--error); }
    .preview {
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: cover;
      border-radius: 12px;
      border: 2px solid #d7e2ef;
      background: #0c1420;
    }
    .meta {
      margin-top: 8px;
      color: #445468;
      font-size: 13px;
      display: grid;
      gap: 3px;
    }
    @media (max-width: 900px) {
      .wrap { grid-template-columns: 1fr; }
    }
    .span-2 { grid-column: 1 / -1; }
    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      background: #0d1726;
      color: #d5e3f4;
      border-radius: 10px;
      padding: 10px;
      min-height: 220px;
      max-height: 320px;
      overflow: auto;
      white-space: pre-wrap;
      font-size: 12px;
      border: 1px solid #263a56;
    }
    .pill {
      display: inline-block;
      font-weight: 700;
      font-size: 12px;
      border-radius: 999px;
      padding: 4px 10px;
      background: #dbeafe;
      color: #1e3a8a;
      margin-bottom: 8px;
    }
    .firm-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .stream-list {
      display: grid;
      gap: 8px;
    }
    .stream-row {
      display: grid;
      grid-template-columns: 0.8fr 1.8fr auto;
      gap: 8px;
      align-items: center;
    }
    .tiny {
      padding: 8px 10px;
      font-size: 12px;
    }
    @media (max-width: 900px) {
      .firm-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <script type="module" src="https://unpkg.com/esp-web-tools@10/dist/web/install-button.js?module"></script>
  <div class="wrap">
    <section class="card">
      <h1>CYD Stream Control</h1>
      <p>Update URL and image tuning live. Changes are applied immediately.</p>

      <div class="row">
        <label for="rtsp_url">RTSP URL</label>
        <input id="rtsp_url" type="text" placeholder="rtsp://user:pass@ip:port/path" />
      </div>
      <div class="row">
        <label for="active_stream_index">Saved Streams</label>
        <select id="active_stream_index"></select>
      </div>
      <div id="stream_list" class="stream-list"></div>
      <div class="actions">
        <button class="secondary" id="add_stream_btn">Add Stream</button>
      </div>

      <div class="row">
        <label for="jpeg_quality">JPEG Quality</label>
        <div class="inline">
          <input id="jpeg_quality" type="range" min="40" max="95" step="1" />
          <input id="jpeg_quality_num" type="number" min="40" max="95" step="1" />
        </div>
      </div>

      <div class="row">
        <label for="contrast">Contrast</label>
        <div class="inline">
          <input id="contrast" type="range" min="0.6" max="1.8" step="0.01" />
          <input id="contrast_num" type="number" min="0.6" max="1.8" step="0.01" />
        </div>
      </div>

      <div class="row">
        <label for="brightness">Brightness</label>
        <div class="inline">
          <input id="brightness" type="range" min="-40" max="40" step="1" />
          <input id="brightness_num" type="number" min="-40" max="40" step="1" />
        </div>
      </div>

      <div class="row">
        <label for="saturation">Saturation</label>
        <div class="inline">
          <input id="saturation" type="range" min="0.6" max="1.8" step="0.01" />
          <input id="saturation_num" type="number" min="0.6" max="1.8" step="0.01" />
        </div>
      </div>

      <div class="row toggle">
        <input id="swap_rb" type="checkbox" />
        <label for="swap_rb" style="margin:0">Swap Red/Blue Channels</label>
      </div>

      <div class="actions">
        <button class="primary" id="save_btn">Apply Settings</button>
        <button class="secondary" id="refresh_btn">Reload Current</button>
        <button class="secondary" id="preset_fps_btn">Preset: Max FPS</button>
        <button class="secondary" id="preset_quality_btn">Preset: Best Quality</button>
      </div>
      <div id="status_box" class="status">Loading…</div>
    </section>

    <section class="card">
      <img class="preview" src="/preview.mjpg" alt="Live Preview" />
      <div class="meta">
        <div><strong>Status:</strong> <span id="meta_status">-</span></div>
        <div><strong>Active stream:</strong> <span id="meta_active_stream">-</span></div>
        <div><strong>Source:</strong> <span id="meta_source">-</span></div>
        <div><strong>Last frame:</strong> <span id="meta_frame">-</span></div>
        <div><strong>Last error:</strong> <span id="meta_error">-</span></div>
      </div>
    </section>

    <section class="card span-2">
      <div class="pill">Firmware Flasher</div>
      <p>Build and flash CYD firmware directly from this server.</p>
      <div class="firm-grid">
        <div class="row">
          <label for="fw_flavor">Firmware Flavor</label>
          <select id="fw_flavor">
            <option value="audio_on">Audio On</option>
            <option value="no_audio">Audio Off (Higher FPS)</option>
          </select>
        </div>
        <div class="row">
          <label for="fw_upload_port">USB Port (optional)</label>
          <input id="fw_upload_port" type="text" placeholder="/dev/cu.usbserial-1110" />
        </div>
        <div class="row">
          <label for="fw_ssid">Wi-Fi SSID</label>
          <input id="fw_ssid" type="text" placeholder="Your Wi-Fi name" />
        </div>
        <div class="row">
          <label for="fw_password">Wi-Fi Password</label>
          <input id="fw_password" type="text" placeholder="Your Wi-Fi password" />
        </div>
        <div class="row">
          <label for="fw_server_host">Server Host/IP</label>
          <input id="fw_server_host" type="text" placeholder="192.168.1.72" />
        </div>
        <div class="row">
          <label for="fw_server_port">Server Port</label>
          <input id="fw_server_port" type="number" min="1" max="65535" step="1" value="8124" />
        </div>
      </div>
      <div class="actions">
        <button class="primary" id="flash_btn">Build + Flash CYD</button>
        <button class="secondary" id="refresh_flash_btn">Refresh Flash Status</button>
      </div>
      <div id="flash_status_box" class="status">Idle</div>
      <div id="flash_ports" class="meta"></div>
      <div id="flash_log" class="mono"></div>
      <div class="row" style="margin-top: 16px;">
        <label>Browser USB Flash (works with container deployments)</label>
        <p style="margin: 0 0 8px; color: var(--muted);">
          Use Chrome/Edge on the device physically connected to CYD via USB.
        </p>
        <div class="firm-grid">
          <div class="row">
            <label for="webflash_ssid">Wi-Fi SSID</label>
            <input id="webflash_ssid" type="text" placeholder="Your Wi-Fi name" />
          </div>
          <div class="row">
            <label for="webflash_password">Wi-Fi Password</label>
            <input id="webflash_password" type="text" placeholder="Your Wi-Fi password" />
          </div>
          <div class="row">
            <label for="webflash_server_host">Server Host/IP</label>
            <input id="webflash_server_host" type="text" placeholder="192.168.1.72" />
          </div>
          <div class="row">
            <label for="webflash_server_port">Server Port</label>
            <input id="webflash_server_port" type="number" min="1" max="65535" step="1" value="8124" />
          </div>
        </div>
        <div class="actions">
          <div>
            <div style="font-size: 12px; color: var(--muted); margin-bottom: 4px;">Audio On</div>
            <button class="secondary" id="webflash_prepare_audio_btn">Prepare + Flash</button>
            <div id="webflash_audio_slot"></div>
          </div>
          <div>
            <div style="font-size: 12px; color: var(--muted); margin-bottom: 4px;">Audio Off (Higher FPS)</div>
            <button class="secondary" id="webflash_prepare_no_audio_btn">Prepare + Flash</button>
            <div id="webflash_no_audio_slot"></div>
          </div>
        </div>
        <div id="webflash_status" class="status" style="margin-top: 10px;">Ready</div>
      </div>
    </section>
  </div>

  <script>
    const ids = ["jpeg_quality", "contrast", "brightness", "saturation"];
    function bindPair(id) {
      const slider = document.getElementById(id);
      const number = document.getElementById(id + "_num");
      slider.addEventListener("input", () => number.value = slider.value);
      number.addEventListener("input", () => slider.value = number.value);
    }
    ids.forEach(bindPair);

    const statusBox = document.getElementById("status_box");
    function setStatus(msg, mode) {
      statusBox.textContent = msg;
      statusBox.className = "status " + (mode || "");
    }

    const flashStatusBox = document.getElementById("flash_status_box");
    const flashLog = document.getElementById("flash_log");
    const flashPorts = document.getElementById("flash_ports");
    const webflashStatus = document.getElementById("webflash_status");
    const webflashAudioSlot = document.getElementById("webflash_audio_slot");
    const webflashNoAudioSlot = document.getElementById("webflash_no_audio_slot");
    function setFlashStatus(msg, mode) {
      flashStatusBox.textContent = msg;
      flashStatusBox.className = "status " + (mode || "");
    }
    function setWebflashStatus(msg, mode) {
      webflashStatus.textContent = msg;
      webflashStatus.className = "status " + (mode || "");
    }

    let isDirty = false;
    const streamListEl = document.getElementById("stream_list");
    const activeStreamEl = document.getElementById("active_stream_index");

    function getStreamRows() {
      const rows = [];
      document.querySelectorAll(".stream-row").forEach((row) => {
        const name = (row.querySelector(".stream-name").value || "").trim();
        const url = (row.querySelector(".stream-url").value || "").trim();
        if (url) {
          rows.push({ name: name || `Stream ${rows.length + 1}`, url });
        }
      });
      return rows;
    }

    function refreshActiveStreamOptions(selectedIndex) {
      const streams = getStreamRows();
      activeStreamEl.innerHTML = "";
      streams.forEach((stream, i) => {
        const opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = `${i + 1}. ${stream.name}`;
        activeStreamEl.appendChild(opt);
      });
      if (streams.length > 0) {
        const idx = Math.max(0, Math.min(Number(selectedIndex || 0), streams.length - 1));
        activeStreamEl.value = String(idx);
      }
    }

    function addStreamRow(stream, shouldRefresh) {
      const row = document.createElement("div");
      row.className = "stream-row";
      row.innerHTML = `
        <input class="stream-name" type="text" maxlength="64" placeholder="Name" value="${(stream.name || "").replace(/"/g, "&quot;")}" />
        <input class="stream-url" type="text" placeholder="rtsp://..." value="${(stream.url || "").replace(/"/g, "&quot;")}" />
        <button type="button" class="secondary tiny remove-stream-btn">Remove</button>
      `;
      streamListEl.appendChild(row);
      if (shouldRefresh) {
        refreshActiveStreamOptions(activeStreamEl.value || 0);
      }
    }

    function setStreams(streams, activeIndex) {
      streamListEl.innerHTML = "";
      (streams || []).forEach((s) => addStreamRow(s, false));
      if ((streams || []).length === 0) {
        addStreamRow({ name: "Stream 1", url: "" }, false);
      }
      refreshActiveStreamOptions(activeIndex || 0);
    }

    function applySettingsToForm(s) {
      document.getElementById("rtsp_url").value = s.rtsp_url || "";
      document.getElementById("jpeg_quality").value = s.jpeg_quality;
      document.getElementById("jpeg_quality_num").value = s.jpeg_quality;
      document.getElementById("contrast").value = s.contrast;
      document.getElementById("contrast_num").value = s.contrast;
      document.getElementById("brightness").value = s.brightness;
      document.getElementById("brightness_num").value = s.brightness;
      document.getElementById("saturation").value = s.saturation;
      document.getElementById("saturation_num").value = s.saturation;
      document.getElementById("swap_rb").checked = !!s.swap_rb;
      setStreams(s.streams || [], s.active_stream_index || 0);
    }

    function updateState(st) {
      st = st || {};
      document.getElementById("meta_status").textContent = st.status || "-";
      document.getElementById("meta_source").textContent = st.source || "-";
      document.getElementById("meta_active_stream").textContent = st.active_stream_name || "-";
      document.getElementById("meta_error").textContent = st.last_error || "-";
      if (st.last_frame_ms) {
        document.getElementById("meta_frame").textContent = new Date(st.last_frame_ms).toLocaleTimeString();
      } else {
        document.getElementById("meta_frame").textContent = "-";
      }
    }

    function hydrate(data) {
      const s = data.settings;
      if (!isDirty && s) {
        applySettingsToForm(s);
      }
      updateState(data.state || {});
    }

    async function loadSettings() {
      const res = await fetch("/api/settings");
      const data = await res.json();
      isDirty = false;
      hydrate(data);
      populateFirmwareDefaultsFromSettings();
      setStatus("Settings loaded", "ok");
    }

    async function saveSettings() {
      const streams = getStreamRows();
      let activeIndex = Number(activeStreamEl.value || 0);
      if (streams.length === 0) {
        activeIndex = 0;
      } else if (activeIndex < 0 || activeIndex >= streams.length) {
        activeIndex = 0;
      }
      const payload = {
        rtsp_url: document.getElementById("rtsp_url").value.trim(),
        streams,
        active_stream_index: activeIndex,
        jpeg_quality: Number(document.getElementById("jpeg_quality_num").value),
        contrast: Number(document.getElementById("contrast_num").value),
        brightness: Number(document.getElementById("brightness_num").value),
        saturation: Number(document.getElementById("saturation_num").value),
        swap_rb: document.getElementById("swap_rb").checked
      };
      const res = await fetch("/api/settings", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) {
        setStatus((data.errors || ["Failed to apply settings"]).join(" | "), "err");
        return;
      }
      isDirty = false;
      hydrate({settings: data.settings, state: {}});
      if (data.warning) {
        setStatus("Applied with warning: " + data.warning, "warn");
      } else {
        setStatus("Applied. Stream will reconnect if URL changed.", "ok");
      }
      populateFirmwareDefaultsFromSettings();
    }

    async function applyPreset(kind) {
      const presets = {
        max_fps: { preset: "max_fps", jpeg_quality: 68, contrast: 1.06, brightness: -2, saturation: 1.08 },
        best_quality: { preset: "best_quality", jpeg_quality: 88, contrast: 1.18, brightness: -6, saturation: 1.24 }
      };
      const p = presets[kind];
      if (!p) return;
      const res = await fetch("/api/settings", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(p)
      });
      const data = await res.json();
      if (!res.ok) {
        setStatus((data.errors || ["Failed to apply preset"]).join(" | "), "err");
        return;
      }
      isDirty = false;
      hydrate({settings: data.settings, state: {}});
      setStatus(`Preset applied: ${kind}`, "ok");
    }

    document.getElementById("save_btn").addEventListener("click", saveSettings);
    document.getElementById("refresh_btn").addEventListener("click", loadSettings);
    document.getElementById("preset_fps_btn").addEventListener("click", () => applyPreset("max_fps"));
    document.getElementById("preset_quality_btn").addEventListener("click", () => applyPreset("best_quality"));
    document.getElementById("add_stream_btn").addEventListener("click", () => {
      addStreamRow({ name: `Stream ${document.querySelectorAll(".stream-row").length + 1}`, url: "" }, true);
      isDirty = true;
    });
    streamListEl.addEventListener("click", (e) => {
      if (e.target && e.target.classList.contains("remove-stream-btn")) {
        const row = e.target.closest(".stream-row");
        if (row) {
          row.remove();
          refreshActiveStreamOptions(activeStreamEl.value || 0);
          isDirty = true;
        }
      }
    });
    activeStreamEl.addEventListener("change", () => {
      const streams = getStreamRows();
      const idx = Number(activeStreamEl.value || 0);
      if (streams[idx]) {
        document.getElementById("rtsp_url").value = streams[idx].url;
      }
      isDirty = true;
    });
    document.getElementById("rtsp_url").addEventListener("input", () => {
      const idx = Number(activeStreamEl.value || 0);
      const rows = document.querySelectorAll(".stream-row");
      if (rows[idx]) {
        rows[idx].querySelector(".stream-url").value = document.getElementById("rtsp_url").value;
      }
      refreshActiveStreamOptions(idx);
      isDirty = true;
    });
    document.addEventListener("input", (e) => {
      if (!e.target) return;
      if (e.target.classList && (e.target.classList.contains("stream-name") || e.target.classList.contains("stream-url"))) {
        if (e.target.classList.contains("stream-url")) {
          const rows = Array.from(document.querySelectorAll(".stream-row"));
          const row = e.target.closest(".stream-row");
          const idx = rows.indexOf(row);
          if (idx === Number(activeStreamEl.value || 0)) {
            document.getElementById("rtsp_url").value = e.target.value;
          }
        }
        refreshActiveStreamOptions(activeStreamEl.value || 0);
      }
      isDirty = true;
    });
    document.querySelectorAll("input,select").forEach((el) => {
      el.addEventListener("change", () => { isDirty = true; });
    });

    function populateFirmwareDefaultsFromSettings() {
      const url = document.getElementById("rtsp_url").value || "";
      const m = url.match(/@([^:/]+)(?::\\d+)?\\//);
      document.getElementById("fw_server_host").value = (m && m[1]) ? m[1] : "192.168.1.72";
      document.getElementById("fw_server_port").value = 8124;
      document.getElementById("webflash_server_host").value = document.getElementById("fw_server_host").value;
      document.getElementById("webflash_server_port").value = document.getElementById("fw_server_port").value;
    }

    function mountWebflashButton(slotEl, manifestUrl) {
      slotEl.innerHTML = "";
      const btn = document.createElement("esp-web-install-button");
      btn.setAttribute("manifest", manifestUrl);
      slotEl.appendChild(btn);
    }

    async function prepareWebflash(flavor) {
      const payload = {
        flavor,
        ssid: document.getElementById("webflash_ssid").value.trim(),
        password: document.getElementById("webflash_password").value,
        server_host: document.getElementById("webflash_server_host").value.trim(),
        server_port: Number(document.getElementById("webflash_server_port").value)
      };
      setWebflashStatus("Preparing custom firmware...", "warn");
      const res = await fetch("/api/webflash/prepare", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        setWebflashStatus(data.error || "Failed to prepare browser flash package", "err");
        return;
      }
      const slot = flavor === "audio_on" ? webflashAudioSlot : webflashNoAudioSlot;
      mountWebflashButton(slot, data.manifest_url);
      setWebflashStatus("Ready. Click Install to choose USB port and flash.", "ok");
    }

    async function loadFlashStatus() {
      const res = await fetch("/api/flash/status");
      const data = await res.json();
      const supported = !!data.supported;
      const status = data.status || "idle";
      const running = !!data.running;
      const label = running ? `Running: ${status}` : `Last: ${status}`;
      setFlashStatus(label + (data.last_error ? ` | ${data.last_error}` : ""), running ? "warn" : (status === "success" ? "ok" : (status === "failed" ? "err" : "")));
      flashLog.textContent = (data.log_lines || []).join("\\n");
      flashLog.scrollTop = flashLog.scrollHeight;
      const ports = data.ports || [];
      flashPorts.innerHTML = `<div><strong>Detected USB Ports:</strong> ${ports.length ? ports.join(", ") : "none detected"}</div>`;
      if (!document.getElementById("fw_upload_port").value && ports.length === 1) {
        document.getElementById("fw_upload_port").value = ports[0];
      }
      document.getElementById("flash_btn").disabled = !supported || running;
      document.getElementById("refresh_flash_btn").disabled = false;
      ["fw_flavor","fw_upload_port","fw_ssid","fw_password","fw_server_host","fw_server_port"].forEach((id) => {
        document.getElementById(id).disabled = !supported || running;
      });
    }

    async function startFlash() {
      const payload = {
        flavor: document.getElementById("fw_flavor").value,
        ssid: document.getElementById("fw_ssid").value.trim(),
        password: document.getElementById("fw_password").value,
        server_host: document.getElementById("fw_server_host").value.trim(),
        server_port: Number(document.getElementById("fw_server_port").value),
        upload_port: document.getElementById("fw_upload_port").value.trim()
      };
      const res = await fetch("/api/flash", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) {
        setFlashStatus(data.error || "Failed to start flash job", "err");
        return;
      }
      setFlashStatus("Flash started", "warn");
      await loadFlashStatus();
    }

    document.getElementById("flash_btn").addEventListener("click", startFlash);
    document.getElementById("refresh_flash_btn").addEventListener("click", loadFlashStatus);
    document.getElementById("webflash_prepare_audio_btn").addEventListener("click", () => prepareWebflash("audio_on").catch(() => setWebflashStatus("Failed to prepare web flash", "err")));
    document.getElementById("webflash_prepare_no_audio_btn").addEventListener("click", () => prepareWebflash("no_audio").catch(() => setWebflashStatus("Failed to prepare web flash", "err")));

    loadSettings().catch(() => setStatus("Failed to load settings", "err"));
    populateFirmwareDefaultsFromSettings();
    loadFlashStatus().catch(() => setFlashStatus("Failed to load flash status", "err"));
    setInterval(() => {
      fetch("/api/settings")
        .then(r => r.json())
        .then(d => updateState(d.state))
        .catch(() => {});
    }, 1500);
    setInterval(() => {
      loadFlashStatus().catch(() => {});
    }, 2000);
  </script>
</body>
</html>"""
    )


if __name__ == '__main__':
    init_video_source()
    app.run(host='0.0.0.0', port=VIDEO_SERVER_PORT)
