import os
import subprocess
import threading
import time
import json
import secrets
import tempfile
import hashlib
import wave
from pathlib import Path
import urllib.request
import urllib.error

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request
from video_server.video_preprocessor import process_videos

app = Flask(__name__)

FRAME_SIZE = (320, 240)
VIDEO_SERVER_PORT = int(os.getenv("VIDEO_SERVER_PORT", "8123"))
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MOTION_VOICE_ENTITY = os.getenv("MOTION_VOICE_DEFAULT_ENTITY", "media_player.library_pair").strip() or "media_player.library_pair"


def _detect_app_version():
    env_version = os.getenv("APP_VERSION", "").strip()
    if env_version:
        return env_version
    try:
        git_sha = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short=8", "HEAD"],
            text=True,
        ).strip()
        if git_sha:
            return git_sha
    except Exception:
        pass
    hasher = hashlib.sha1()
    for candidate in (
        Path(__file__),
        REPO_ROOT / "server" / "requirements.txt",
        REPO_ROOT / "server" / "Dockerfile",
    ):
        try:
            hasher.update(candidate.read_bytes())
        except OSError:
            continue
    digest = hasher.hexdigest()[:8]
    return f"build-{digest}" if digest else "build-unknown"


APP_VERSION = _detect_app_version()


def _int_env(name: str, default: int, minimum=None, maximum=None) -> int:
    raw = os.getenv(name, "").strip()
    value = default
    if raw:
        try:
            value = int(raw)
        except Exception:
            value = default
    if minimum is not None and value < minimum:
        value = minimum
    if maximum is not None and value > maximum:
        value = maximum
    return value


def _float_env(name: str, default: float, minimum=None, maximum=None) -> float:
    raw = os.getenv(name, "").strip()
    value = default
    if raw:
        try:
            value = float(raw)
        except Exception:
            value = default
    if minimum is not None and value < minimum:
        value = minimum
    if maximum is not None and value > maximum:
        value = maximum
    return value


RTSP_OPEN_TIMEOUT_MS = _int_env("RTSP_OPEN_TIMEOUT_MS", 6000, minimum=500, maximum=120000)
RTSP_READ_TIMEOUT_MS = _int_env("RTSP_READ_TIMEOUT_MS", 6000, minimum=500, maximum=120000)
RTSP_STALE_RECONNECT_MS = _int_env("RTSP_STALE_RECONNECT_MS", 12000, minimum=2000, maximum=300000)
RTSP_CAPTURE_BUFFER_SIZE = _int_env("RTSP_CAPTURE_BUFFER_SIZE", 1, minimum=0, maximum=32)
RTSP_CAPTURE_WIDTH = _int_env("RTSP_CAPTURE_WIDTH", 0, minimum=0, maximum=4096)
RTSP_CAPTURE_HEIGHT = _int_env("RTSP_CAPTURE_HEIGHT", 0, minimum=0, maximum=4096)
RTSP_READ_RETRY_COUNT = _int_env("RTSP_READ_RETRY_COUNT", 12, minimum=1, maximum=120)
RTSP_FPS_MAX = _int_env("RTSP_FPS_MAX", 18, minimum=1, maximum=60)
RTSP_FPS_BALANCED = _int_env("RTSP_FPS_BALANCED", 12, minimum=1, maximum=60)
RTSP_FPS_BEST_QUALITY = _int_env("RTSP_FPS_BEST_QUALITY", 10, minimum=1, maximum=60)
RTSP_TRANSPORT_PROBE_TIMEOUT_MS = _int_env("RTSP_TRANSPORT_PROBE_TIMEOUT_MS", 6000, minimum=1000, maximum=30000)
MOTION_DETECT_WIDTH = _int_env("MOTION_DETECT_WIDTH", 160, minimum=64, maximum=640)
MOTION_DETECT_HEIGHT = _int_env("MOTION_DETECT_HEIGHT", 120, minimum=48, maximum=480)
MOTION_BG_HISTORY = _int_env("MOTION_BG_HISTORY", 90, minimum=10, maximum=2000)
MOTION_BG_VAR_THRESHOLD = _float_env("MOTION_BG_VAR_THRESHOLD", 32.0, minimum=4.0, maximum=255.0)
MOTION_BG_LEARNING_RATE = _float_env("MOTION_BG_LEARNING_RATE", 0.02, minimum=0.0, maximum=1.0)
MOTION_MIN_CONTOUR_AREA_PCT = _float_env("MOTION_MIN_CONTOUR_AREA_PCT", 0.12, minimum=0.001, maximum=20.0)
MOTION_WARMUP_FRAMES = _int_env("MOTION_WARMUP_FRAMES", 12, minimum=0, maximum=600)

if not os.getenv("OPENCV_FFMPEG_CAPTURE_OPTIONS"):
    ffmpeg_timeout_us = max(RTSP_READ_TIMEOUT_MS, RTSP_OPEN_TIMEOUT_MS) * 1000
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;tcp|stimeout;{ffmpeg_timeout_us}|rw_timeout;{ffmpeg_timeout_us}|"
        "max_delay;1000000|buffer_size;1048576"
    )

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
    "motion_enabled": os.getenv("MOTION_ENABLED", "1") == "1",
    "motion_threshold": float(os.getenv("MOTION_THRESHOLD", "2.5")),
    "motion_hold_ms": int(os.getenv("MOTION_HOLD_MS", "4000")),
    "motion_detect_width": MOTION_DETECT_WIDTH,
    "motion_detect_height": MOTION_DETECT_HEIGHT,
    "motion_bg_history": MOTION_BG_HISTORY,
    "motion_bg_var_threshold": MOTION_BG_VAR_THRESHOLD,
    "motion_bg_learning_rate": MOTION_BG_LEARNING_RATE,
    "motion_min_contour_area_pct": MOTION_MIN_CONTOUR_AREA_PCT,
    "motion_warmup_frames": MOTION_WARMUP_FRAMES,
    "motion_audio_enabled": os.getenv("MOTION_AUDIO_ENABLED", "1") == "1",
    "motion_voice_enabled": os.getenv("MOTION_VOICE_ENABLED", "0") == "1",
    "motion_voice_cooldown_ms": int(os.getenv("MOTION_VOICE_COOLDOWN_MS", "30000")),
    "motion_voice_daytime_only": os.getenv("MOTION_VOICE_DAYTIME_ONLY", "0") == "1",
    "motion_voice_start_hour": _int_env("MOTION_VOICE_START_HOUR", 8, minimum=0, maximum=23),
    "motion_voice_end_hour": _int_env("MOTION_VOICE_END_HOUR", 22, minimum=0, maximum=23),
    "motion_voice_message": os.getenv("MOTION_VOICE_MESSAGE", "Motion detected on {stream_name}"),
    "motion_voice_default_entity": DEFAULT_MOTION_VOICE_ENTITY,
    "ha_direct_tts_enabled": os.getenv("HA_DIRECT_TTS_ENABLED", "1") == "1",
    "ha_base_url": os.getenv("HA_BASE_URL", "").strip(),
    "ha_tts_entity": os.getenv("HA_TTS_ENTITY", "tts.google_translate_en_com").strip(),
    "ha_webhook_url": os.getenv("HA_WEBHOOK_URL", "").strip(),
    "ha_bearer_token": os.getenv("HA_BEARER_TOKEN", "").strip(),
    "ha_webhook_secret": os.getenv("HA_WEBHOOK_SECRET", "").strip(),
}
if settings["rtsp_url"]:
    settings["streams"] = [{"name": "Stream 1", "url": settings["rtsp_url"], "motion_voice": False, "voice_entity": ""}]
    settings["active_stream_index"] = 0

capture_state_lock = threading.Lock()
capture_open_lock = threading.Lock()
transport_probe_lock = threading.Lock()
transport_probe_cache = {}
capture_state = {
    "source": "",
    "status": "starting",
    "last_error": "",
    "last_frame_ms": 0,
}

client_sessions_lock = threading.Lock()
client_sessions = {}

video_data = []
stream_workers_lock = threading.Lock()
stream_workers = {}
preview_ffmpeg_workers_lock = threading.Lock()
preview_ffmpeg_workers = {}
motion_state_lock = threading.Lock()
motion_state = {
    "active": False,
    "last_motion_ms": 0,
    "last_motion_ratio": 0.0,
    "triggered_until_ms": 0,
    "source_stream_idx": 0,
}
motion_audio_lock = threading.Lock()
motion_audio_cache = {
    "path": "",
    "mtime_ms": 0,
    "data": b"",
}
motion_voice_lock = threading.Lock()
motion_voice_state = {
    "last_sent_ms": 0,
    "last_sent_stream_idx": -1,
    "last_sent_stream_name": "",
    "last_sent_speaker_entity": "",
    "last_error": "",
    "last_sent_by_stream": {},
}

PLAYER_DIR = (REPO_ROOT / "player").resolve()
LOCAL_OVERRIDES_PATH = PLAYER_DIR / "src" / "LocalOverrides.h"
SETTINGS_PATH = (Path(__file__).resolve().parent / "cache" / "settings.json").resolve()
STATIC_FIRMWARE_DIR = (Path(__file__).resolve().parent / "static" / "firmware").resolve()
MOTION_ALERT_WAV_PATH = (Path(__file__).resolve().parent / "cache" / "motion_alert.wav").resolve()
MOTION_ALERT_RAW_PATH = (Path(__file__).resolve().parent / "cache" / "motion_alert_u8_16k.raw").resolve()

WEBFLASH_SSID_TOKEN = b"CFG_WIFI_SSID_PLACEHOLDER_XXXXXXXXXXXXXXXX"
WEBFLASH_PASSWORD_TOKEN = b"CFG_WIFI_PASSWORD_PLACEHOLDER_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
WEBFLASH_HOST_TOKEN = b"CFG_VIDEO_SERVER_HOST_PLACEHOLDER_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
WEBFLASH_PORT_TOKEN = b"8124P"

FLASH_TARGETS = {
    "cyd": {
        "label": "Cheap Yellow Display (ESP32-2432S028R)",
        "chip": "esp32",
        "chip_family": "ESP32",
        "usb_env": {
            "audio_on": "cheap-yellow-display",
            "no_audio": "cheap-yellow-display-no-audio",
        },
        "webflash": {
            "audio_on": {
                "bootloader": "bootloader-audio.bin",
                "partitions": "partitions-audio.bin",
                "template": "template-firmware-audio.bin",
            },
            "no_audio": {
                "bootloader": "bootloader-no-audio.bin",
                "partitions": "partitions-no-audio.bin",
                "template": "template-firmware-no-audio.bin",
            },
        },
    },
    "ttgo_tdisplay": {
        "label": "TTGO T-Display",
        "chip": "esp32",
        "chip_family": "ESP32",
        "usb_env": {
            "audio_on": "tdisplay-wifi",
            "no_audio": "tdisplay-wifi-no-audio",
        },
        "webflash": {
            "audio_on": {
                "bootloader": "tdisplay-bootloader-audio.bin",
                "partitions": "tdisplay-partitions-audio.bin",
                "template": "tdisplay-template-firmware-audio.bin",
            },
            "no_audio": {
                "bootloader": "tdisplay-bootloader-no-audio.bin",
                "partitions": "tdisplay-partitions-no-audio.bin",
                "template": "tdisplay-template-firmware-no-audio.bin",
            },
        },
    },
    "esp32_s3_2p8": {
        "label": "ESP32-S3 2.8\" 240x320",
        "chip": "esp32s3",
        "chip_family": "ESP32-S3",
        "usb_env": {
            "audio_on": "esp32-s3-2p8-wifi",
            "no_audio": "esp32-s3-2p8-wifi-no-audio",
        },
        "webflash": {
            "audio_on": {
                "bootloader": "esp32s3-2p8-bootloader-audio.bin",
                "partitions": "esp32s3-2p8-partitions-audio.bin",
                "template": "esp32s3-2p8-template-firmware-audio.bin",
            },
            "no_audio": {
                "bootloader": "esp32s3-2p8-bootloader-no-audio.bin",
                "partitions": "esp32s3-2p8-partitions-no-audio.bin",
                "template": "esp32s3-2p8-template-firmware-no-audio.bin",
            },
        },
    },
}

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
        streams.append({
            "name": name[:64],
            "url": url[:2048],
            "motion_voice": bool(item.get("motion_voice", False)),
            "voice_entity": str(item.get("voice_entity", "")).strip()[:128],
        })
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


def _motion_detector_settings(cfg: dict) -> dict:
    out = {}
    try:
        out["motion_detect_width"] = max(64, min(640, int(cfg.get("motion_detect_width", MOTION_DETECT_WIDTH))))
    except Exception:
        out["motion_detect_width"] = MOTION_DETECT_WIDTH
    try:
        out["motion_detect_height"] = max(48, min(480, int(cfg.get("motion_detect_height", MOTION_DETECT_HEIGHT))))
    except Exception:
        out["motion_detect_height"] = MOTION_DETECT_HEIGHT
    try:
        out["motion_bg_history"] = max(10, min(2000, int(cfg.get("motion_bg_history", MOTION_BG_HISTORY))))
    except Exception:
        out["motion_bg_history"] = MOTION_BG_HISTORY
    try:
        out["motion_bg_var_threshold"] = max(4.0, min(255.0, float(cfg.get("motion_bg_var_threshold", MOTION_BG_VAR_THRESHOLD))))
    except Exception:
        out["motion_bg_var_threshold"] = MOTION_BG_VAR_THRESHOLD
    try:
        out["motion_bg_learning_rate"] = max(0.0, min(1.0, float(cfg.get("motion_bg_learning_rate", MOTION_BG_LEARNING_RATE))))
    except Exception:
        out["motion_bg_learning_rate"] = MOTION_BG_LEARNING_RATE
    try:
        out["motion_min_contour_area_pct"] = max(0.001, min(20.0, float(cfg.get("motion_min_contour_area_pct", MOTION_MIN_CONTOUR_AREA_PCT))))
    except Exception:
        out["motion_min_contour_area_pct"] = MOTION_MIN_CONTOUR_AREA_PCT
    try:
        out["motion_warmup_frames"] = max(0, min(600, int(cfg.get("motion_warmup_frames", MOTION_WARMUP_FRAMES))))
    except Exception:
        out["motion_warmup_frames"] = MOTION_WARMUP_FRAMES
    return out


def _motion_detector_signature(motion_cfg: dict):
    return (
        int(motion_cfg["motion_detect_width"]),
        int(motion_cfg["motion_detect_height"]),
        int(motion_cfg["motion_bg_history"]),
        float(motion_cfg["motion_bg_var_threshold"]),
        float(motion_cfg["motion_bg_learning_rate"]),
        float(motion_cfg["motion_min_contour_area_pct"]),
        int(motion_cfg["motion_warmup_frames"]),
    )


def _settings_for_response():
    cfg = _get_settings()
    streams = _normalize_streams(cfg.get("streams", []))
    idx = int(cfg.get("active_stream_index", 0))
    if len(streams) == 0 and cfg.get("rtsp_url", "").strip():
        streams = [{"name": "Stream 1", "url": cfg["rtsp_url"].strip(), "motion_voice": False, "voice_entity": ""}]
        idx = 0
    if len(streams) == 0:
        idx = 0
    elif idx < 0 or idx >= len(streams):
        idx = 0
    cfg["streams"] = streams
    cfg["active_stream_index"] = idx
    cfg["rtsp_url"] = streams[idx]["url"] if streams else ""
    cfg["ha_base_url"] = _normalize_ha_base_url(cfg.get("ha_base_url", ""))
    cfg.update(_motion_detector_settings(cfg))
    if not str(cfg.get("motion_voice_default_entity", "")).strip():
        cfg["motion_voice_default_entity"] = DEFAULT_MOTION_VOICE_ENTITY
    cfg["motion_voice_daytime_only"] = bool(cfg.get("motion_voice_daytime_only", False))
    try:
        cfg["motion_voice_start_hour"] = max(0, min(23, int(cfg.get("motion_voice_start_hour", 8))))
    except Exception:
        cfg["motion_voice_start_hour"] = 8
    try:
        cfg["motion_voice_end_hour"] = max(0, min(23, int(cfg.get("motion_voice_end_hour", 22))))
    except Exception:
        cfg["motion_voice_end_hour"] = 22
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
        if "motion_enabled" in loaded:
            settings["motion_enabled"] = bool(loaded["motion_enabled"])
        if "motion_threshold" in loaded:
            try:
                settings["motion_threshold"] = float(loaded["motion_threshold"])
            except Exception:
                pass
        if "motion_hold_ms" in loaded:
            try:
                settings["motion_hold_ms"] = int(loaded["motion_hold_ms"])
            except Exception:
                pass
        if "motion_detect_width" in loaded:
            try:
                settings["motion_detect_width"] = max(64, min(640, int(loaded["motion_detect_width"])))
            except Exception:
                pass
        if "motion_detect_height" in loaded:
            try:
                settings["motion_detect_height"] = max(48, min(480, int(loaded["motion_detect_height"])))
            except Exception:
                pass
        if "motion_bg_history" in loaded:
            try:
                settings["motion_bg_history"] = max(10, min(2000, int(loaded["motion_bg_history"])))
            except Exception:
                pass
        if "motion_bg_var_threshold" in loaded:
            try:
                settings["motion_bg_var_threshold"] = max(4.0, min(255.0, float(loaded["motion_bg_var_threshold"])))
            except Exception:
                pass
        if "motion_bg_learning_rate" in loaded:
            try:
                settings["motion_bg_learning_rate"] = max(0.0, min(1.0, float(loaded["motion_bg_learning_rate"])))
            except Exception:
                pass
        if "motion_min_contour_area_pct" in loaded:
            try:
                settings["motion_min_contour_area_pct"] = max(0.001, min(20.0, float(loaded["motion_min_contour_area_pct"])))
            except Exception:
                pass
        if "motion_warmup_frames" in loaded:
            try:
                settings["motion_warmup_frames"] = max(0, min(600, int(loaded["motion_warmup_frames"])))
            except Exception:
                pass
        if "motion_audio_enabled" in loaded:
            settings["motion_audio_enabled"] = bool(loaded["motion_audio_enabled"])
        if "motion_voice_enabled" in loaded:
            settings["motion_voice_enabled"] = bool(loaded["motion_voice_enabled"])
        if "motion_voice_cooldown_ms" in loaded:
            try:
                value = int(loaded["motion_voice_cooldown_ms"])
                settings["motion_voice_cooldown_ms"] = max(1000, min(3_600_000, value))
            except Exception:
                pass
        if "motion_voice_daytime_only" in loaded:
            settings["motion_voice_daytime_only"] = bool(loaded["motion_voice_daytime_only"])
        if "motion_voice_start_hour" in loaded:
            try:
                settings["motion_voice_start_hour"] = max(0, min(23, int(loaded["motion_voice_start_hour"])))
            except Exception:
                pass
        if "motion_voice_end_hour" in loaded:
            try:
                settings["motion_voice_end_hour"] = max(0, min(23, int(loaded["motion_voice_end_hour"])))
            except Exception:
                pass
        if "motion_voice_message" in loaded:
            settings["motion_voice_message"] = str(loaded["motion_voice_message"])[:240]
        if "motion_voice_default_entity" in loaded:
            settings["motion_voice_default_entity"] = str(loaded["motion_voice_default_entity"]).strip()[:128]
        if "ha_direct_tts_enabled" in loaded:
            settings["ha_direct_tts_enabled"] = bool(loaded["ha_direct_tts_enabled"])
        if "ha_base_url" in loaded:
            settings["ha_base_url"] = str(loaded["ha_base_url"]).strip()[:2048]
        if "ha_tts_entity" in loaded:
            settings["ha_tts_entity"] = str(loaded["ha_tts_entity"]).strip()[:128]
        if "ha_webhook_url" in loaded:
            settings["ha_webhook_url"] = str(loaded["ha_webhook_url"]).strip()[:2048]
        if "ha_bearer_token" in loaded:
            settings["ha_bearer_token"] = str(loaded["ha_bearer_token"]).strip()[:1024]
        if "ha_webhook_secret" in loaded:
            settings["ha_webhook_secret"] = str(loaded["ha_webhook_secret"]).strip()[:1024]
        if not settings.get("motion_voice_default_entity"):
            settings["motion_voice_default_entity"] = DEFAULT_MOTION_VOICE_ENTITY
        if len(settings.get("streams", [])) == 0 and settings["rtsp_url"]:
            settings["streams"] = [{"name": "Stream 1", "url": settings["rtsp_url"], "motion_voice": False, "voice_entity": ""}]
            settings["active_stream_index"] = 0
        _sync_rtsp_url_locked()


def _sanitize_client_id(raw_value):
    raw = str(raw_value or "").strip()
    if not raw:
        return "default"
    clean = "".join(ch for ch in raw if ch.isalnum() or ch in ("-", "_", ".", ":"))
    return (clean[:64] or "default")


def _get_request_client_id():
    return _sanitize_client_id(
        request.args.get("cid")
        or request.args.get("client_id")
        or request.headers.get("X-Client-Id", "")
    )


def _cleanup_client_sessions(keep_client_id=None):
    now = _now_ms()
    idle_ms = 1_800_000
    with client_sessions_lock:
        stale = []
        for client_id, session in client_sessions.items():
            if keep_client_id is not None and client_id == keep_client_id:
                continue
            if now - int(session.get("last_access_ms", 0)) > idle_ms:
                stale.append(client_id)
        for client_id in stale:
            del client_sessions[client_id]


def _get_client_active_stream_index(client_id, stream_count):
    default_idx = int(_settings_for_response().get("active_stream_index", 0))
    now = _now_ms()
    with client_sessions_lock:
        session = client_sessions.get(client_id)
        if session is None:
            session = {
                "active_stream_index": default_idx,
                "last_access_ms": now,
            }
            client_sessions[client_id] = session
        idx = int(session.get("active_stream_index", default_idx))
        if stream_count <= 0:
            idx = 0
        elif idx < 0 or idx >= stream_count:
            idx = 0
        session["active_stream_index"] = idx
        session["last_access_ms"] = now
        return idx


def _set_client_active_stream_index(client_id, idx, stream_count):
    now = _now_ms()
    if stream_count <= 0:
        idx = 0
    elif idx < 0 or idx >= stream_count:
        idx = 0
    with client_sessions_lock:
        client_sessions[client_id] = {
            "active_stream_index": idx,
            "last_access_ms": now,
        }
    return idx


def _set_active_stream_from_channel(channel_index, client_id):
    cfg = _settings_for_response()
    streams = cfg.get("streams", [])
    if len(streams) == 0:
        _set_client_active_stream_index(client_id, 0, 0)
        return 0
    idx = channel_index % len(streams)
    _set_client_active_stream_index(client_id, idx, len(streams))
    return idx


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

    env_name = FLASH_TARGETS[job["board"]]["usb_env"][job["flavor"]]
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
            "board": job["board"],
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


def _build_motion_detector(motion_cfg: dict):
    return cv2.createBackgroundSubtractorMOG2(
        history=int(motion_cfg["motion_bg_history"]),
        varThreshold=float(motion_cfg["motion_bg_var_threshold"]),
        detectShadows=False,
    )


def _motion_detection_frame(frame, motion_cfg: dict):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    src_h, src_w = gray.shape[:2]
    target_w = int(motion_cfg["motion_detect_width"])
    target_h = int(motion_cfg["motion_detect_height"])
    if src_w == 0 or src_h == 0:
        return np.zeros((target_h, target_w), dtype=np.uint8)
    interpolation = cv2.INTER_AREA if src_w > target_w or src_h > target_h else cv2.INTER_LINEAR
    gray = cv2.resize(gray, (target_w, target_h), interpolation=interpolation)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _measure_motion_ratio(frame, motion_detector, detector_frames_seen: int, motion_cfg: dict) -> float:
    if motion_detector is None:
        return 0.0
    detect_frame = _motion_detection_frame(frame, motion_cfg)
    mask = motion_detector.apply(detect_frame, learningRate=float(motion_cfg["motion_bg_learning_rate"]))
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    frame_area = float(mask.shape[0] * mask.shape[1]) if mask.size else 1.0
    min_contour_area = max(12.0, frame_area * (float(motion_cfg["motion_min_contour_area_pct"]) / 100.0))
    motion_area = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_contour_area:
            continue
        motion_area += area
    if detector_frames_seen < int(motion_cfg["motion_warmup_frames"]):
        return 0.0
    return (motion_area * 100.0) / frame_area


def _set_motion_voice_error(message: str):
    with motion_voice_lock:
        motion_voice_state["last_error"] = str(message or "")[:240]


def _normalize_ha_base_url(url: str) -> str:
    base = str(url or "").strip()
    if not base:
        return ""
    # Tolerate accidental duplicated schemes from copy/paste (e.g. http://http://host:8123).
    normalized = True
    while normalized:
        normalized = False
        if base.startswith("http://http://"):
            base = "http://" + base[len("http://http://"):]
            normalized = True
        elif base.startswith("https://https://"):
            base = "https://" + base[len("https://https://"):]
            normalized = True
        elif base.startswith("http://https://"):
            base = "https://" + base[len("http://https://"):]
            normalized = True
        elif base.startswith("https://http://"):
            base = "http://" + base[len("https://http://"):]
            normalized = True
    if base.endswith("/"):
        base = base[:-1]
    return base


def _format_motion_voice_message(template: str, stream_name: str, stream_idx: int, ratio: float) -> str:
    tmpl = str(template or "").strip() or "Motion detected on {stream_name}"
    try:
        rendered = tmpl.format(
            stream_name=stream_name,
            stream_index=int(stream_idx) + 1,
            motion_ratio=f"{float(ratio):.2f}",
        )
        return rendered[:240]
    except Exception:
        return f"Motion detected on {stream_name}"[:240]


def _post_motion_voice_alert(url: str, token: str, secret: str, payload: dict):
    headers = {"Content-Type": "application/json", "User-Agent": "esp32-tv-server/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if secret:
        headers["X-Webhook-Secret"] = secret
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            code = int(getattr(resp, "status", 0) or resp.getcode())
            if code >= 300:
                _set_motion_voice_error(f"HA webhook HTTP {code}")
    except urllib.error.HTTPError as ex:
        _set_motion_voice_error(f"HA webhook HTTP {ex.code}")
    except Exception as ex:
        _set_motion_voice_error(f"HA webhook failed: {ex}")


def _post_motion_voice_tts(base_url: str, token: str, tts_entity: str, speaker_entity: str, message: str):
    if not base_url:
        _set_motion_voice_error("HA base URL missing")
        return
    if not token:
        _set_motion_voice_error("HA bearer token missing")
        return
    if not tts_entity:
        _set_motion_voice_error("HA TTS entity missing")
        return
    if not speaker_entity:
        _set_motion_voice_error("Speaker entity missing")
        return
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "esp32-tv-server/1.0",
    }
    base = _normalize_ha_base_url(base_url)

    # Wake grouped/cast players before TTS.
    try:
        wake_url = f"{base}/api/services/media_player/turn_on"
        wake_payload = {"entity_id": speaker_entity}
        wake_req = urllib.request.Request(
            url=wake_url,
            data=json.dumps(wake_payload, ensure_ascii=True).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(wake_req, timeout=4):
            pass
    except Exception:
        # Don't block TTS on wake-up failures.
        pass

    entity = str(tts_entity).strip().lower()
    legacy_service = ""
    if "google_translate" in entity:
        legacy_service = "google_translate_say"
    elif "cloud" in entity:
        legacy_service = "cloud_say"

    if legacy_service:
        url = f"{base}/api/services/tts/{legacy_service}"
        payload = {
            "entity_id": speaker_entity,
            "message": message,
            "cache": False,
        }
        service_label = legacy_service
    else:
        url = f"{base}/api/services/tts/speak"
        payload = {
            "entity_id": tts_entity,
            "media_player_entity_id": speaker_entity,
            "message": message,
        }
        service_label = "speak"

    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            code = int(getattr(resp, "status", 0) or resp.getcode())
            if code >= 300:
                _set_motion_voice_error(f"HA TTS {service_label} HTTP {code}")
    except urllib.error.HTTPError as ex:
        _set_motion_voice_error(f"HA TTS {service_label} HTTP {ex.code}")
    except Exception as ex:
        _set_motion_voice_error(f"HA TTS {service_label} failed: {ex}")


def _is_motion_voice_enabled_for_stream(cfg: dict, stream_idx: int) -> bool:
    streams = cfg.get("streams", [])
    if stream_idx < 0 or stream_idx >= len(streams):
        return False
    stream = streams[stream_idx]
    return bool(stream.get("motion_voice", False))


def _resolve_motion_voice_entity(cfg: dict, stream_idx: int) -> str:
    streams = cfg.get("streams", [])
    stream_entity = ""
    if stream_idx >= 0 and stream_idx < len(streams):
        stream_entity = str(streams[stream_idx].get("voice_entity", "")).strip()
    if stream_entity:
        return stream_entity[:128]
    fallback_entity = str(cfg.get("motion_voice_default_entity", "")).strip() or DEFAULT_MOTION_VOICE_ENTITY
    return fallback_entity[:128]


def _is_hour_within_window(hour: int, start_hour: int, end_hour: int) -> bool:
    if start_hour == end_hour:
        return True
    if start_hour < end_hour:
        return start_hour <= hour < end_hour
    return hour >= start_hour or hour < end_hour


def _is_motion_voice_time_allowed(cfg: dict) -> bool:
    if not bool(cfg.get("motion_voice_daytime_only", False)):
        return True
    try:
        start_hour = max(0, min(23, int(cfg.get("motion_voice_start_hour", 8))))
    except Exception:
        start_hour = 8
    try:
        end_hour = max(0, min(23, int(cfg.get("motion_voice_end_hour", 22))))
    except Exception:
        end_hour = 22
    local_hour = int(time.localtime().tm_hour)
    return _is_hour_within_window(local_hour, start_hour, end_hour)


def _maybe_send_motion_voice_alert(stream_idx: int, ratio: float, detected_ms: int):
    cfg = _settings_for_response()
    if not cfg.get("motion_enabled", False):
        return
    if not cfg.get("motion_voice_enabled", False):
        return
    if not _is_motion_voice_enabled_for_stream(cfg, int(stream_idx)):
        return
    if not _is_motion_voice_time_allowed(cfg):
        return
    use_direct_tts = bool(cfg.get("ha_direct_tts_enabled", True))
    base_url = _normalize_ha_base_url(str(cfg.get("ha_base_url", "")))
    webhook_url = str(cfg.get("ha_webhook_url", "")).strip()
    if use_direct_tts and not base_url:
        if not webhook_url:
            return
        use_direct_tts = False
    if not use_direct_tts and not webhook_url:
        return
    token = str(cfg.get("ha_bearer_token", "")).strip()
    secret = str(cfg.get("ha_webhook_secret", "")).strip()
    tts_entity = str(cfg.get("ha_tts_entity", "")).strip()
    cooldown_ms = int(cfg.get("motion_voice_cooldown_ms", 30000))
    now = _now_ms()
    stream = cfg["streams"][int(stream_idx)]
    stream_name = str(stream.get("name", f"Stream {int(stream_idx) + 1}"))
    speaker_entity = _resolve_motion_voice_entity(cfg, int(stream_idx))
    with motion_voice_lock:
        by_stream = motion_voice_state.get("last_sent_by_stream", {})
        key = str(int(stream_idx))
        last_sent_ms = int(by_stream.get(key, 0))
        if now - last_sent_ms < cooldown_ms:
            return
        by_stream[key] = now
        motion_voice_state["last_sent_by_stream"] = by_stream
        motion_voice_state["last_sent_ms"] = now
        motion_voice_state["last_sent_stream_idx"] = int(stream_idx)
        motion_voice_state["last_sent_stream_name"] = stream_name
        motion_voice_state["last_sent_speaker_entity"] = speaker_entity
        motion_voice_state["last_error"] = ""
    payload = {
        "event": "motion_detected",
        "detected_ms": int(detected_ms),
        "stream_index": int(stream_idx),
        "stream_name": stream_name,
        "stream_url": str(stream.get("url", "")),
        "speaker_entity": speaker_entity,
        "motion_ratio": float(ratio),
        "message": _format_motion_voice_message(
            cfg.get("motion_voice_message", ""),
            stream_name,
            int(stream_idx),
            float(ratio),
        ),
    }
    if use_direct_tts:
        threading.Thread(
            target=_post_motion_voice_tts,
            args=(base_url, token, tts_entity, speaker_entity, payload["message"]),
            daemon=True,
        ).start()
    else:
        threading.Thread(
            target=_post_motion_voice_alert,
            args=(webhook_url, token, secret, payload),
            daemon=True,
        ).start()


def _set_motion_trigger(stream_idx: int, ratio: float, hold_ms: int):
    now = _now_ms()
    should_alert_voice = False
    with motion_state_lock:
        was_active_same_stream = bool(
            motion_state.get("active", False)
            and now < int(motion_state.get("triggered_until_ms", 0))
            and int(motion_state.get("source_stream_idx", -1)) == int(stream_idx)
        )
        motion_state["active"] = True
        motion_state["last_motion_ms"] = now
        motion_state["last_motion_ratio"] = float(ratio)
        motion_state["triggered_until_ms"] = max(now + int(hold_ms), int(motion_state.get("triggered_until_ms", 0)))
        motion_state["source_stream_idx"] = int(stream_idx)
        should_alert_voice = not was_active_same_stream
    if should_alert_voice:
        _maybe_send_motion_voice_alert(int(stream_idx), float(ratio), int(now))


def _is_motion_active_for_stream(stream_idx: int):
    now = _now_ms()
    with motion_state_lock:
        if now >= int(motion_state.get("triggered_until_ms", 0)):
            motion_state["active"] = False
            return False
        return motion_state.get("active", False) and int(motion_state.get("source_stream_idx", 0)) == int(stream_idx)


def _decode_wav_to_u8_16k(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        in_rate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    if sample_width not in (1, 2):
        raise ValueError("motion alert wav must be 8-bit or 16-bit PCM")
    if sample_width == 1:
        pcm = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
    else:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    if in_rate != 16000 and len(pcm) > 1:
        x_in = np.linspace(0.0, 1.0, num=len(pcm), endpoint=False)
        out_len = max(1, int(len(pcm) * (16000.0 / float(in_rate))))
        x_out = np.linspace(0.0, 1.0, num=out_len, endpoint=False)
        pcm = np.interp(x_out, x_in, pcm).astype(np.float32)
    peak = float(np.max(np.abs(pcm))) if len(pcm) else 0.0
    if peak > 0:
        pcm = (pcm / peak) * 100.0
    u8 = np.clip(pcm + 128.0, 0, 255).astype(np.uint8)
    return u8.tobytes()


def _get_motion_alert_audio():
    candidate = None
    if MOTION_ALERT_RAW_PATH.exists():
        candidate = MOTION_ALERT_RAW_PATH
    elif MOTION_ALERT_WAV_PATH.exists():
        candidate = MOTION_ALERT_WAV_PATH
    if candidate is None:
        t = np.linspace(0, 0.15, int(16000 * 0.15), endpoint=False)
        return (np.sin(2 * np.pi * 1200 * t) * 60 + 128).astype(np.uint8).tobytes()
    mtime_ms = int(candidate.stat().st_mtime * 1000)
    with motion_audio_lock:
        if (
            motion_audio_cache["path"] == str(candidate)
            and motion_audio_cache["mtime_ms"] == mtime_ms
            and motion_audio_cache["data"]
        ):
            return motion_audio_cache["data"]
        data = _decode_wav_to_u8_16k(candidate) if candidate.suffix.lower() == ".wav" else candidate.read_bytes()
        motion_audio_cache["path"] = str(candidate)
        motion_audio_cache["mtime_ms"] = mtime_ms
        motion_audio_cache["data"] = data
        return data


def _webflash_templates_exist(board: str):
    if board not in FLASH_TARGETS:
        return False
    required = []
    for flavor in ("audio_on", "no_audio"):
        cfg = FLASH_TARGETS[board]["webflash"][flavor]
        required.append(STATIC_FIRMWARE_DIR / cfg["bootloader"])
        required.append(STATIC_FIRMWARE_DIR / cfg["partitions"])
        required.append(STATIC_FIRMWARE_DIR / cfg["template"])
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


def _build_custom_webflash_firmware(board: str, flavor: str, ssid: str, password: str, server_host: str, server_port: int) -> bytes:
    template_name = FLASH_TARGETS[board]["webflash"][flavor]["template"]
    chip = str(FLASH_TARGETS.get(board, {}).get("chip", "esp32")).strip() or "esp32"
    blob = (STATIC_FIRMWARE_DIR / template_name).read_bytes()
    blob = _patch_token(blob, WEBFLASH_SSID_TOKEN, ssid, 32)
    blob = _patch_token(blob, WEBFLASH_PASSWORD_TOKEN, password, 63)
    blob = _patch_token(blob, WEBFLASH_HOST_TOKEN, server_host, 63)
    blob = _patch_token(blob, WEBFLASH_PORT_TOKEN, str(server_port), 4)
    # Recalculate ESP image footer after binary patching.
    try:
        from esptool.bin_image import LoadFirmwareImage
        image = LoadFirmwareImage(chip, blob)
        for i, segment in enumerate(image.segments):
            if not hasattr(segment, "name"):
                segment.name = f"segment{i}"
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            image.save(tmp_path)
            blob = Path(tmp_path).read_bytes()
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass
    except Exception as ex:
        raise RuntimeError(f"failed to rebuild esp image footer ({chip}): {ex}")
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


def _cleanup_stream_workers(keep_idx=None):
    now = _now_ms()
    idle_ms = 120_000
    with stream_workers_lock:
        stale = []
        for idx, worker in stream_workers.items():
            if keep_idx is not None and idx == keep_idx:
                continue
            if now - int(worker.get("last_access_ms", 0)) > idle_ms:
                stale.append(idx)
        for idx in stale:
            stream_workers[idx]["running"] = False
            del stream_workers[idx]


def _stream_target_fps(preset: str) -> int:
    if preset == "best_quality":
        return RTSP_FPS_BEST_QUALITY
    if preset == "max_fps":
        return RTSP_FPS_MAX
    return RTSP_FPS_BALANCED


def _set_capture_prop(cap, prop_name: str, value: int):
    prop = getattr(cv2, prop_name, None)
    if prop is None:
        return
    try:
        cap.set(prop, float(value))
    except Exception:
        pass


def _configure_capture(cap):
    _set_capture_prop(cap, "CAP_PROP_OPEN_TIMEOUT_MSEC", RTSP_OPEN_TIMEOUT_MS)
    _set_capture_prop(cap, "CAP_PROP_READ_TIMEOUT_MSEC", RTSP_READ_TIMEOUT_MS)
    if RTSP_CAPTURE_BUFFER_SIZE > 0:
        _set_capture_prop(cap, "CAP_PROP_BUFFERSIZE", RTSP_CAPTURE_BUFFER_SIZE)
    if RTSP_CAPTURE_WIDTH > 0:
        _set_capture_prop(cap, "CAP_PROP_FRAME_WIDTH", RTSP_CAPTURE_WIDTH)
    if RTSP_CAPTURE_HEIGHT > 0:
        _set_capture_prop(cap, "CAP_PROP_FRAME_HEIGHT", RTSP_CAPTURE_HEIGHT)


def _capture_options_for_transport(transport: str) -> str:
    ffmpeg_timeout_us = max(RTSP_READ_TIMEOUT_MS, RTSP_OPEN_TIMEOUT_MS) * 1000
    mode = "udp" if transport == "udp" else "tcp"
    return (
        f"rtsp_transport;{mode}|stimeout;{ffmpeg_timeout_us}|rw_timeout;{ffmpeg_timeout_us}|"
        "max_delay;1000000|buffer_size;1048576"
    )


def _probe_transport_command(url: str, transport: str) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-rtsp_transport",
        transport,
        "-i",
        url,
        "-an",
        "-frames:v",
        "1",
        "-vf",
        "scale=160:120:force_original_aspect_ratio=decrease,pad=160:120:(ow-iw)/2:(oh-ih)/2:color=black",
        "-q:v",
        "8",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]


def _probe_rtsp_transport(url: str) -> str:
    url = str(url or "").strip()
    if not url:
        return "tcp"
    with transport_probe_lock:
        cached = transport_probe_cache.get(url)
    if cached in ("tcp", "udp"):
        return cached
    selected = "tcp"
    timeout_s = max(1.0, RTSP_TRANSPORT_PROBE_TIMEOUT_MS / 1000.0)
    for candidate in ("tcp", "udp"):
        try:
            proc = subprocess.run(
                _probe_transport_command(url, candidate),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0 and proc.stdout:
            selected = candidate
            break
    with transport_probe_lock:
        transport_probe_cache[url] = selected
    return selected


def _open_capture_with_transport(cap, url: str, transport: str):
    previous = os.getenv("OPENCV_FFMPEG_CAPTURE_OPTIONS")
    options = _capture_options_for_transport(transport)
    with capture_open_lock:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options
        try:
            cap.open(url)
        finally:
            if previous is None:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = previous


def _preview_ffmpeg_command(url: str, transport: str) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-rtsp_transport",
        transport,
        "-i",
        url,
        "-an",
        "-vf",
        "fps=6,scale=320:240:force_original_aspect_ratio=decrease,pad=320:240:(ow-iw)/2:(oh-ih)/2:color=black",
        "-q:v",
        "7",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]


def _cleanup_preview_ffmpeg_workers():
    now = _now_ms()
    idle_ms = 120_000
    with preview_ffmpeg_workers_lock:
        stale = []
        for idx, worker in preview_ffmpeg_workers.items():
            if now - int(worker.get("last_access_ms", 0)) > idle_ms:
                stale.append(idx)
        for idx in stale:
            preview_ffmpeg_workers[idx]["running"] = False
            del preview_ffmpeg_workers[idx]


def _preview_ffmpeg_capture_worker(worker):
    while worker["running"]:
        url = worker["url"]
        transport = str(worker.get("transport", "") or "").strip().lower()
        if transport not in ("tcp", "udp"):
            transport = _probe_rtsp_transport(url)
            worker["transport"] = transport
        try:
            proc = subprocess.Popen(
                _preview_ffmpeg_command(url, transport),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except OSError:
            worker["last_error"] = f"ffmpeg preview failed to start ({transport})"
            time.sleep(1.0)
            continue
        stdout = proc.stdout
        if stdout is None:
            proc.kill()
            worker["last_error"] = "ffmpeg preview has no stdout"
            time.sleep(1.0)
            continue
        worker["last_error"] = ""
        worker["status"] = "streaming"
        buffer = bytearray()
        try:
            while worker["running"]:
                chunk = stdout.read(8192)
                if not chunk:
                    break
                buffer.extend(chunk)
                while True:
                    start = buffer.find(b"\xff\xd8")
                    if start < 0:
                        if len(buffer) > 2:
                            del buffer[:-2]
                        break
                    end = buffer.find(b"\xff\xd9", start + 2)
                    if end < 0:
                        if start > 0:
                            del buffer[:start]
                        break
                    jpeg = bytes(buffer[start:end + 2])
                    del buffer[:end + 2]
                    worker["latest_jpeg"] = jpeg
                    worker["last_frame_ms"] = _now_ms()
        finally:
            try:
                stdout.close()
            except Exception:
                pass
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=0.5)
            except Exception:
                pass
        worker["status"] = "reconnecting"
        if worker["running"]:
            time.sleep(0.5)


def _get_or_start_preview_ffmpeg_worker(stream_idx: int, stream_url: str):
    with preview_ffmpeg_workers_lock:
        worker = preview_ffmpeg_workers.get(stream_idx)
        if worker is not None and worker.get("url") != stream_url:
            worker["running"] = False
            del preview_ffmpeg_workers[stream_idx]
            worker = None
        if worker is None:
            worker = {
                "idx": stream_idx,
                "url": stream_url,
                "transport": "",
                "running": True,
                "latest_jpeg": _encode_black_frame(),
                "last_frame_ms": 0,
                "last_access_ms": _now_ms(),
                "status": "starting",
                "last_error": "",
            }
            preview_ffmpeg_workers[stream_idx] = worker
            threading.Thread(target=_preview_ffmpeg_capture_worker, args=(worker,), daemon=True).start()
        worker["last_access_ms"] = _now_ms()
        return worker


def _stream_capture_worker(worker):
    while worker["running"]:
        url = worker["url"]
        if not url:
            worker["status"] = "idle"
            worker["last_error"] = ""
            time.sleep(0.3)
            continue
        transport = str(worker.get("transport", "") or "").strip().lower()
        if transport not in ("tcp", "udp"):
            transport = _probe_rtsp_transport(url)
            worker["transport"] = transport
        worker["status"] = "connecting"
        worker["last_error"] = ""
        cap = cv2.VideoCapture()
        _set_capture_prop(cap, "CAP_PROP_OPEN_TIMEOUT_MSEC", RTSP_OPEN_TIMEOUT_MS)
        _open_capture_with_transport(cap, url, transport)
        _configure_capture(cap)
        if not cap.isOpened():
            worker["status"] = "error"
            worker["last_error"] = f"failed to open RTSP stream ({transport})"
            time.sleep(1.0)
            continue
        worker["status"] = "streaming"
        last_emit = 0.0
        motion_detector = None
        motion_detector_sig = None
        detector_frames_seen = 0
        last_read_ok_ms = _now_ms()
        consecutive_read_failures = 0
        while worker["running"]:
            now_ms = _now_ms()
            if now_ms - last_read_ok_ms > RTSP_STALE_RECONNECT_MS:
                worker["status"] = "reconnecting"
                worker["last_error"] = f"stream stalled for {RTSP_STALE_RECONNECT_MS}ms"
                break
            ok, frame = cap.read()
            if not ok:
                consecutive_read_failures += 1
                if consecutive_read_failures < RTSP_READ_RETRY_COUNT:
                    # Briefly tolerate dropped reads so we don't reconnect on transient RTSP jitter.
                    time.sleep(0.03)
                    continue
                worker["status"] = "reconnecting"
                worker["last_error"] = f"stream read failed ({consecutive_read_failures} retries)"
                break
            consecutive_read_failures = 0
            last_read_ok_ms = _now_ms()
            cfg = _get_settings()
            preset = cfg.get("preset", "balanced")
            target_fps = _stream_target_fps(preset)
            now = time.time()
            if target_fps > 0 and last_emit > 0 and (now - last_emit) < (1.0 / target_fps):
                continue
            source_frame = frame
            preview_frame = _letterbox_frame_for_size(source_frame, FRAME_SIZE)
            frame = _fit_frame(source_frame)
            if cfg.get("motion_enabled", False):
                motion_cfg = _motion_detector_settings(cfg)
                next_motion_sig = _motion_detector_signature(motion_cfg)
                if motion_detector is None or motion_detector_sig != next_motion_sig:
                    motion_detector = _build_motion_detector(motion_cfg)
                    motion_detector_sig = next_motion_sig
                    detector_frames_seen = 0
                motion_ratio = _measure_motion_ratio(frame, motion_detector, detector_frames_seen, motion_cfg)
                worker["motion_ratio"] = motion_ratio
                detector_frames_seen += 1
                if motion_ratio >= float(cfg.get("motion_threshold", 2.5)):
                    _set_motion_trigger(worker["idx"], motion_ratio, int(cfg.get("motion_hold_ms", 4000)))
            else:
                worker["motion_ratio"] = 0.0
            frame = _enhance_frame(frame, cfg["contrast"], cfg["brightness"], cfg["saturation"], preset)
            preview_frame = _enhance_frame(preview_frame, cfg["contrast"], cfg["brightness"], cfg["saturation"], preset)
            if cfg["swap_rb"]:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), cfg["jpeg_quality"]])
            if not ok:
                continue
            ok_preview, preview_buf = cv2.imencode(".jpg", preview_frame, [int(cv2.IMWRITE_JPEG_QUALITY), cfg["jpeg_quality"]])
            worker["latest_jpeg"] = buf.tobytes()
            if ok_preview:
                worker["latest_preview_jpeg"] = preview_buf.tobytes()
            worker["last_frame_ms"] = _now_ms()
            worker["status"] = "streaming"
            last_emit = now
        cap.release()
        time.sleep(0.4)


def _get_or_start_stream_worker(stream_idx: int, stream_url: str):
    with stream_workers_lock:
        worker = stream_workers.get(stream_idx)
        if worker is not None and worker.get("url") != stream_url:
            worker["running"] = False
            del stream_workers[stream_idx]
            worker = None
        if worker is None:
            worker = {
                "idx": stream_idx,
                "url": stream_url,
                "transport": "",
                "running": True,
                "latest_jpeg": _encode_black_frame(),
                "latest_preview_jpeg": _encode_black_frame(),
                "status": "starting",
                "last_error": "",
                "last_frame_ms": 0,
                "last_access_ms": _now_ms(),
                "motion_ratio": 0.0,
            }
            stream_workers[stream_idx] = worker
            threading.Thread(target=_stream_capture_worker, args=(worker,), daemon=True).start()
        worker["last_access_ms"] = _now_ms()
        return worker


def _ensure_all_stream_workers():
    if not _is_rtsp_mode():
        return
    cfg = _settings_for_response()
    streams = cfg.get("streams", [])
    for idx, stream in enumerate(streams):
        stream_url = str(stream.get("url", "")).strip()
        if not stream_url:
            continue
        _get_or_start_stream_worker(idx, stream_url)
    _cleanup_stream_workers()


def _resize_jpeg_to(jpeg_bytes: bytes, size: tuple[int, int], quality: int = 80) -> bytes:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jpeg_bytes
    img = _fit_frame_for_size(img, size)
    ok, out = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return jpeg_bytes
    return out.tobytes()


def _fit_frame_for_size(frame, target_size):
    target_w, target_h = target_size
    src_h, src_w = frame.shape[:2]
    if src_w == 0 or src_h == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = max(target_w / src_w, target_h / src_h)
    resized_w = int(src_w * scale)
    resized_h = int(src_h * scale)
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=interpolation)
    x0 = (resized_w - target_w) // 2
    y0 = (resized_h - target_h) // 2
    return resized[y0:y0 + target_h, x0:x0 + target_w]


def _letterbox_frame_for_size(frame, target_size):
    target_w, target_h = target_size
    src_h, src_w = frame.shape[:2]
    if src_w == 0 or src_h == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = min(target_w / src_w, target_h / src_h)
    resized_w = max(1, int(src_w * scale))
    resized_h = max(1, int(src_h * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=interpolation)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - resized_w) // 2
    y0 = (target_h - resized_h) // 2
    canvas[y0:y0 + resized_h, x0:x0 + resized_w] = resized
    return canvas


def _get_frame_bytes(channel_index, ms, client_id):
    if _is_rtsp_mode():
        cfg = _settings_for_response()
        streams = cfg.get("streams", [])
        if len(streams) == 0:
            return _encode_black_frame()
        idx = _set_active_stream_from_channel(channel_index, client_id)
        worker = _get_or_start_stream_worker(idx, streams[idx]["url"])
        _cleanup_stream_workers(keep_idx=idx)
        _cleanup_client_sessions(keep_client_id=client_id)
        return worker.get("latest_jpeg") or _encode_black_frame()
    _ensure_movies_loaded()
    audio, frames = video_data[channel_index % len(video_data)]
    # use binary search to find the closest frame
    start = 0
    end = len(frames) - 1
    while start <= end:
        mid = (start + end) // 2
        if frames[mid][0] == ms:
            return frames[mid][1]
        elif frames[mid][0] < ms:
            start = mid + 1
        else:
            end = mid - 1
    if end < 0:
        end = 0
    elif start >= len(frames):
        start = len(frames) - 1
    return frames[start][1]


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


def init_video_source():
    global video_data
    _load_settings()
    print(
        "RTSP tuning:"
        f" open_timeout_ms={RTSP_OPEN_TIMEOUT_MS}"
        f" read_timeout_ms={RTSP_READ_TIMEOUT_MS}"
        f" stale_reconnect_ms={RTSP_STALE_RECONNECT_MS}"
        f" fps(max/balanced/quality)={RTSP_FPS_MAX}/{RTSP_FPS_BALANCED}/{RTSP_FPS_BEST_QUALITY}"
    )
    if _is_rtsp_mode():
        print("RTSP mode enabled")
        _ensure_all_stream_workers()
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
    client_id = _get_request_client_id()
    if _is_rtsp_mode():
        idx = _set_active_stream_from_channel(channel_index, client_id)
        _cleanup_client_sessions(keep_client_id=client_id)
        if length <= 0:
            return Response(b'', mimetype='audio/x-raw')
        cfg = _settings_for_response()
        if cfg.get("motion_enabled", False) and cfg.get("motion_audio_enabled", True) and _is_motion_active_for_stream(idx):
            alert = _get_motion_alert_audio()
            if len(alert) > 0:
                offset = max(0, int(start)) % len(alert)
                if length <= len(alert) - offset:
                    return Response(alert[offset:offset + length], mimetype='audio/x-raw')
                out = bytearray()
                remaining = int(length)
                pos = offset
                while remaining > 0:
                    take = min(remaining, len(alert) - pos)
                    out.extend(alert[pos:pos + take])
                    remaining -= take
                    pos = 0
                return Response(bytes(out), mimetype='audio/x-raw')
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
    client_id = _get_request_client_id()
    data = _get_frame_bytes(channel_index, ms, client_id)
    return Response(data, mimetype='image/jpeg')


@app.route('/frame_tdisplay/<int:channel_index>/<int:ms>')
def get_frame_tdisplay(channel_index, ms):
    client_id = _get_request_client_id()
    data = _get_frame_bytes(channel_index, ms, client_id)
    # TTGO T-Display screen is 240x135; send matching stream dimensions.
    data = _resize_jpeg_to(data, (240, 135), quality=78)
    return Response(data, mimetype='image/jpeg')


def _preview_worker_for(channel_index=None, client_id=None):
    cfg = _settings_for_response()
    streams = cfg.get("streams", [])
    if len(streams) <= 0:
        return None
    if channel_index is None:
        resolved_client_id = client_id or _get_request_client_id()
        idx = _get_client_active_stream_index(resolved_client_id, len(streams))
        _cleanup_client_sessions(keep_client_id=resolved_client_id)
    else:
        idx = max(0, min(int(channel_index), len(streams) - 1))
    worker = _get_or_start_stream_worker(idx, streams[idx]["url"])
    _cleanup_stream_workers(keep_idx=idx)
    return worker


def _preview_stream_target(channel_index=None, client_id=None):
    cfg = _settings_for_response()
    streams = cfg.get("streams", [])
    if len(streams) <= 0:
        return None, None
    if channel_index is None:
        resolved_client_id = client_id or _get_request_client_id()
        idx = _get_client_active_stream_index(resolved_client_id, len(streams))
        _cleanup_client_sessions(keep_client_id=resolved_client_id)
    else:
        idx = max(0, min(int(channel_index), len(streams) - 1))
    return idx, streams[idx]["url"]


def _latest_preview_frame(channel_index=None, client_id=None):
    idx, stream_url = _preview_stream_target(channel_index=channel_index, client_id=client_id)
    if idx is None or not stream_url:
        return _encode_black_frame()
    worker = _get_or_start_stream_worker(idx, stream_url)
    _cleanup_stream_workers(keep_idx=idx)
    frame = None
    if int(worker.get("last_frame_ms", 0) or 0) > 0:
        frame = worker.get("latest_preview_jpeg")
    if not frame:
        preview_worker = _get_or_start_preview_ffmpeg_worker(idx, stream_url)
        _cleanup_preview_ffmpeg_workers()
        if int(preview_worker.get("last_frame_ms", 0) or 0) > 0:
            frame = preview_worker.get("latest_jpeg")
    if not frame:
        frame = worker.get("latest_preview_jpeg")
    if not frame:
        frame = _encode_black_frame()
    return frame


def _stream_preview_mjpeg(channel_index=None, client_id=None):
    boundary = b"--frame\r\n"
    while True:
        frame = _latest_preview_frame(channel_index=channel_index, client_id=client_id)
        yield boundary
        yield b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.08)


@app.route("/preview.mjpg")
def preview_mjpg():
    client_id = _get_request_client_id()
    return Response(
        _stream_preview_mjpeg(client_id=client_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/preview/<int:channel_index>.mjpg")
def preview_channel_mjpg(channel_index):
    return Response(
        _stream_preview_mjpeg(channel_index=channel_index),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/preview/<int:channel_index>.jpg")
def preview_channel_jpg(channel_index):
    frame = _latest_preview_frame(channel_index=channel_index)
    return Response(frame, mimetype="image/jpeg")


@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    client_id = _get_request_client_id()
    _ensure_all_stream_workers()
    cfg = _settings_for_response()
    with capture_state_lock:
        state = dict(capture_state)
    streams = cfg.get("streams", [])
    if len(streams) > 0:
        idx = _get_client_active_stream_index(client_id, len(streams))
        cfg["active_stream_index"] = idx
        cfg["rtsp_url"] = streams[idx]["url"]
        worker = _get_or_start_stream_worker(idx, streams[idx]["url"])
        _cleanup_stream_workers(keep_idx=idx)
        _cleanup_client_sessions(keep_client_id=client_id)
        state["status"] = worker.get("status", state.get("status", "starting"))
        state["source"] = worker.get("url", streams[idx]["url"])
        state["last_error"] = worker.get("last_error", state.get("last_error", ""))
        state["last_frame_ms"] = worker.get("last_frame_ms", state.get("last_frame_ms", 0))
        state["active_stream_index"] = idx
        state["active_stream_name"] = streams[idx]["name"]
    with motion_state_lock:
        motion_snapshot = dict(motion_state)
    with motion_voice_lock:
        voice_snapshot = {
            "last_sent_ms": int(motion_voice_state.get("last_sent_ms", 0)),
            "last_sent_stream_idx": int(motion_voice_state.get("last_sent_stream_idx", -1)),
            "last_sent_stream_name": str(motion_voice_state.get("last_sent_stream_name", "")),
            "last_sent_speaker_entity": str(motion_voice_state.get("last_sent_speaker_entity", "")),
            "last_error": str(motion_voice_state.get("last_error", "")),
        }
    with stream_workers_lock:
        worker_snapshots = {
            int(idx): {
                "status": str(worker.get("status", "")),
                "last_error": str(worker.get("last_error", "")),
                "last_frame_ms": int(worker.get("last_frame_ms", 0) or 0),
                "motion_ratio": float(worker.get("motion_ratio", 0.0) or 0.0),
            }
            for idx, worker in stream_workers.items()
        }
    with preview_ffmpeg_workers_lock:
        preview_snapshots = {
            int(idx): {
                "status": str(worker.get("status", "")),
                "last_error": str(worker.get("last_error", "")),
                "last_frame_ms": int(worker.get("last_frame_ms", 0) or 0),
            }
            for idx, worker in preview_ffmpeg_workers.items()
        }
    now_ms = _now_ms()
    stream_states = []
    for idx, stream in enumerate(streams):
        worker = worker_snapshots.get(idx, {})
        preview_worker = preview_snapshots.get(idx, {})
        worker_last_frame_ms = int(worker.get("last_frame_ms", 0) or 0)
        preview_last_frame_ms = int(preview_worker.get("last_frame_ms", 0) or 0)
        effective_last_frame_ms = max(worker_last_frame_ms, preview_last_frame_ms)
        effective_status = str(worker.get("status", "idle"))
        effective_last_error = str(worker.get("last_error", ""))
        if preview_last_frame_ms > worker_last_frame_ms:
            effective_status = str(preview_worker.get("status", effective_status or "streaming"))
            effective_last_error = str(preview_worker.get("last_error", effective_last_error))
        if effective_last_frame_ms > 0 and effective_status in ("starting", "connecting", "reconnecting", "error"):
            effective_status = "streaming"
        motion_active_for_idx = bool(
            motion_snapshot.get("active", False)
            and now_ms < int(motion_snapshot.get("triggered_until_ms", 0))
            and int(motion_snapshot.get("source_stream_idx", -1)) == int(idx)
        )
        voice_for_idx = int(voice_snapshot.get("last_sent_stream_idx", -1)) == int(idx)
        stream_states.append({
            "idx": int(idx),
            "name": str(stream.get("name", f"Stream {idx + 1}")),
            "source": str(stream.get("url", "")),
            "status": effective_status,
            "last_error": effective_last_error,
            "last_frame_ms": effective_last_frame_ms,
            "motion_ratio": float(worker.get("motion_ratio", 0.0) or 0.0),
            "motion_active": motion_active_for_idx,
            "voice_last_sent_ms": int(voice_snapshot.get("last_sent_ms", 0) or 0) if voice_for_idx else 0,
            "voice_last_error": str(voice_snapshot.get("last_error", "")) if voice_for_idx else "",
            "voice_last_speaker_entity": str(voice_snapshot.get("last_sent_speaker_entity", "")) if voice_for_idx else "",
        })
    state["motion_active"] = bool(
        motion_snapshot.get("active", False)
        and _now_ms() < int(motion_snapshot.get("triggered_until_ms", 0))
    )
    state["motion_last_ms"] = int(motion_snapshot.get("last_motion_ms", 0))
    state["motion_ratio"] = float(motion_snapshot.get("last_motion_ratio", 0.0))
    state["motion_voice_last_sent_ms"] = voice_snapshot["last_sent_ms"]
    state["motion_voice_last_stream_idx"] = voice_snapshot["last_sent_stream_idx"]
    state["motion_voice_last_stream_name"] = voice_snapshot["last_sent_stream_name"]
    state["motion_voice_last_speaker_entity"] = voice_snapshot["last_sent_speaker_entity"]
    state["motion_voice_last_error"] = voice_snapshot["last_error"]
    return jsonify({"settings": cfg, "state": state, "stream_states": stream_states, "app_version": APP_VERSION})


@app.route("/api/settings", methods=["POST"])
def api_set_settings():
    global video_data
    client_id = _get_request_client_id()
    payload = request.get_json(silent=True) or {}
    errors = []
    updates = {}
    requested_active_index = None

    if "rtsp_url" in payload:
        updates["rtsp_url"] = str(payload["rtsp_url"]).strip()
    if "streams" in payload:
        streams = _normalize_streams(payload.get("streams", []))
        updates["streams"] = streams
        if len(streams) == 0:
            updates["active_stream_index"] = 0
    if "active_stream_index" in payload:
        try:
            requested_active_index = int(payload["active_stream_index"])
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
    if "motion_enabled" in payload:
        updates["motion_enabled"] = bool(payload["motion_enabled"])
    if "motion_threshold" in payload:
        try:
            motion_threshold = float(payload["motion_threshold"])
            if motion_threshold < 0.1 or motion_threshold > 50.0:
                raise ValueError
            updates["motion_threshold"] = motion_threshold
        except Exception:
            errors.append("motion_threshold must be between 0.1 and 50.0")
    if "motion_hold_ms" in payload:
        try:
            motion_hold_ms = int(payload["motion_hold_ms"])
            if motion_hold_ms < 200 or motion_hold_ms > 60000:
                raise ValueError
            updates["motion_hold_ms"] = motion_hold_ms
        except Exception:
            errors.append("motion_hold_ms must be between 200 and 60000")
    if "motion_detect_width" in payload:
        try:
            value = int(payload["motion_detect_width"])
            if value < 64 or value > 640:
                raise ValueError
            updates["motion_detect_width"] = value
        except Exception:
            errors.append("motion_detect_width must be an integer between 64 and 640")
    if "motion_detect_height" in payload:
        try:
            value = int(payload["motion_detect_height"])
            if value < 48 or value > 480:
                raise ValueError
            updates["motion_detect_height"] = value
        except Exception:
            errors.append("motion_detect_height must be an integer between 48 and 480")
    if "motion_bg_history" in payload:
        try:
            value = int(payload["motion_bg_history"])
            if value < 10 or value > 2000:
                raise ValueError
            updates["motion_bg_history"] = value
        except Exception:
            errors.append("motion_bg_history must be an integer between 10 and 2000")
    if "motion_bg_var_threshold" in payload:
        try:
            value = float(payload["motion_bg_var_threshold"])
            if value < 4.0 or value > 255.0:
                raise ValueError
            updates["motion_bg_var_threshold"] = value
        except Exception:
            errors.append("motion_bg_var_threshold must be between 4.0 and 255.0")
    if "motion_bg_learning_rate" in payload:
        try:
            value = float(payload["motion_bg_learning_rate"])
            if value < 0.0 or value > 1.0:
                raise ValueError
            updates["motion_bg_learning_rate"] = value
        except Exception:
            errors.append("motion_bg_learning_rate must be between 0.0 and 1.0")
    if "motion_min_contour_area_pct" in payload:
        try:
            value = float(payload["motion_min_contour_area_pct"])
            if value < 0.001 or value > 20.0:
                raise ValueError
            updates["motion_min_contour_area_pct"] = value
        except Exception:
            errors.append("motion_min_contour_area_pct must be between 0.001 and 20.0")
    if "motion_warmup_frames" in payload:
        try:
            value = int(payload["motion_warmup_frames"])
            if value < 0 or value > 600:
                raise ValueError
            updates["motion_warmup_frames"] = value
        except Exception:
            errors.append("motion_warmup_frames must be an integer between 0 and 600")
    if "motion_audio_enabled" in payload:
        updates["motion_audio_enabled"] = bool(payload["motion_audio_enabled"])
    if "motion_voice_enabled" in payload:
        updates["motion_voice_enabled"] = bool(payload["motion_voice_enabled"])
    if "motion_voice_cooldown_ms" in payload:
        try:
            motion_voice_cooldown_ms = int(payload["motion_voice_cooldown_ms"])
            if motion_voice_cooldown_ms < 1000 or motion_voice_cooldown_ms > 3_600_000:
                raise ValueError
            updates["motion_voice_cooldown_ms"] = motion_voice_cooldown_ms
        except Exception:
            errors.append("motion_voice_cooldown_ms must be between 1000 and 3600000")
    if "motion_voice_daytime_only" in payload:
        updates["motion_voice_daytime_only"] = bool(payload["motion_voice_daytime_only"])
    if "motion_voice_start_hour" in payload:
        try:
            hour = int(payload["motion_voice_start_hour"])
            if hour < 0 or hour > 23:
                raise ValueError
            updates["motion_voice_start_hour"] = hour
        except Exception:
            errors.append("motion_voice_start_hour must be an integer between 0 and 23")
    if "motion_voice_end_hour" in payload:
        try:
            hour = int(payload["motion_voice_end_hour"])
            if hour < 0 or hour > 23:
                raise ValueError
            updates["motion_voice_end_hour"] = hour
        except Exception:
            errors.append("motion_voice_end_hour must be an integer between 0 and 23")
    if "motion_voice_message" in payload:
        updates["motion_voice_message"] = str(payload["motion_voice_message"] or "")[:240]
    if "motion_voice_default_entity" in payload:
        updates["motion_voice_default_entity"] = str(payload["motion_voice_default_entity"] or "").strip()[:128]
    if "ha_direct_tts_enabled" in payload:
        updates["ha_direct_tts_enabled"] = bool(payload["ha_direct_tts_enabled"])
    if "ha_base_url" in payload:
        base_url = _normalize_ha_base_url(payload["ha_base_url"])
        if base_url and not (base_url.startswith("http://") or base_url.startswith("https://")):
            errors.append("ha_base_url must start with http:// or https://")
        else:
            updates["ha_base_url"] = base_url[:2048]
    if "ha_tts_entity" in payload:
        updates["ha_tts_entity"] = str(payload["ha_tts_entity"] or "").strip()[:128]
    if "ha_webhook_url" in payload:
        url = str(payload["ha_webhook_url"] or "").strip()
        if url and not (url.startswith("http://") or url.startswith("https://")):
            errors.append("ha_webhook_url must start with http:// or https://")
        else:
            updates["ha_webhook_url"] = url[:2048]
    if "ha_bearer_token" in payload:
        updates["ha_bearer_token"] = str(payload["ha_bearer_token"] or "").strip()[:1024]
    if "ha_webhook_secret" in payload:
        updates["ha_webhook_secret"] = str(payload["ha_webhook_secret"] or "").strip()[:1024]

    current_cfg = _settings_for_response()
    effective_motion_voice_enabled = bool(updates.get("motion_voice_enabled", current_cfg.get("motion_voice_enabled", False)))
    effective_direct_tts_enabled = bool(updates.get("ha_direct_tts_enabled", current_cfg.get("ha_direct_tts_enabled", True)))
    effective_base_url = _normalize_ha_base_url(str(updates.get("ha_base_url", current_cfg.get("ha_base_url", ""))).strip())
    effective_tts_entity = str(updates.get("ha_tts_entity", current_cfg.get("ha_tts_entity", ""))).strip()
    effective_bearer_token = str(updates.get("ha_bearer_token", current_cfg.get("ha_bearer_token", ""))).strip()
    effective_webhook_url = str(updates.get("ha_webhook_url", current_cfg.get("ha_webhook_url", ""))).strip()
    if effective_motion_voice_enabled:
        if effective_direct_tts_enabled:
            if not effective_base_url:
                errors.append("ha_base_url is required when direct Home Assistant TTS is enabled")
            if not effective_bearer_token:
                errors.append("ha_bearer_token is required when direct Home Assistant TTS is enabled")
            if not effective_tts_entity:
                errors.append("ha_tts_entity is required when direct Home Assistant TTS is enabled")
        elif not effective_webhook_url:
            errors.append("ha_webhook_url is required when webhook mode is selected")

    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    with settings_lock:
        settings.update(updates)
        if "rtsp_url" in updates and "streams" not in updates:
            if updates["rtsp_url"]:
                settings["streams"] = [{"name": "Stream 1", "url": updates["rtsp_url"], "motion_voice": False, "voice_entity": ""}]
                settings["active_stream_index"] = 0
            else:
                settings["streams"] = []
                settings["active_stream_index"] = 0
        _sync_rtsp_url_locked()
        applied = dict(settings)
    stream_count = len(applied.get("streams", []))
    if requested_active_index is not None:
        selected_idx = _set_client_active_stream_index(client_id, requested_active_index, stream_count)
    else:
        selected_idx = _get_client_active_stream_index(client_id, stream_count)
    if stream_count > 0:
        applied["active_stream_index"] = selected_idx
        applied["rtsp_url"] = applied["streams"][selected_idx]["url"]
    else:
        applied["active_stream_index"] = 0
        applied["rtsp_url"] = ""
    _cleanup_client_sessions(keep_client_id=client_id)
    persist_warning = ""
    try:
        _persist_settings()
    except Exception as ex:
        persist_warning = f"Settings applied in memory, but failed to save to disk: {ex}"
        print("warning:", persist_warning)
    # Allow live switching between movie mode and RTSP mode without restarting the server.
    now_rtsp_mode = len(applied.get("streams", [])) > 0
    if now_rtsp_mode:
        _set_capture_state(status="starting", source="", last_error="")
        _ensure_all_stream_workers()
    else:
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
    board = str(payload.get("board", "cyd")).strip()
    flavor = str(payload.get("flavor", "audio_on")).strip()
    ssid = str(payload.get("ssid", "")).strip()
    password = str(payload.get("password", ""))
    server_host = str(payload.get("server_host", "")).strip()
    upload_port = str(payload.get("upload_port", "")).strip()
    try:
        server_port = int(payload.get("server_port", 8124))
    except Exception:
        return jsonify({"ok": False, "error": "server_port must be an integer"}), 400

    if board not in FLASH_TARGETS:
        return jsonify({"ok": False, "error": f"board must be one of: {', '.join(FLASH_TARGETS.keys())}"}), 400
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
        "board": board,
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
    payload = request.get_json(silent=True) or {}
    board = str(payload.get("board", "cyd")).strip()
    flavor = str(payload.get("flavor", "no_audio")).strip()
    ssid = str(payload.get("ssid", "")).strip()
    password = str(payload.get("password", ""))
    server_host = str(payload.get("server_host", "")).strip()
    try:
        server_port = int(payload.get("server_port", 8124))
    except Exception:
        return jsonify({"ok": False, "error": "server_port must be an integer"}), 400

    if board not in FLASH_TARGETS:
        return jsonify({"ok": False, "error": f"board must be one of: {', '.join(FLASH_TARGETS.keys())}"}), 400
    if not _webflash_templates_exist(board):
        return jsonify({"ok": False, "error": f"Web flash templates are not available for board '{board}' in this deployment"}), 501
    if flavor not in ("audio_on", "no_audio"):
        return jsonify({"ok": False, "error": "flavor must be audio_on or no_audio"}), 400
    if not ssid:
        return jsonify({"ok": False, "error": "SSID is required"}), 400
    if not server_host:
        return jsonify({"ok": False, "error": "Server host is required"}), 400
    if server_port < 1 or server_port > 65535:
        return jsonify({"ok": False, "error": "Server port must be 1-65535"}), 400

    try:
        firmware = _build_custom_webflash_firmware(board, flavor, ssid, password, server_host, server_port)
    except Exception as ex:
        return jsonify({"ok": False, "error": f"Failed to build custom firmware: {ex}"}), 500

    payload_id = secrets.token_hex(12)
    _cleanup_webflash_payloads()
    with webflash_lock:
        webflash_payloads[payload_id] = {
            "created_ms": _now_ms(),
            "board": board,
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
    board = entry.get("board", "cyd")
    if board not in FLASH_TARGETS:
        return jsonify({"ok": False, "error": "Manifest board is not supported"}), 400
    chip_family = str(FLASH_TARGETS[board].get("chip_family", "ESP32")).strip() or "ESP32"
    wf = FLASH_TARGETS[board]["webflash"][flavor]
    manifest = {
        "name": f"ESP32 TV {FLASH_TARGETS[board]['label']} Custom",
        "version": "1.0.0",
        "new_install_prompt_erase": True,
        "builds": [{
            "chipFamily": chip_family,
            "parts": [
                {"path": f"/static/firmware/{wf['bootloader']}", "offset": 4096},
                {"path": f"/static/firmware/{wf['partitions']}", "offset": 32768},
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
      align-items: start;
    }
    .card {
      background: var(--panel);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 18px 40px rgba(0,0,0,0.25);
      backdrop-filter: blur(6px);
    }
    .page-head {
      display: grid;
      gap: 12px;
      background: rgba(255,255,255,0.9);
    }
    .page-copy {
      display: grid;
      gap: 6px;
    }
    .column-stack {
      display: grid;
      gap: 18px;
      align-content: start;
    }
    .main-stack {
      grid-column: 1;
    }
    .side-stack {
      grid-column: 2;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 26px;
      letter-spacing: .2px;
    }
    p { margin: 0 0 14px; color: var(--muted); }
    .row { margin: 10px 0; }
    label { display:block; font-weight: 700; margin-bottom: 6px; font-size: 14px; }
    .field-head {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 6px;
    }
    .field-head label {
      margin: 0;
      flex: 1;
      min-width: 0;
    }
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
      flex-wrap: wrap;
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
    .section-card {
      position: relative;
      overflow: hidden;
      padding: 16px;
      border: 1px solid rgba(255,255,255,0.35);
    }
    .section-card::before {
      content: "";
      position: absolute;
      inset: 0 0 auto 0;
      height: 4px;
      background: linear-gradient(90deg, var(--section-accent, #0d8fdf), transparent 70%);
      opacity: 0.9;
    }
    .streams-card { --section-accent: #2f7dbd; background: rgba(239,247,255,0.92); }
    .feeds-card { --section-accent: #2563eb; background: rgba(242,247,255,0.92); }
    .image-card { --section-accent: #b97a1f; background: rgba(255,249,237,0.94); }
    .status-card { --section-accent: #64748b; background: rgba(247,250,252,0.94); }
    .motion-card { --section-accent: #0f766e; background: rgba(239,251,248,0.94); }
    .voice-card { --section-accent: #c4634f; background: rgba(255,245,241,0.94); }
    .firmware-card { --section-accent: #475569; background: rgba(243,246,250,0.94); }
    .subsection-card {
      margin-top: 14px;
      background: rgba(255,255,255,0.65);
      box-shadow: none;
      border: 1px solid rgba(209, 219, 231, 0.9);
      padding: 14px;
    }
    .subsection-card::before {
      height: 3px;
      opacity: 0.7;
    }
    .group-head {
      display: flex;
      align-items: flex-start;
      gap: 10px;
    }
    .group-toggle {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      background: transparent;
      color: inherit;
      padding: 0;
      text-align: left;
    }
    .group-copy {
      display: grid;
      gap: 3px;
      min-width: 0;
    }
    .group-kicker {
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 800;
      color: var(--section-accent, #0d8fdf);
    }
    .group-title {
      font-size: 18px;
      font-weight: 800;
      color: #17263b;
    }
    .group-summary {
      font-size: 12px;
      line-height: 1.45;
      color: #5a697c;
    }
    .group-chevron {
      width: 12px;
      height: 12px;
      border-right: 2px solid currentColor;
      border-bottom: 2px solid currentColor;
      transform: rotate(45deg);
      transition: transform 0.18s ease;
      color: #4b5b71;
      flex: 0 0 auto;
      margin-right: 4px;
    }
    .section-card.is-open > .group-head .group-chevron {
      transform: rotate(225deg);
      margin-top: 6px;
    }
    .info-btn {
      width: 34px;
      height: 34px;
      border-radius: 999px;
      padding: 0;
      background: rgba(255,255,255,0.78);
      color: var(--section-accent, #0d8fdf);
      border: 1px solid rgba(209, 219, 231, 0.9);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.9);
      font-size: 14px;
      font-weight: 800;
      flex: 0 0 auto;
    }
    .mini-info {
      width: 22px;
      height: 22px;
      font-size: 11px;
      border-radius: 999px;
      padding: 0;
    }
    .section-body {
      margin-top: 14px;
      max-height: 0;
      opacity: 0;
      overflow: hidden;
      pointer-events: none;
      transition: max-height 0.24s ease, opacity 0.18s ease;
    }
    .section-card.is-open > .section-body {
      max-height: 5000px;
      opacity: 1;
      overflow: visible;
      pointer-events: auto;
    }
    .help-overlay {
      position: fixed;
      inset: 0;
      background: rgba(9, 17, 29, 0.45);
      display: grid;
      place-items: center;
      padding: 20px;
      z-index: 40;
    }
    .help-overlay[hidden] {
      display: none !important;
    }
    .help-modal {
      width: min(560px, 100%);
      background: rgba(255,255,255,0.97);
      color: var(--ink);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 24px 60px rgba(0,0,0,0.28);
      border: 1px solid rgba(214, 222, 234, 0.95);
    }
    .help-modal-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    .help-title {
      font-size: 20px;
      font-weight: 800;
      color: #17263b;
    }
    .help-body {
      display: grid;
      gap: 10px;
      font-size: 14px;
      line-height: 1.55;
      color: #405064;
    }
    .preview {
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      object-position: center;
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
    .all-feeds-box {
      margin-top: 0;
      padding-top: 0;
    }
    .all-feeds-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      margin-top: 8px;
    }
    .feed-tile {
      width: 100%;
      text-align: left;
      border: 1px solid #d6deea;
      border-radius: 10px;
      background: #f8fbff;
      padding: 0;
      overflow: hidden;
      color: #1f2937;
    }
    .feed-tile.active {
      border-color: #0d8fdf;
      box-shadow: inset 0 0 0 1px #0d8fdf;
      background: #eaf5ff;
    }
    .feed-thumb {
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      object-position: center;
      border-radius: 12px;
      border: 2px solid #d7e2ef;
      background: #0c1420;
      display: block;
    }
    .feed-head {
      font-size: 13px;
      font-weight: 700;
      color: #1f2f46;
      margin-top: 6px;
      padding: 0 8px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .feed-url {
      font-size: 11px;
      color: #58687d;
      margin-top: 3px;
      padding: 0 8px 8px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .feed-stats {
      border-top: 1px solid #d7e2ef;
      background: #f2f7fd;
      padding: 6px 8px 8px;
      display: grid;
      gap: 4px;
    }
    .feed-stat {
      font-size: 11px;
      color: #475569;
      line-height: 1.25;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    @media (max-width: 900px) {
      .wrap { grid-template-columns: 1fr; }
      .main-stack,
      .side-stack {
        grid-column: auto;
      }
      .group-head {
        align-items: stretch;
      }
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
      grid-template-columns: 0.8fr 1.2fr 1fr auto auto;
      gap: 8px;
      align-items: center;
    }
    .stream-voice-entity {
      min-width: 0;
    }
    .stream-voice-toggle {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      font-weight: 600;
      color: #334155;
      white-space: nowrap;
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
    <section class="card page-head span-2">
      <div class="page-copy">
        <h1>CYD Stream Control</h1>
        <div class="pill">Version {{ app_version }}</div>
        <p>Controls are grouped into collapsible cards. Open only the section you need, then apply all changes when you are ready.</p>
      </div>
      <div class="actions">
        <button class="primary" id="save_btn">Apply All Changes</button>
        <button class="secondary" id="refresh_btn">Reload Current</button>
        <button class="secondary" id="open_video_feeds_btn" type="button">Open Video Feeds</button>
      </div>
      <div id="status_box" class="status">Loading…</div>
    </section>

    <div class="column-stack main-stack">
    <section class="card section-card streams-card is-open" data-section="streams">
      <div class="group-head">
        <button type="button" class="group-toggle" data-toggle-section="streams" aria-expanded="true">
          <span class="group-copy">
            <span class="group-kicker">Streams</span>
            <span class="group-title">Source Selection</span>
            <span class="group-summary">Add, rename, and switch the feeds available to this browser.</span>
          </span>
          <span class="group-chevron" aria-hidden="true"></span>
        </button>
        <button type="button" class="info-btn" data-help-key="streams" aria-label="About streams">i</button>
      </div>
      <div class="section-body">
        <div class="row">
          <label for="rtsp_url">Current Stream URL</label>
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
      </div>
    </section>

    <section class="card section-card image-card" data-section="image">
      <div class="group-head">
        <button type="button" class="group-toggle" data-toggle-section="image" aria-expanded="false">
          <span class="group-copy">
            <span class="group-kicker">Image</span>
            <span class="group-title">Display Tuning</span>
            <span class="group-summary">Adjust picture quality, color balance, and quick display presets.</span>
          </span>
          <span class="group-chevron" aria-hidden="true"></span>
        </button>
        <button type="button" class="info-btn" data-help-key="image" aria-label="About image tuning">i</button>
      </div>
      <div class="section-body">
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
          <button class="secondary" id="preset_fps_btn">Preset: Max FPS</button>
          <button class="secondary" id="preset_quality_btn">Preset: Best Quality</button>
        </div>
      </div>
    </section>

    <section class="card section-card status-card is-open" data-section="status">
      <div class="group-head">
        <button type="button" class="group-toggle" data-toggle-section="status" aria-expanded="true">
          <span class="group-copy">
            <span class="group-kicker">System</span>
            <span class="group-title">Live Status</span>
            <span class="group-summary">The current state of this browser session and the active server stream.</span>
          </span>
          <span class="group-chevron" aria-hidden="true"></span>
        </button>
        <button type="button" class="info-btn" data-help-key="status" aria-label="About system status">i</button>
      </div>
      <div class="section-body">
        <div class="meta">
          <div><strong>Stream State:</strong> <span id="runtime_status_state">-</span></div>
          <div><strong>Selected Stream:</strong> <span id="runtime_status_stream">-</span></div>
          <div><strong>Current Source:</strong> <span id="runtime_status_source">-</span></div>
          <div><strong>Last Frame:</strong> <span id="runtime_status_frame">-</span></div>
          <div><strong>Motion:</strong> <span id="runtime_status_motion">-</span></div>
          <div><strong>Last Error:</strong> <span id="runtime_status_error">-</span></div>
        </div>
      </div>
    </section>

    <section class="card section-card motion-card" data-section="motion">
      <div class="group-head">
        <button type="button" class="group-toggle" data-toggle-section="motion" aria-expanded="false">
          <span class="group-copy">
            <span class="group-kicker">Motion</span>
            <span class="group-title">Detection Rules</span>
            <span class="group-summary">Set how easily movement is detected, and whether audio alerts should play.</span>
          </span>
          <span class="group-chevron" aria-hidden="true"></span>
        </button>
        <button type="button" class="info-btn" data-help-key="motion" aria-label="About motion detection">i</button>
      </div>
      <div class="section-body">
        <div class="row toggle">
          <input id="motion_enabled" type="checkbox" />
          <label for="motion_enabled" style="margin:0">Motion Detection Enabled</label>
        </div>
        <div class="row">
          <label for="motion_threshold">Motion Threshold (%)</label>
          <input id="motion_threshold" type="number" min="0.1" max="50" step="0.1" />
        </div>
        <div class="row">
          <label for="motion_hold_ms">Motion Hold (ms)</label>
          <input id="motion_hold_ms" type="number" min="200" max="60000" step="100" />
        </div>
        <div class="row toggle">
          <input id="motion_audio_enabled" type="checkbox" />
          <label for="motion_audio_enabled" style="margin:0">Play Alert Audio On Motion</label>
        </div>
        <p style="margin: 6px 0 0; font-size: 12px; color: var(--muted);">Optional alert files: `server/cache/motion_alert.wav` or `server/cache/motion_alert_u8_16k.raw`.</p>

        <div class="subsection-card section-card" data-section="motion-advanced">
          <div class="group-head">
            <button type="button" class="group-toggle" data-toggle-section="motion-advanced" aria-expanded="false">
              <span class="group-copy">
                <span class="group-kicker">Advanced</span>
                <span class="group-title">Detector Tuning</span>
                <span class="group-summary">Most users can leave these alone. Change only if you need less noise or faster reaction.</span>
              </span>
              <span class="group-chevron" aria-hidden="true"></span>
            </button>
            <button type="button" class="info-btn" data-help-key="motion-advanced" aria-label="About advanced motion settings">i</button>
          </div>
          <div class="section-body">
            <div class="row">
              <div class="field-head">
                <label for="motion_detect_width">Detection Width</label>
                <button type="button" class="info-btn mini-info" data-help-key="motion_detect_width_field" aria-label="About detection width">i</button>
              </div>
              <input id="motion_detect_width" type="number" min="64" max="640" step="1" />
            </div>
            <div class="row">
              <div class="field-head">
                <label for="motion_detect_height">Detection Height</label>
                <button type="button" class="info-btn mini-info" data-help-key="motion_detect_height_field" aria-label="About detection height">i</button>
              </div>
              <input id="motion_detect_height" type="number" min="48" max="480" step="1" />
            </div>
            <div class="row">
              <div class="field-head">
                <label for="motion_bg_history">Background History</label>
                <button type="button" class="info-btn mini-info" data-help-key="motion_bg_history_field" aria-label="About background history">i</button>
              </div>
              <input id="motion_bg_history" type="number" min="10" max="2000" step="1" />
            </div>
            <div class="row">
              <div class="field-head">
                <label for="motion_bg_var_threshold">Sensitivity To Scene Changes</label>
                <button type="button" class="info-btn mini-info" data-help-key="motion_bg_var_threshold_field" aria-label="About sensitivity to scene changes">i</button>
              </div>
              <input id="motion_bg_var_threshold" type="number" min="4" max="255" step="0.1" />
            </div>
            <div class="row">
              <div class="field-head">
                <label for="motion_bg_learning_rate">Background Adaptation Speed</label>
                <button type="button" class="info-btn mini-info" data-help-key="motion_bg_learning_rate_field" aria-label="About background adaptation speed">i</button>
              </div>
              <input id="motion_bg_learning_rate" type="number" min="0" max="1" step="0.001" />
            </div>
            <div class="row">
              <div class="field-head">
                <label for="motion_min_contour_area_pct">Minimum Movement Size (%)</label>
                <button type="button" class="info-btn mini-info" data-help-key="motion_min_contour_area_pct_field" aria-label="About minimum movement size">i</button>
              </div>
              <input id="motion_min_contour_area_pct" type="number" min="0.001" max="20" step="0.001" />
            </div>
            <div class="row">
              <div class="field-head">
                <label for="motion_warmup_frames">Startup Stabilization (frames)</label>
                <button type="button" class="info-btn mini-info" data-help-key="motion_warmup_frames_field" aria-label="About startup stabilization">i</button>
              </div>
              <input id="motion_warmup_frames" type="number" min="0" max="600" step="1" />
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="card section-card voice-card" data-section="voice">
      <div class="group-head">
        <button type="button" class="group-toggle" data-toggle-section="voice" aria-expanded="false">
          <span class="group-copy">
            <span class="group-kicker">Voice</span>
            <span class="group-title">Alerts & Home Assistant</span>
            <span class="group-summary">Control spoken alerts, timing rules, and Home Assistant delivery settings.</span>
          </span>
          <span class="group-chevron" aria-hidden="true"></span>
        </button>
        <button type="button" class="info-btn" data-help-key="voice" aria-label="About voice alerts">i</button>
      </div>
      <div class="section-body">
        <div class="row toggle">
          <input id="motion_voice_enabled" type="checkbox" />
          <label for="motion_voice_enabled" style="margin:0">Home Assistant Voice Alerts Enabled</label>
        </div>
        <div class="row toggle">
          <input id="motion_voice_daytime_only" type="checkbox" />
          <label for="motion_voice_daytime_only" style="margin:0">Voice Alerts Daytime Only</label>
        </div>
        <div class="row">
          <label for="motion_voice_start_hour">Daytime Start Hour (0-23)</label>
          <input id="motion_voice_start_hour" type="number" min="0" max="23" step="1" />
        </div>
        <div class="row">
          <label for="motion_voice_end_hour">Daytime End Hour (0-23)</label>
          <input id="motion_voice_end_hour" type="number" min="0" max="23" step="1" />
        </div>
        <div class="row">
          <label for="motion_voice_cooldown_ms">Voice Alert Cooldown (ms)</label>
          <input id="motion_voice_cooldown_ms" type="number" min="1000" max="3600000" step="500" />
        </div>
        <div class="row">
          <label for="motion_voice_message">Voice Message Template</label>
          <input id="motion_voice_message" type="text" maxlength="240" placeholder="Motion detected on {stream_name}" />
        </div>
        <div class="row">
          <label for="motion_voice_default_entity">Default Speaker Entity (optional)</label>
          <input id="motion_voice_default_entity" type="text" maxlength="128" placeholder="media_player.library_pair" />
        </div>
        <div class="row toggle">
          <input id="ha_direct_tts_enabled" type="checkbox" />
          <label for="ha_direct_tts_enabled" style="margin:0">Use Direct Home Assistant TTS (No YAML)</label>
        </div>
        <div class="row voice-mode-direct">
          <label for="ha_base_url">Home Assistant URL</label>
          <input id="ha_base_url" type="text" placeholder="http://homeassistant.local:8123" />
        </div>
        <div class="row">
          <label for="ha_bearer_token">Home Assistant Bearer Token</label>
          <input id="ha_bearer_token" type="text" placeholder="Long-lived access token" />
        </div>
        <div class="row voice-mode-direct">
          <label for="ha_tts_entity">TTS Engine Entity</label>
          <input id="ha_tts_entity" type="text" placeholder="tts.google_translate_en_com" />
        </div>
        <div class="row voice-mode-webhook">
          <label for="ha_webhook_url">Webhook URL (fallback mode)</label>
          <input id="ha_webhook_url" type="text" placeholder="http://homeassistant.local:8123/api/webhook/your_id" />
        </div>
        <div class="row voice-mode-webhook">
          <label for="ha_webhook_secret">Webhook Secret Header (optional)</label>
          <input id="ha_webhook_secret" type="text" placeholder="Sends as X-Webhook-Secret" />
        </div>
        <p style="margin: 6px 0 0; font-size: 12px; color: var(--muted);">
          Set per-stream voice alerts in the stream list above. Direct TTS mode does not require Home Assistant YAML. Daytime windows use the server's local timezone.
        </p>
      </div>
    </section>
    </div>

    <div class="column-stack side-stack">
      <section class="card section-card feeds-card is-open" data-section="feeds">
        <div class="group-head">
          <button type="button" class="group-toggle" data-toggle-section="feeds" aria-expanded="true">
            <span class="group-copy">
              <span class="group-kicker">Feeds</span>
              <span class="group-title">All Saved Feeds</span>
              <span class="group-summary">Live previews with per-feed source, last frame, motion, and voice status.</span>
            </span>
            <span class="group-chevron" aria-hidden="true"></span>
          </button>
          <button type="button" class="info-btn" data-help-key="feeds" aria-label="About feed previews">i</button>
        </div>
        <div class="section-body">
          <div class="all-feeds-box">
            <div id="all_feeds_grid" class="all-feeds-grid"></div>
          </div>
        </div>
      </section>
    </div>

    <section class="card section-card firmware-card span-2" data-section="firmware">
      <div class="group-head">
        <button type="button" class="group-toggle" data-toggle-section="firmware" aria-expanded="false">
          <span class="group-copy">
            <span class="group-kicker">Firmware</span>
            <span class="group-title">Build & Flash Devices</span>
            <span class="group-summary">USB flash from this machine, or prepare browser flashing packages for container deployments.</span>
          </span>
          <span class="group-chevron" aria-hidden="true"></span>
        </button>
        <button type="button" class="info-btn" data-help-key="firmware" aria-label="About firmware flashing">i</button>
      </div>
      <div class="section-body">
        <div class="firm-grid">
          <div class="row">
            <label for="fw_board">Board</label>
            <select id="fw_board">
              <option value="cyd">Cheap Yellow Display (ESP32-2432S028R)</option>
              <option value="ttgo_tdisplay">TTGO T-Display</option>
              <option value="esp32_s3_2p8">ESP32-S3 2.8&quot; 240x320</option>
            </select>
          </div>
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
            <input id="fw_ssid" type="text" placeholder="Your Wi-Fi name" value="Homelan" />
          </div>
          <div class="row">
            <label for="fw_password">Wi-Fi Password</label>
            <input id="fw_password" type="text" placeholder="Your Wi-Fi password" />
          </div>
          <div class="row">
            <label for="fw_server_host">Server Host/IP</label>
            <input id="fw_server_host" type="text" placeholder="192.168.1.16" />
          </div>
          <div class="row">
            <label for="fw_server_port">Server Port</label>
            <input id="fw_server_port" type="number" min="1" max="65535" step="1" value="8124" />
          </div>
        </div>
        <div class="actions">
          <button class="primary" id="flash_btn">Build + Flash Device</button>
          <button class="secondary" id="refresh_flash_btn">Refresh Flash Status</button>
        </div>
        <div id="flash_status_box" class="status">Idle</div>
        <div id="flash_ports" class="meta"></div>
        <div id="flash_log" class="mono"></div>

        <div class="subsection-card section-card is-open" data-section="webflash">
          <div class="group-head">
            <button type="button" class="group-toggle" data-toggle-section="webflash" aria-expanded="true">
              <span class="group-copy">
                <span class="group-kicker">Browser Flash</span>
                <span class="group-title">Prepare Web Installer</span>
                <span class="group-summary">Use Chrome or Edge on the machine physically connected to the ESP device.</span>
              </span>
              <span class="group-chevron" aria-hidden="true"></span>
            </button>
            <button type="button" class="info-btn" data-help-key="webflash" aria-label="About browser flashing">i</button>
          </div>
          <div class="section-body">
            <div class="firm-grid">
              <div class="row">
                <label for="webflash_board">Board</label>
                <select id="webflash_board">
                  <option value="cyd">Cheap Yellow Display (ESP32-2432S028R)</option>
                  <option value="ttgo_tdisplay">TTGO T-Display</option>
                  <option value="esp32_s3_2p8">ESP32-S3 2.8&quot; 240x320</option>
                </select>
              </div>
              <div class="row">
                <label for="webflash_ssid">Wi-Fi SSID</label>
                <input id="webflash_ssid" type="text" placeholder="Your Wi-Fi name" value="Homelan" />
              </div>
              <div class="row">
                <label for="webflash_password">Wi-Fi Password</label>
                <input id="webflash_password" type="text" placeholder="Your Wi-Fi password" />
              </div>
              <div class="row">
                <label for="webflash_server_host">Server Host/IP</label>
                <input id="webflash_server_host" type="text" placeholder="192.168.1.16" />
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
        </div>
      </div>
    </section>
  </div>

  <div id="help_overlay" class="help-overlay" hidden>
    <div class="help-modal" role="dialog" aria-modal="true" aria-labelledby="help_title">
      <div class="help-modal-head">
        <div>
          <div class="group-kicker">Help</div>
          <div id="help_title" class="help-title">Section Help</div>
        </div>
        <button type="button" class="secondary tiny" id="help_close_btn">Close</button>
      </div>
      <div id="help_body" class="help-body"></div>
    </div>
  </div>

  <script>
    const CLIENT_ID_KEY = "esp32_tv_client_id";
    function getClientId() {
      let cid = localStorage.getItem(CLIENT_ID_KEY) || "";
      if (!cid) {
        if (window.crypto && window.crypto.randomUUID) {
          cid = "web-" + window.crypto.randomUUID();
        } else {
          cid = "web-" + Math.random().toString(36).slice(2) + Date.now().toString(36);
        }
        localStorage.setItem(CLIENT_ID_KEY, cid);
      }
      return cid;
    }
    const clientId = getClientId();
    function withCid(path) {
      return path + (path.includes("?") ? "&" : "?") + "cid=" + encodeURIComponent(clientId);
    }

    function pageIsVisible() {
      return !document.hidden;
    }

    const SECTION_STATE_KEY = "esp32_tv_admin_sections_v1";
    const DEFAULT_SECTION_STATE = {
      streams: true,
      feeds: true,
      image: false,
      status: true,
      motion: false,
      "motion-advanced": false,
      voice: false,
      firmware: false,
      webflash: true
    };
    const HELP_CONTENT = {
      streams: {
        title: "Streams",
        body: [
          "Use this section to define which RTSP feeds are available to this browser session.",
          "The Current Stream URL follows the currently selected saved stream, so changing it updates that saved entry.",
          "Apply All Changes after editing streams to make the server reconnect cleanly."
        ]
      },
      feeds: {
        title: "All Saved Feeds",
        body: [
          "Each tile shows a live preview for one saved stream.",
          "The text beneath each tile shows the source URL, last frame time, motion state, and the last voice alert result for that stream.",
          "This panel is read-only. It does not switch streams and does not track a selected tile."
        ]
      },
      image: {
        title: "Display Tuning",
        body: [
          "These controls affect how frames are encoded and how they look on the display.",
          "JPEG Quality affects compression, while contrast, brightness, and saturation shape the final image.",
          "Use presets first, then fine-tune only if needed."
        ]
      },
      status: {
        title: "Live Status",
        body: [
          "This section reflects the current state of the stream selected by this browser.",
          "It shows stream health, the active source, the most recent frame time, motion state, and the latest stream error.",
          "It is read-only and updates automatically."
        ]
      },
      motion: {
        title: "Motion Detection",
        body: [
          "Keep the basic controls simple: enable motion, set the threshold, and choose how long a trigger stays active.",
          "Lower thresholds make detection more sensitive. Higher thresholds reduce false triggers.",
          "Audio alerts play locally from the server if a compatible alert file is present."
        ]
      },
      "motion-advanced": {
        title: "Advanced Motion Settings",
        body: [
          "These controls tune the background-subtraction detector introduced to reduce noise and flicker.",
          "Lower sensitivity thresholds and smaller movement sizes react faster but can cause more false positives.",
          "Change one setting at a time and watch the per-feed motion readout before adjusting further.",
          "Each field in this section now has its own info button with a deeper explanation."
        ]
      },
      motion_detect_width_field: {
        title: "Detection Width",
        body: [
          "This is the width of the reduced grayscale frame used only for motion analysis, not for display output.",
          "A larger value preserves more detail and can detect smaller movements, but increases CPU use and may react more to compression noise.",
          "A smaller value is cheaper and often more stable, but may miss subtle movement. The default of 160 is a balanced choice for most streams.",
          "Raise this only if small subjects are being missed after you have already lowered the main motion threshold."
        ]
      },
      motion_detect_height_field: {
        title: "Detection Height",
        body: [
          "This is the height of the reduced analysis frame paired with Detection Width.",
          "Together, width and height define how much visual detail the detector sees before it builds the motion mask.",
          "Keeping the aspect ratio roughly close to the source image avoids stretching and gives more consistent contour sizes.",
          "The default of 120 works with width 160 as a low-cost 4:3 analysis frame."
        ]
      },
      motion_bg_history_field: {
        title: "Background History",
        body: [
          "This controls how many past frames the detector remembers while building its background model.",
          "Higher values make the detector slower to 'forget' older scene information. That usually reduces flicker and short-term false positives, but it can delay adaptation to real lighting changes.",
          "Lower values make the detector adapt faster, which helps if the scene changes often, but it can also make genuine motion disappear into the background too quickly.",
          "If you see repeated false triggers from noisy cameras, try increasing this. If the detector stays confused after a scene change, lower it."
        ]
      },
      motion_bg_var_threshold_field: {
        title: "Sensitivity To Scene Changes",
        body: [
          "This is the core threshold used by the background subtractor to decide whether a pixel is 'different enough' from the learned background.",
          "Lower values make the detector more sensitive. It will pick up subtle movement faster, but it will also react more to video noise, timestamp flicker, and tiny exposure shifts.",
          "Higher values make the detector less sensitive. That reduces false positives, but requires more obvious movement before anything is counted.",
          "If motion is triggering too easily, raise this first. If the detector misses clear movement, lower it slightly."
        ]
      },
      motion_bg_learning_rate_field: {
        title: "Background Adaptation Speed",
        body: [
          "This controls how quickly the detector updates its idea of what the normal background looks like.",
          "A higher value means the background model changes quickly. That helps the detector recover faster from lighting changes, but moving objects can become 'normal' too soon.",
          "A lower value means the detector changes slowly. That makes it more stable, but it can stay upset for longer after a sudden lighting shift.",
          "Use lower values for stable cameras. Use slightly higher values only if the scene changes often and the detector stays noisy for too long."
        ]
      },
      motion_min_contour_area_pct_field: {
        title: "Minimum Movement Size",
        body: [
          "After the motion mask is built, the server looks for moving blobs. This setting ignores blobs smaller than a percentage of the analysis frame.",
          "Lower values let tiny movements count, which is useful for distant subjects, but they can also let compression blocks, rain, insects, or timestamp edges trigger motion.",
          "Higher values ignore small moving regions and require larger moving shapes, which is usually better for reliability.",
          "If you are getting lots of tiny false triggers, increase this. If small real subjects are being ignored, decrease it."
        ]
      },
      motion_warmup_frames_field: {
        title: "Startup Stabilization",
        body: [
          "When a stream or detector starts, the background model needs a few frames to settle. During this warm-up period, motion is measured but intentionally ignored.",
          "A higher value reduces startup false positives, especially on noisy RTSP streams or after reconnects.",
          "A lower value makes motion alerts become active sooner, but can cause immediate false triggers while the detector is still learning the scene.",
          "If you see motion trigger right after reconnecting to a stream, increase this setting."
        ]
      },
      voice: {
        title: "Voice Alerts",
        body: [
          "This section controls spoken motion alerts through Home Assistant.",
          "You can restrict alerts to daytime hours, set a cooldown, and choose direct TTS or webhook delivery.",
          "Per-stream voice routing is still controlled in the stream list."
        ]
      },
      firmware: {
        title: "Firmware Flashing",
        body: [
          "Use this section to build and flash firmware directly from this machine when the board is attached over USB.",
          "Board, Wi-Fi, and server details are written into the generated firmware before flashing.",
          "This section is collapsed by default because it is usually used less often than stream controls."
        ]
      },
      webflash: {
        title: "Browser Flash",
        body: [
          "Browser Flash prepares a manifest and firmware payload for esp-web-tools.",
          "Use Chrome or Edge on the machine physically connected to the device, then click Prepare + Flash.",
          "This is the preferred route for container deployments where local serial access is limited."
        ]
      }
    };

    function loadSectionState() {
      try {
        const raw = localStorage.getItem(SECTION_STATE_KEY);
        const parsed = raw ? JSON.parse(raw) : {};
        return Object.assign({}, DEFAULT_SECTION_STATE, parsed || {});
      } catch (err) {
        return Object.assign({}, DEFAULT_SECTION_STATE);
      }
    }

    let sectionState = loadSectionState();

    function saveSectionState() {
      try {
        localStorage.setItem(SECTION_STATE_KEY, JSON.stringify(sectionState));
      } catch (err) {
      }
    }

    function applySectionState() {
      document.querySelectorAll(".section-card[data-section]").forEach((card) => {
        const key = card.dataset.section;
        const open = Object.prototype.hasOwnProperty.call(sectionState, key) ? !!sectionState[key] : true;
        card.classList.toggle("is-open", open);
        const toggle = card.querySelector(".group-toggle");
        if (toggle) {
          toggle.setAttribute("aria-expanded", open ? "true" : "false");
        }
      });
    }

    function toggleSection(key) {
      sectionState[key] = !sectionState[key];
      saveSectionState();
      applySectionState();
    }

    const helpOverlay = document.getElementById("help_overlay");
    const helpTitleEl = document.getElementById("help_title");
    const helpBodyEl = document.getElementById("help_body");

    function openHelp(key) {
      const entry = HELP_CONTENT[key];
      if (!entry || !helpOverlay || !helpTitleEl || !helpBodyEl) return;
      helpTitleEl.textContent = entry.title;
      helpBodyEl.innerHTML = (entry.body || []).map((line) => `<div>${line}</div>`).join("");
      helpOverlay.hidden = false;
      document.body.style.overflow = "hidden";
    }

    function closeHelp() {
      if (!helpOverlay) return;
      helpOverlay.hidden = true;
      document.body.style.overflow = "";
    }

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
    function updateVoiceModeVisibility() {
      const directEnabled = !!document.getElementById("ha_direct_tts_enabled").checked;
      document.querySelectorAll(".voice-mode-direct").forEach((el) => {
        el.style.display = directEnabled ? "" : "none";
      });
      document.querySelectorAll(".voice-mode-webhook").forEach((el) => {
        el.style.display = directEnabled ? "none" : "";
      });
    }

    let isDirty = false;
    let latestRuntimeState = {};
    let latestStreamStates = [];
    const streamListEl = document.getElementById("stream_list");
    const activeStreamEl = document.getElementById("active_stream_index");
    const allFeedsGridEl = document.getElementById("all_feeds_grid");

    function refreshAdminFeedSnapshots() {
      if (!allFeedsGridEl) return;
      const stamp = Date.now();
      allFeedsGridEl.querySelectorAll(".feed-thumb[data-snapshot-base]").forEach((img) => {
        img.src = img.dataset.snapshotBase + "&ts=" + stamp;
      });
    }

    function stopAdminFeedStreams() {
      if (!allFeedsGridEl) return;
      allFeedsGridEl.querySelectorAll(".feed-thumb[data-snapshot-base]").forEach((img) => {
        img.removeAttribute("src");
      });
    }

    function syncAdminFeedStreams() {
      if (pageIsVisible()) {
        refreshAdminFeedSnapshots();
      } else {
        stopAdminFeedStreams();
      }
    }

    function renderAllFeedsGrid(streams) {
      if (!allFeedsGridEl) return;
      allFeedsGridEl.innerHTML = "";
      if (!streams || streams.length === 0) {
        const empty = document.createElement("div");
        empty.className = "feed-url";
        empty.textContent = "No saved feeds yet.";
        allFeedsGridEl.appendChild(empty);
        return;
      }
      streams.forEach((stream, i) => {
        const item = document.createElement("article");
        item.className = "feed-tile";
        item.dataset.idx = String(i);
        const img = document.createElement("img");
        img.className = "feed-thumb";
        img.alt = stream.name || `Stream ${i + 1}`;
        img.dataset.idx = String(i);
        img.dataset.snapshotBase = withCid(`/preview/${i}.jpg`);
        const head = document.createElement("div");
        head.className = "feed-head";
        const route = stream.motion_voice
          ? (stream.voice_entity ? ` [voice:${stream.voice_entity}]` : " [voice]")
          : "";
        head.textContent = `${i + 1}. ${stream.name || `Stream ${i + 1}`}${route}`;
        const url = document.createElement("div");
        url.className = "feed-url";
        url.textContent = stream.url || "-";
        const stats = document.createElement("div");
        stats.className = "feed-stats";
        const sourceRow = document.createElement("div");
        sourceRow.className = "feed-stat";
        sourceRow.innerHTML = "<strong>Source:</strong> ";
        const sourceValue = document.createElement("span");
        sourceValue.className = "feed-source";
        sourceValue.textContent = stream.url || "-";
        sourceRow.appendChild(sourceValue);
        const frameRow = document.createElement("div");
        frameRow.className = "feed-stat";
        frameRow.innerHTML = "<strong>Last frame:</strong> ";
        const frameValue = document.createElement("span");
        frameValue.className = "feed-frame";
        frameValue.textContent = "-";
        frameRow.appendChild(frameValue);
        const motionRow = document.createElement("div");
        motionRow.className = "feed-stat";
        motionRow.innerHTML = "<strong>Motion detection:</strong> ";
        const motionValue = document.createElement("span");
        motionValue.className = "feed-motion";
        motionValue.textContent = "idle";
        motionRow.appendChild(motionValue);
        const voiceRow = document.createElement("div");
        voiceRow.className = "feed-stat";
        voiceRow.innerHTML = "<strong>Voice alert:</strong> ";
        const voiceValue = document.createElement("span");
        voiceValue.className = "feed-voice";
        voiceValue.textContent = "idle";
        voiceRow.appendChild(voiceValue);
        stats.appendChild(sourceRow);
        stats.appendChild(frameRow);
        stats.appendChild(motionRow);
        stats.appendChild(voiceRow);
        item.appendChild(img);
        item.appendChild(head);
        item.appendChild(url);
        item.appendChild(stats);
        allFeedsGridEl.appendChild(item);
      });
      syncAdminFeedStreams();
    }

    function getStreamRows() {
      const rows = [];
      document.querySelectorAll(".stream-row").forEach((row) => {
        const name = (row.querySelector(".stream-name").value || "").trim();
        const url = (row.querySelector(".stream-url").value || "").trim();
        const motionVoice = !!row.querySelector(".stream-motion-voice").checked;
        const voiceEntity = (row.querySelector(".stream-voice-entity").value || "").trim();
        if (url) {
          rows.push({
            name: name || `Stream ${rows.length + 1}`,
            url,
            motion_voice: motionVoice,
            voice_entity: voiceEntity,
          });
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
        const route = stream.motion_voice
          ? (stream.voice_entity ? `[voice:${stream.voice_entity}]` : "[voice:default]")
          : "";
        opt.textContent = `${i + 1}. ${stream.name} ${route}`;
        activeStreamEl.appendChild(opt);
      });
      let idx = 0;
      if (streams.length > 0) {
        idx = Math.max(0, Math.min(Number(selectedIndex || 0), streams.length - 1));
        activeStreamEl.value = String(idx);
      }
      renderAllFeedsGrid(streams);
    }

    function addStreamRow(stream, shouldRefresh) {
      const row = document.createElement("div");
      row.className = "stream-row";
      row.innerHTML = `
        <input class="stream-name" type="text" maxlength="64" placeholder="Name" value="${(stream.name || "").replace(/"/g, "&quot;")}" />
        <input class="stream-url" type="text" placeholder="rtsp://..." value="${(stream.url || "").replace(/"/g, "&quot;")}" />
        <input class="stream-voice-entity" type="text" maxlength="128" placeholder="media_player.entity (optional)" value="${(stream.voice_entity || "").replace(/"/g, "&quot;")}" />
        <label class="stream-voice-toggle"><input class="stream-motion-voice" type="checkbox" ${stream.motion_voice ? "checked" : ""} /> Voice</label>
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
        addStreamRow({ name: "Stream 1", url: "", motion_voice: false, voice_entity: "" }, false);
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
      document.getElementById("motion_enabled").checked = !!s.motion_enabled;
      document.getElementById("motion_threshold").value = (s.motion_threshold ?? 2.5);
      document.getElementById("motion_hold_ms").value = (s.motion_hold_ms ?? 4000);
      document.getElementById("motion_detect_width").value = (s.motion_detect_width ?? 160);
      document.getElementById("motion_detect_height").value = (s.motion_detect_height ?? 120);
      document.getElementById("motion_bg_history").value = (s.motion_bg_history ?? 90);
      document.getElementById("motion_bg_var_threshold").value = (s.motion_bg_var_threshold ?? 32.0);
      document.getElementById("motion_bg_learning_rate").value = (s.motion_bg_learning_rate ?? 0.02);
      document.getElementById("motion_min_contour_area_pct").value = (s.motion_min_contour_area_pct ?? 0.12);
      document.getElementById("motion_warmup_frames").value = (s.motion_warmup_frames ?? 12);
      document.getElementById("motion_audio_enabled").checked = (s.motion_audio_enabled ?? true);
      document.getElementById("motion_voice_enabled").checked = !!s.motion_voice_enabled;
      document.getElementById("motion_voice_daytime_only").checked = !!s.motion_voice_daytime_only;
      document.getElementById("motion_voice_start_hour").value = (s.motion_voice_start_hour ?? 8);
      document.getElementById("motion_voice_end_hour").value = (s.motion_voice_end_hour ?? 22);
      document.getElementById("motion_voice_cooldown_ms").value = (s.motion_voice_cooldown_ms ?? 30000);
      document.getElementById("motion_voice_message").value = (s.motion_voice_message ?? "Motion detected on {stream_name}");
      document.getElementById("motion_voice_default_entity").value = (s.motion_voice_default_entity || "media_player.library_pair");
      document.getElementById("ha_direct_tts_enabled").checked = (s.ha_direct_tts_enabled ?? true);
      document.getElementById("ha_base_url").value = (s.ha_base_url ?? "");
      document.getElementById("ha_tts_entity").value = (s.ha_tts_entity || "tts.google_translate_en_com");
      document.getElementById("ha_webhook_url").value = (s.ha_webhook_url ?? "");
      document.getElementById("ha_bearer_token").value = (s.ha_bearer_token ?? "");
      document.getElementById("ha_webhook_secret").value = (s.ha_webhook_secret ?? "");
      updateVoiceModeVisibility();
      setStreams(s.streams || [], s.active_stream_index || 0);
    }

    function updateState(st, streamStates) {
      st = st || {};
      latestRuntimeState = st;
      latestStreamStates = Array.isArray(streamStates) ? streamStates : [];
      const byIdx = new Map();
      latestStreamStates.forEach((entry) => {
        byIdx.set(Number(entry.idx), entry || {});
      });
      document.querySelectorAll(".feed-tile").forEach((item) => {
        const idx = Number(item.dataset.idx || 0);
        const entry = byIdx.get(idx) || {};
        const sourceEl = item.querySelector(".feed-source");
        const frameEl = item.querySelector(".feed-frame");
        const motionEl = item.querySelector(".feed-motion");
        const voiceEl = item.querySelector(".feed-voice");
        const fallbackSource = (item.querySelector(".feed-url") || {}).textContent || "-";
        if (sourceEl) {
          sourceEl.textContent = entry.source || fallbackSource;
        }
        if (frameEl) {
          frameEl.textContent = entry.last_frame_ms ? new Date(Number(entry.last_frame_ms)).toLocaleTimeString() : "-";
        }
        if (motionEl) {
          motionEl.textContent = entry.motion_active
            ? `detected (${(Number(entry.motion_ratio || 0)).toFixed(2)}%)`
            : "idle";
        }
        if (voiceEl) {
          const voiceErr = entry.voice_last_error || "";
          if (voiceErr) {
            voiceEl.textContent = `error: ${voiceErr}`;
          } else if (entry.voice_last_sent_ms) {
            const speaker = entry.voice_last_speaker_entity || "default";
            voiceEl.textContent = `sent -> ${speaker} @ ${new Date(Number(entry.voice_last_sent_ms)).toLocaleTimeString()}`;
          } else {
            voiceEl.textContent = "idle";
          }
        }
      });
      const runtimeStateEl = document.getElementById("runtime_status_state");
      const runtimeStreamEl = document.getElementById("runtime_status_stream");
      const runtimeSourceEl = document.getElementById("runtime_status_source");
      const runtimeFrameEl = document.getElementById("runtime_status_frame");
      const runtimeMotionEl = document.getElementById("runtime_status_motion");
      const runtimeErrorEl = document.getElementById("runtime_status_error");
      if (runtimeStateEl) runtimeStateEl.textContent = st.status || "-";
      if (runtimeStreamEl) runtimeStreamEl.textContent = st.active_stream_name || "-";
      if (runtimeSourceEl) runtimeSourceEl.textContent = st.source || "-";
      if (runtimeFrameEl) runtimeFrameEl.textContent = st.last_frame_ms ? new Date(Number(st.last_frame_ms)).toLocaleTimeString() : "-";
      if (runtimeMotionEl) {
        runtimeMotionEl.textContent = st.motion_active
          ? `detected (${(Number(st.motion_ratio || 0)).toFixed(2)}%)`
          : "idle";
      }
      if (runtimeErrorEl) runtimeErrorEl.textContent = st.last_error || "-";
    }

    function hydrate(data) {
      const s = data.settings;
      if (!isDirty && s) {
        applySettingsToForm(s);
      }
      const nextState = ("state" in data) ? (data.state || {}) : latestRuntimeState;
      const nextStreamStates = ("stream_states" in data) ? (data.stream_states || []) : latestStreamStates;
      updateState(nextState, nextStreamStates);
    }

    async function loadSettings() {
      const res = await fetch(withCid("/api/settings"));
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
        swap_rb: document.getElementById("swap_rb").checked,
        motion_enabled: document.getElementById("motion_enabled").checked,
        motion_threshold: Number(document.getElementById("motion_threshold").value),
        motion_hold_ms: Number(document.getElementById("motion_hold_ms").value),
        motion_detect_width: Number(document.getElementById("motion_detect_width").value),
        motion_detect_height: Number(document.getElementById("motion_detect_height").value),
        motion_bg_history: Number(document.getElementById("motion_bg_history").value),
        motion_bg_var_threshold: Number(document.getElementById("motion_bg_var_threshold").value),
        motion_bg_learning_rate: Number(document.getElementById("motion_bg_learning_rate").value),
        motion_min_contour_area_pct: Number(document.getElementById("motion_min_contour_area_pct").value),
        motion_warmup_frames: Number(document.getElementById("motion_warmup_frames").value),
        motion_audio_enabled: document.getElementById("motion_audio_enabled").checked,
        motion_voice_enabled: document.getElementById("motion_voice_enabled").checked,
        motion_voice_daytime_only: document.getElementById("motion_voice_daytime_only").checked,
        motion_voice_start_hour: Number(document.getElementById("motion_voice_start_hour").value),
        motion_voice_end_hour: Number(document.getElementById("motion_voice_end_hour").value),
        motion_voice_cooldown_ms: Number(document.getElementById("motion_voice_cooldown_ms").value),
        motion_voice_message: document.getElementById("motion_voice_message").value.trim(),
        motion_voice_default_entity: document.getElementById("motion_voice_default_entity").value.trim(),
        ha_direct_tts_enabled: document.getElementById("ha_direct_tts_enabled").checked,
        ha_base_url: document.getElementById("ha_base_url").value.trim(),
        ha_tts_entity: document.getElementById("ha_tts_entity").value.trim(),
        ha_webhook_url: document.getElementById("ha_webhook_url").value.trim(),
        ha_bearer_token: document.getElementById("ha_bearer_token").value.trim(),
        ha_webhook_secret: document.getElementById("ha_webhook_secret").value.trim()
      };
      const res = await fetch(withCid("/api/settings"), {
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
      hydrate({settings: data.settings});
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
      const res = await fetch(withCid("/api/settings"), {
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
      hydrate({settings: data.settings});
      setStatus(`Preset applied: ${kind}`, "ok");
    }

    document.getElementById("save_btn").addEventListener("click", saveSettings);
    document.getElementById("refresh_btn").addEventListener("click", loadSettings);
    document.getElementById("open_video_feeds_btn").addEventListener("click", () => {
      window.open("/video-feeds", "_blank");
    });
    document.getElementById("preset_fps_btn").addEventListener("click", () => applyPreset("max_fps"));
    document.getElementById("preset_quality_btn").addEventListener("click", () => applyPreset("best_quality"));
    document.getElementById("ha_direct_tts_enabled").addEventListener("change", () => {
      updateVoiceModeVisibility();
      isDirty = true;
    });
    document.getElementById("add_stream_btn").addEventListener("click", () => {
      addStreamRow({ name: `Stream ${document.querySelectorAll(".stream-row").length + 1}`, url: "", motion_voice: false, voice_entity: "" }, true);
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
    streamListEl.addEventListener("change", (e) => {
      if (e.target && e.target.classList.contains("stream-motion-voice")) {
        refreshActiveStreamOptions(activeStreamEl.value || 0);
        isDirty = true;
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
      if (e.target.classList && (e.target.classList.contains("stream-name") || e.target.classList.contains("stream-url") || e.target.classList.contains("stream-voice-entity"))) {
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
    document.querySelectorAll("[data-toggle-section]").forEach((el) => {
      el.addEventListener("click", () => toggleSection(el.dataset.toggleSection));
    });
    document.querySelectorAll(".info-btn[data-help-key]").forEach((el) => {
      el.addEventListener("click", () => openHelp(el.dataset.helpKey));
    });
    document.getElementById("help_close_btn").addEventListener("click", closeHelp);
    if (helpOverlay) {
      helpOverlay.addEventListener("click", (e) => {
        if (e.target === helpOverlay) {
          closeHelp();
        }
      });
    }
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && helpOverlay && !helpOverlay.hidden) {
        closeHelp();
      }
    });
    applySectionState();

    function populateFirmwareDefaultsFromSettings() {
      const url = document.getElementById("rtsp_url").value || "";
      const m = url.match(/@([^:/]+)(?::\d+)?\//);
      document.getElementById("fw_server_host").value = (m && m[1]) ? m[1] : "192.168.1.16";
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
        board: document.getElementById("webflash_board").value,
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
      ["fw_board","fw_flavor","fw_upload_port","fw_ssid","fw_password","fw_server_host","fw_server_port"].forEach((id) => {
        document.getElementById(id).disabled = !supported || running;
      });
    }

    async function startFlash() {
      const payload = {
        board: document.getElementById("fw_board").value,
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
    document.getElementById("fw_board").addEventListener("change", (e) => {
      document.getElementById("webflash_board").value = e.target.value;
    });
    document.getElementById("webflash_board").addEventListener("change", (e) => {
      document.getElementById("fw_board").value = e.target.value;
    });
    document.getElementById("webflash_prepare_audio_btn").addEventListener("click", () => prepareWebflash("audio_on").catch(() => setWebflashStatus("Failed to prepare web flash", "err")));
    document.getElementById("webflash_prepare_no_audio_btn").addEventListener("click", () => prepareWebflash("no_audio").catch(() => setWebflashStatus("Failed to prepare web flash", "err")));

    loadSettings().catch(() => setStatus("Failed to load settings", "err"));
    populateFirmwareDefaultsFromSettings();
    loadFlashStatus().catch(() => setFlashStatus("Failed to load flash status", "err"));
    setInterval(() => {
      if (!pageIsVisible()) return;
      refreshAdminFeedSnapshots();
    }, 350);
    setInterval(() => {
      if (!pageIsVisible()) return;
      fetch(withCid("/api/settings"))
        .then(r => r.json())
        .then(d => updateState(d.state, d.stream_states || []))
        .catch(() => {});
    }, 2000);
    setInterval(() => {
      if (!pageIsVisible()) return;
      loadFlashStatus().catch(() => {});
    }, 5000);
    document.addEventListener("visibilitychange", () => {
      syncAdminFeedStreams();
      if (!pageIsVisible()) return;
      fetch(withCid("/api/settings"))
        .then(r => r.json())
        .then(d => updateState(d.state, d.stream_states || []))
        .catch(() => {});
      loadFlashStatus().catch(() => {});
    });
  </script>
</body>
</html>""",
        app_version=APP_VERSION
    )


@app.route("/video-feeds")
def video_feeds_ui():
    return render_template_string(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Video Feeds</title>
  <style>
    :root {
      --bg-1: #0b1320;
      --bg-2: #16243a;
      --panel: rgba(255,255,255,0.9);
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
        radial-gradient(1100px 700px at 90% -10%, #265685, transparent 70%),
        radial-gradient(1100px 700px at -10% 110%, #1c3a59, transparent 70%),
        linear-gradient(160deg, var(--bg-1), var(--bg-2));
      min-height: 100vh;
      padding: 22px;
    }
    .wrap {
      max-width: 1400px;
      margin: 0 auto;
      display: grid;
      gap: 18px;
    }
    .card {
      background: var(--panel);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 18px 40px rgba(0,0,0,0.25);
      backdrop-filter: blur(6px);
    }
    .head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      flex-wrap: wrap;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 28px;
      letter-spacing: .2px;
    }
    p { margin: 0; color: var(--muted); }
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
    .actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }
    .button-link, button {
      border: 0;
      border-radius: 10px;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    .button-link {
      background: #e2e8f0;
      color: #1f2937;
    }
    button {
      background: var(--accent);
      color: #fff;
    }
    .status {
      padding: 10px 12px;
      border-radius: 12px;
      background: #eef6ff;
      font-size: 13px;
      color: #355070;
    }
    .status.ok { color: var(--ok); }
    .status.warn { color: var(--warn); }
    .status.err { color: var(--error); }
    .feeds-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    .feed-tile {
      width: 100%;
      display: block;
      text-align: left;
      border: 1px solid #d6deea;
      border-radius: 14px;
      background: #f8fbff;
      padding: 0;
      overflow: hidden;
      color: #1f2937;
      box-shadow: 0 8px 20px rgba(8, 21, 37, 0.06);
    }
    .feed-tile.active {
      border-color: #0d8fdf;
      box-shadow: inset 0 0 0 2px #0d8fdf, 0 10px 24px rgba(13, 143, 223, 0.14);
      background: #eaf5ff;
    }
    .feed-thumb {
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      object-position: center;
      display: block;
      background: #0c1420;
      border-bottom: 1px solid #d7e2ef;
    }
    .feed-head {
      font-size: 14px;
      font-weight: 800;
      color: #1f2f46;
      padding: 10px 10px 0;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .feed-url {
      font-size: 11px;
      color: #58687d;
      padding: 4px 10px 10px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .feed-stats {
      border-top: 1px solid #d7e2ef;
      background: #f2f7fd;
      padding: 8px 10px 10px;
      display: grid;
      gap: 5px;
    }
    .feed-stat {
      font-size: 12px;
      color: #475569;
      line-height: 1.3;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    @media (max-width: 900px) {
      .head {
        flex-direction: column;
        align-items: stretch;
      }
      .feeds-grid {
        grid-template-columns: 1fr;
      }
      .actions {
        width: 100%;
      }
      .button-link, button {
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="card">
      <div class="head">
        <div>
          <div class="pill">Version {{ app_version }}</div>
          <h1>Video Feeds</h1>
          <p>Saved feeds only. This page is a read-only live monitor for every configured feed.</p>
        </div>
        <div class="actions">
          <a class="button-link" href="/admin">Back To Control</a>
          <button type="button" id="refresh_feeds_btn">Refresh Now</button>
        </div>
      </div>
    </section>

    <section class="card">
      <div id="video_feeds_status" class="status">Loading feeds...</div>
    </section>

    <section class="card">
      <div id="video_feeds_grid" class="feeds-grid"></div>
    </section>
  </div>

  <script>
    const statusEl = document.getElementById("video_feeds_status");
    const gridEl = document.getElementById("video_feeds_grid");
    let renderedSignature = "";

    function pageIsVisible() {
      return !document.hidden;
    }

    function setStatus(message, mode) {
      statusEl.textContent = message;
      statusEl.className = "status " + (mode || "");
    }

    function streamsSignature(streams) {
      return JSON.stringify((streams || []).map((stream) => [
        stream.name || "",
        stream.url || "",
        !!stream.motion_voice,
        stream.voice_entity || ""
      ]));
    }

    function stopVideoFeedStreams() {
      gridEl.querySelectorAll(".feed-thumb[data-stream-src]").forEach((img) => {
        img.removeAttribute("src");
      });
    }

    function startVideoFeedStreams() {
      gridEl.querySelectorAll(".feed-thumb[data-stream-src]").forEach((img) => {
        if (img.getAttribute("src")) return;
        img.src = img.dataset.streamSrc;
      });
    }

    function syncVideoFeedStreams() {
      if (pageIsVisible()) {
        startVideoFeedStreams();
      } else {
        stopVideoFeedStreams();
      }
    }

    function renderFeeds(streams) {
      const signature = streamsSignature(streams);
      if (signature === renderedSignature) {
        syncVideoFeedStreams();
        return;
      }
      renderedSignature = signature;
      gridEl.innerHTML = "";
      if (!streams || streams.length === 0) {
        const empty = document.createElement("div");
        empty.className = "feed-stat";
        empty.textContent = "No saved feeds configured.";
        gridEl.appendChild(empty);
        return;
      }
      streams.forEach((stream, i) => {
        const item = document.createElement("article");
        item.className = "feed-tile";
        item.dataset.idx = String(i);

        const img = document.createElement("img");
        img.className = "feed-thumb";
        img.alt = stream.name || `Stream ${i + 1}`;
        img.dataset.streamSrc = `/preview/${i}.mjpg`;

        const head = document.createElement("div");
        head.className = "feed-head";
        const route = stream.motion_voice
          ? (stream.voice_entity ? ` [voice:${stream.voice_entity}]` : " [voice]")
          : "";
        head.textContent = `${i + 1}. ${stream.name || `Stream ${i + 1}`}${route}`;

        const url = document.createElement("div");
        url.className = "feed-url";
        url.textContent = stream.url || "-";

        const stats = document.createElement("div");
        stats.className = "feed-stats";

        const sourceRow = document.createElement("div");
        sourceRow.className = "feed-stat";
        sourceRow.innerHTML = "<strong>Source:</strong> ";
        const sourceValue = document.createElement("span");
        sourceValue.className = "feed-source";
        sourceValue.textContent = stream.url || "-";
        sourceRow.appendChild(sourceValue);

        const frameRow = document.createElement("div");
        frameRow.className = "feed-stat";
        frameRow.innerHTML = "<strong>Last frame:</strong> ";
        const frameValue = document.createElement("span");
        frameValue.className = "feed-frame";
        frameValue.textContent = "-";
        frameRow.appendChild(frameValue);

        const motionRow = document.createElement("div");
        motionRow.className = "feed-stat";
        motionRow.innerHTML = "<strong>Motion detection:</strong> ";
        const motionValue = document.createElement("span");
        motionValue.className = "feed-motion";
        motionValue.textContent = "idle";
        motionRow.appendChild(motionValue);

        const voiceRow = document.createElement("div");
        voiceRow.className = "feed-stat";
        voiceRow.innerHTML = "<strong>Voice alert:</strong> ";
        const voiceValue = document.createElement("span");
        voiceValue.className = "feed-voice";
        voiceValue.textContent = "idle";
        voiceRow.appendChild(voiceValue);

        stats.appendChild(sourceRow);
        stats.appendChild(frameRow);
        stats.appendChild(motionRow);
        stats.appendChild(voiceRow);

        item.appendChild(img);
        item.appendChild(head);
        item.appendChild(url);
        item.appendChild(stats);
        gridEl.appendChild(item);
      });
      syncVideoFeedStreams();
    }

    function updateStates(streamStates) {
      const latestStreamStates = Array.isArray(streamStates) ? streamStates : [];
      const byIdx = new Map();
      latestStreamStates.forEach((entry) => {
        byIdx.set(Number(entry.idx), entry || {});
      });
      document.querySelectorAll(".feed-tile").forEach((item) => {
        const idx = Number(item.dataset.idx || 0);
        const entry = byIdx.get(idx) || {};
        const sourceEl = item.querySelector(".feed-source");
        const frameEl = item.querySelector(".feed-frame");
        const motionEl = item.querySelector(".feed-motion");
        const voiceEl = item.querySelector(".feed-voice");
        const fallbackSource = (item.querySelector(".feed-url") || {}).textContent || "-";
        if (sourceEl) {
          sourceEl.textContent = entry.source || fallbackSource;
        }
        if (frameEl) {
          frameEl.textContent = entry.last_frame_ms ? new Date(Number(entry.last_frame_ms)).toLocaleTimeString() : "-";
        }
        if (motionEl) {
          motionEl.textContent = entry.motion_active
            ? `detected (${(Number(entry.motion_ratio || 0)).toFixed(2)}%)`
            : "idle";
        }
        if (voiceEl) {
          const voiceErr = entry.voice_last_error || "";
          if (voiceErr) {
            voiceEl.textContent = `error: ${voiceErr}`;
          } else if (entry.voice_last_sent_ms) {
            const speaker = entry.voice_last_speaker_entity || "default";
            voiceEl.textContent = `sent -> ${speaker} @ ${new Date(Number(entry.voice_last_sent_ms)).toLocaleTimeString()}`;
          } else {
            voiceEl.textContent = "idle";
          }
        }
      });
    }

    async function refreshFeeds() {
      const res = await fetch("/api/settings");
      const data = await res.json();
      const settings = data.settings || {};
      const streams = settings.streams || [];
      const streamStates = data.stream_states || [];
      renderFeeds(streams);
      updateStates(streamStates);
      if (streams.length === 0) {
        setStatus("No saved feeds configured yet.", "warn");
      } else {
        const streamingCount = streamStates.filter((entry) => (entry && entry.status) === "streaming").length;
        const suffix = streamingCount ? ` | ${streamingCount} streaming` : "";
        setStatus(`Showing ${streams.length} saved feed${streams.length === 1 ? "" : "s"}${suffix}`, "ok");
      }
    }

    document.getElementById("refresh_feeds_btn").addEventListener("click", () => {
      refreshFeeds().catch(() => setStatus("Failed to load feeds", "err"));
    });

    refreshFeeds().catch(() => setStatus("Failed to load feeds", "err"));
    setInterval(() => {
      if (!pageIsVisible()) return;
      refreshFeeds().catch(() => {});
    }, 2000);
    document.addEventListener("visibilitychange", () => {
      syncVideoFeedStreams();
      if (!pageIsVisible()) return;
      refreshFeeds().catch(() => {});
    });
  </script>
</body>
</html>""",
        app_version=APP_VERSION
    )


if __name__ == '__main__':
    init_video_source()
    app.run(host='0.0.0.0', port=VIDEO_SERVER_PORT)
