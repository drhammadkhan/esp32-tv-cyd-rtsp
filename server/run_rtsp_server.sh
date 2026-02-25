#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RTSP_URL_DEFAULT='rtsp://user:pass@camera:8554/Streaming/Channels/102'
RTSP_URL="${RTSP_URL:-$RTSP_URL_DEFAULT}"
VIDEO_SERVER_PORT="${VIDEO_SERVER_PORT:-8124}"

while true; do
  echo "$(date '+%Y-%m-%d %H:%M:%S') starting server on :${VIDEO_SERVER_PORT}" >> /tmp/esp32-tv-supervisor.log
  RTSP_URL="$RTSP_URL" VIDEO_SERVER_PORT="$VIDEO_SERVER_PORT" ./venv/bin/python ./app.py >> /tmp/esp32-tv-server.log 2>&1 || true
  echo "$(date '+%Y-%m-%d %H:%M:%S') server exited, restarting in 1s" >> /tmp/esp32-tv-supervisor.log
  sleep 1
done
