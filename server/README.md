# Setup

Make sure you have python3 installed.

Then create a virtual environment:

```bash
python3 -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

Install the requirements:

```bash
pip install -r requirements.txt
```

# Adding Vidoes

Place videos in the `movies` directory. The videos should be in the mp4 format and should be fairly short (around 2-3 minutes maximum).

If you change any parameters for the video (e.g. the frame size), just detele the `cache` directory and it will be regenerated when you start the server.

# Running

To run the application:

```bash
source venv/bin/activate
python3 app.py
```

# RTSP Mode (Used for CYD)

You can run the server as an RTSP bridge instead of local `movies/` playback.

```bash
RTSP_URL='rtsp://user:pass@camera:8554/Streaming/Channels/102' \
VIDEO_SERVER_PORT=8124 \
JPEG_QUALITY=76 \
SWAP_RB=0 \
CONTRAST=1.14 \
BRIGHTNESS=-6 \
SATURATION=1.22 \
./venv/bin/python ./app.py
```

Optional restart loop:

```bash
./run_rtsp_server.sh
```

# Docker

Prebuilt image (from GitHub Container Registry):

```bash
docker pull ghcr.io/drhammadkhan/esp32-tv-cyd-rtsp-server:latest
```

Build image:

```bash
cd server
docker build -t esp32-tv-server .
```

Run with RTSP mode (recommended for CYD):

```bash
docker run --rm -it \
  -p 8124:8124 \
  -e VIDEO_SERVER_PORT=8124 \
  -e RTSP_URL='rtsp://user:pass@camera:8554/Streaming/Channels/102' \
  -e JPEG_QUALITY=76 \
  -e SWAP_RB=0 \
  -e CONTRAST=1.14 \
  -e BRIGHTNESS=-6 \
  -e SATURATION=1.22 \
  -e STREAM_PRESET=balanced \
  -v "$(pwd)/cache:/app/cache" \
  esp32-tv-server
```

Run with movie mode (no RTSP_URL):

```bash
docker run --rm -it \
  -p 8124:8124 \
  -e VIDEO_SERVER_PORT=8124 \
  -v "$(pwd)/movies:/app/movies" \
  -v "$(pwd)/cache:/app/cache" \
  esp32-tv-server
```

Open the web UI:

```text
http://<server-ip>:8124/admin
```

Optional compose setup:

```bash
cd server
docker compose up -d --build
```

## GitHub Container Publishing

This repo includes a workflow that publishes the server image to GHCR on pushes to `main`:

- `.github/workflows/publish_server_container.yml`

Published image name:

- `ghcr.io/drhammadkhan/esp32-tv-cyd-rtsp-server`
