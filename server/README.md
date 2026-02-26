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

The `/admin` UI now supports multiple saved RTSP streams. In RTSP mode:
- each saved stream is exposed as a channel
- CYD channel up/down (or touch swipe on CYD, right-to-left for next) switches streams
- settings are persisted in `server/cache/settings.json`

Optional restart loop:

```bash
./run_rtsp_server.sh
```

# Docker

## Quick Deploy (Prebuilt Image)

```bash
cd server
cp .env.example .env
# edit .env with your RTSP URL and tuning values
docker compose -f docker-compose.ghcr.yml up -d
```

Container image:

```bash
docker pull ghcr.io/drhammadkhan/esp32-tv-cyd-rtsp-server:latest
```

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

## CasaOS / Similar UI Container Hosts

1. Create host folders:
```bash
mkdir -p /DATA/AppData/esp32-tv-server/movies
mkdir -p /DATA/AppData/esp32-tv-server/cache
```

2. In CasaOS:
- Open `App Store` -> `Custom Install` -> `Import compose`
- Paste this stack:

```yaml
services:
  esp32-tv-server:
    image: ghcr.io/drhammadkhan/esp32-tv-cyd-rtsp-server:latest
    container_name: esp32-tv-server
    ports:
      - "8124:8124"
    environment:
      VIDEO_SERVER_PORT: "8124"
      RTSP_URL: "rtsp://user:pass@camera:8554/Streaming/Channels/102"
      JPEG_QUALITY: "76"
      SWAP_RB: "0"
      CONTRAST: "1.14"
      BRIGHTNESS: "-6"
      SATURATION: "1.22"
      STREAM_PRESET: "balanced"
    volumes:
      - /DATA/AppData/esp32-tv-server/movies:/app/movies
      - /DATA/AppData/esp32-tv-server/cache:/app/cache
    restart: unless-stopped
```

3. After deploy:
- Open `http://<casaos-host-ip>:8124/admin`
- Set `RTSP URL` and tuning in the web UI
- Point CYD firmware to `<casaos-host-ip>:8124`

Notes:
- Container deployment supports streaming/admin UI fully.
- Firmware USB flashing via backend `platformio` is disabled in container runtime (no local `player/` source tree by default).
- Workaround included: use **Browser USB Flash** in `/admin` on a Chrome/Edge client that has the CYD connected over USB. This uses Web Serial from the browser and works with CasaOS/container deployments.

## CasaOS App Store Source (One-Click App Entry)

This repo includes a CasaOS app definition at:

- `Apps/ESP32-TV-Server/docker-compose.yml`

If your CasaOS version supports custom app stores from GitHub, add this repo:

- `https://github.com/drhammadkhan/esp32-tv-cyd-rtsp`

Then install `ESP32 TV Server` directly from the custom store.

If your CasaOS version only supports compose import, use this raw manifest URL:

- `https://raw.githubusercontent.com/drhammadkhan/esp32-tv-cyd-rtsp/main/Apps/ESP32-TV-Server/docker-compose.yml`

## GitHub Container Publishing

This repo includes a workflow that publishes the server image to GHCR on pushes to `main`:

- `.github/workflows/publish_server_container.yml`

Published image name:

- `ghcr.io/drhammadkhan/esp32-tv-cyd-rtsp-server`
