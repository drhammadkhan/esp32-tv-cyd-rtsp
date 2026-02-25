[![Build Firmware](https://github.com/atomic14/esp32-tv/actions/workflows/build_firmware.yml/badge.svg)](https://github.com/atomic14/esp32-tv/actions/workflows/build_firmware.yml)

# ESP32 Video Streaming!

Yes - it actually works! Streaming video with audio over WiFi to an ESP32!

[![WiFi Streaming](https://img.youtube.com/vi/G6MROvlLeKE/0.jpg)](https://www.youtube.com/watch?v=G6MROvlLeKE)

And - playing video from an SD Card!

[![SD Card Video](https://img.youtube.com/vi/dWgjsJtlbpA/0.jpg)](https://www.youtube.com/watch?v=dWgjsJtlbpA)

There's two projects in this repo - one for the ESP32 firmware and another for the server.

The README files in each project have more details.

The server also supports Docker deployment. See `server/README.md` for `docker build`, `docker run`, and `docker compose` instructions.

# How Does It Work?

The server is pretty simple, it has a few endpoints:

- `/channel_info` - returns a list of channel lengths in audio samples
- `/frame/<int:channel_index>/<int:ms>` - returns a JPEG image for the given channel at the given time (in ms)
- `/audio/<int:channel_index>/<int:start>/<int:length>` - returns 8 bit PCM audio at 16KHz for the given channel starting from the given sample index and for the given length (in samples)

The ESP32 firmware connects to the server and requests the channel info. The video playback is locked to the audio sample playback. The audio is played back using the I2S peripheral and we use that to know how much time has elapsed to request the correct frames. This way the video and audio are always in sync.

You can get around 15 frames per second at 280x240 resolution, the main limitation is WiFi bandwidth and decoding the JPEGs.

# Support for SD Cards

In the README file for the firmware there are instructions on how to convert a video file to a MJPEG AVI file - if you've got a device with an SD Card you can use this instead of WiFi streaming.

# CYD RTSP Setup (This Branch)

This repo now includes a CYD-focused RTSP workflow with tuned color/contrast and two firmware variants.

## Firmware Variants

- `cheap-yellow-display` : audio enabled
- `cheap-yellow-display-no-audio` : audio disabled for better frame rate

Build and flash from `player/`:

```bash
python3 -m platformio run -e cheap-yellow-display -t upload
python3 -m platformio run -e cheap-yellow-display-no-audio -t upload
```

## RTSP Server Mode

Run from `server/` with environment variables:

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

Current CYD network defaults in firmware:

- SSID: `YOUR_WIFI_SSID` (or set with web flasher)
- Host: `192.168.1.100` (or set with web flasher)
- Port: `8124`
