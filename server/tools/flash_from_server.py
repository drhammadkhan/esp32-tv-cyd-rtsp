#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request


CHIP_FAMILY_MAP = {
    "ESP32": "esp32",
    "ESP32-S2": "esp32s2",
    "ESP32-S3": "esp32s3",
    "ESP32-C3": "esp32c3",
    "ESP32-C6": "esp32c6",
    "ESP32-H2": "esp32h2",
}


def _find_boot_app0() -> str:
    override = os.environ.get("ESP_BOOT_APP0_BIN", "").strip()
    if override:
        p = Path(override).expanduser()
        if p.exists():
            return str(p)
    candidates = [
        Path.home() / ".platformio" / "packages" / "framework-arduinoespressif32" / "tools" / "partitions" / "boot_app0.bin",
        Path.home() / ".platformio" / "packages" / "framework-espidf" / "components" / "bootloader" / "subproject" / "main" / "bootloader.bin",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return ""


def _http_json(method: str, url: str, payload=None, timeout=30):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body.decode("utf-8"))


def _download(url: str, dst_path: str, timeout=60):
    req = urllib.request.Request(url, headers={"Accept": "*/*"}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    with open(dst_path, "wb") as f:
        f.write(data)


def _normalize_base(url: str) -> str:
    value = url.strip()
    if not value.startswith("http://") and not value.startswith("https://"):
        value = "http://" + value
    return value.rstrip("/")


def main():
    parser = argparse.ArgumentParser(
        description="Flash ESP firmware over USB without browser by using the server's /api/webflash endpoints."
    )
    parser.add_argument("--server", default="http://127.0.0.1:8124", help="Base URL of running server")
    parser.add_argument("--board", required=True, choices=["cyd", "ttgo_tdisplay", "esp32_s3_2p8"])
    parser.add_argument("--flavor", default="no_audio", choices=["audio_on", "no_audio"])
    parser.add_argument("--ssid", required=True)
    parser.add_argument("--password", default="")
    parser.add_argument("--server-host", required=True, help="IP/host the ESP should connect to")
    parser.add_argument("--server-port", type=int, default=8124)
    parser.add_argument("--port", required=True, help="USB serial port, e.g. /dev/cu.usbmodem11101")
    parser.add_argument("--baud", type=int, default=460800)
    parser.add_argument("--chip", default="", help="Optional esptool chip override (e.g. esp32s3)")
    parser.add_argument("--erase-first", action="store_true", help="Run erase_flash before write_flash")
    args = parser.parse_args()

    base_url = _normalize_base(args.server)
    prepare_url = base_url + "/api/webflash/prepare"

    payload = {
        "board": args.board,
        "flavor": args.flavor,
        "ssid": args.ssid,
        "password": args.password,
        "server_host": args.server_host,
        "server_port": args.server_port,
    }

    try:
        prepared = _http_json("POST", prepare_url, payload=payload)
    except urllib.error.HTTPError as ex:
        body = ex.read().decode("utf-8", errors="replace")
        print(f"ERROR: prepare request failed ({ex.code}): {body}", file=sys.stderr)
        return 2
    except Exception as ex:
        print(f"ERROR: prepare request failed: {ex}", file=sys.stderr)
        return 2

    if not prepared.get("ok"):
        print(f"ERROR: {prepared.get('error', 'unknown prepare error')}", file=sys.stderr)
        return 2

    manifest_url = urllib.parse.urljoin(base_url + "/", prepared["manifest_url"].lstrip("/"))
    try:
        manifest = _http_json("GET", manifest_url, payload=None)
    except Exception as ex:
        print(f"ERROR: failed to fetch manifest: {ex}", file=sys.stderr)
        return 2

    builds = manifest.get("builds") or []
    if not builds:
        print("ERROR: manifest has no builds", file=sys.stderr)
        return 2
    build = builds[0]

    chip = args.chip.strip()
    if not chip:
        chip_family = str(build.get("chipFamily", "")).strip()
        chip = CHIP_FAMILY_MAP.get(chip_family, "").strip()
    if not chip:
        print(
            "ERROR: could not infer chip from manifest; pass --chip explicitly (e.g. esp32 or esp32s3)",
            file=sys.stderr,
        )
        return 2

    parts = build.get("parts") or []
    if not parts:
        print("ERROR: manifest has no parts", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="esp32-webflash-") as tmp:
        flash_parts = []
        print(f"Preparing files in {tmp}")
        for idx, part in enumerate(parts):
            offset = int(part["offset"])
            part_url = urllib.parse.urljoin(base_url + "/", str(part["path"]).lstrip("/"))
            filename = os.path.basename(urllib.parse.urlparse(part_url).path) or f"part-{idx}.bin"
            local_path = os.path.join(tmp, filename)
            print(f"Downloading {part_url}")
            _download(part_url, local_path)
            flash_parts.append([offset, local_path])

        # Some manifests use generic web offsets (0x1000/0x8000/0x10000) and omit boot_app0.
        # ESP32-S3 needs the bootloader at 0x0000 in this project; all chips need boot_app0 at 0xE000 after erase.
        if chip == "esp32s3":
            for item in flash_parts:
                if item[0] == 0x1000:
                    print("Adjusting ESP32-S3 bootloader offset: 0x1000 -> 0x0000")
                    item[0] = 0x0000
                    break

        has_boot_app0 = any(off == 0xE000 for off, _ in flash_parts)
        if not has_boot_app0 and chip.startswith("esp32"):
            boot_app0 = _find_boot_app0()
            if boot_app0:
                print(f"Injecting boot_app0 at 0xE000 from: {boot_app0}")
                flash_parts.append([0xE000, boot_app0])
            else:
                print(
                    "WARNING: boot_app0.bin not found; if device fails to boot, set ESP_BOOT_APP0_BIN and retry.",
                    file=sys.stderr,
                )

        flash_parts.sort(key=lambda x: x[0])
        flash_pairs = []
        for off, path in flash_parts:
            flash_pairs.extend([hex(off), path])

        if args.erase_first:
            erase_cmd = [
                sys.executable,
                "-m",
                "esptool",
                "--chip",
                chip,
                "--port",
                args.port,
                "--baud",
                str(args.baud),
                "erase_flash",
            ]
            print("Running:", " ".join(erase_cmd))
            rc = subprocess.run(erase_cmd).returncode
            if rc != 0:
                return rc

        cmd = [
            sys.executable,
            "-m",
            "esptool",
            "--chip",
            chip,
            "--port",
            args.port,
            "--baud",
            str(args.baud),
            "--before",
            "default_reset",
            "--after",
            "hard_reset",
            "write_flash",
            "-z",
            *flash_pairs,
        ]
        print("Running:", " ".join(cmd))
        return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
