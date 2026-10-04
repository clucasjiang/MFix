"""Discover audio inputs and Windows DirectShow cameras without recording."""
import re
import subprocess
import sys
from dataclasses import dataclass

import imageio_ffmpeg


@dataclass(frozen=True)
class MicrophoneDevice:
    index: int
    name: str
    host_api: str

    @property
    def label(self) -> str:
        return f"{self.name} · {self.host_api} [{self.index}]"


@dataclass(frozen=True)
class CameraDevice:
    name: str
    input_name: str


def list_microphones() -> list[MicrophoneDevice]:
    import sounddevice as sd
    hosts = sd.query_hostapis()
    return [MicrophoneDevice(index, device["name"], hosts[device["hostapi"]]["name"])
            for index, device in enumerate(sd.query_devices()) if device["max_input_channels"] > 0]


def parse_cameras(log: str) -> list[CameraDevice]:
    cameras = []
    current = None
    for line in log.splitlines():
        device = re.search(r'"(.*)"\s+\((video|audio)\)', line)
        if device:
            current = None
            if device.group(2) == "video":
                cameras.append(CameraDevice(device.group(1), device.group(1)))
                current = len(cameras) - 1
        else:
            alternative = re.search(r'Alternative name "(.*)"', line)
            if alternative and current is not None:
                cameras[current] = CameraDevice(cameras[current].name, alternative.group(1))
                current = None
    # Friendly names stay readable; unique DirectShow paths disambiguate twins.
    return list(dict.fromkeys(cameras))


def list_cameras() -> list[CameraDevice]:
    if sys.platform != "win32":
        raise RuntimeError("Camera selection currently requires Windows DirectShow")
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-list_devices", "true",
         "-f", "dshow", "-i", "dummy"],
        capture_output=True, encoding="utf-8", errors="replace", timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    # FFmpeg deliberately exits nonzero after listing; stderr carries devices.
    cameras = parse_cameras(result.stderr)
    if not cameras and "Could not enumerate video devices" not in result.stderr and "(audio)" not in result.stderr:
        raise RuntimeError("Camera discovery failed; check Windows camera access and FFmpeg DirectShow support")
    return cameras
