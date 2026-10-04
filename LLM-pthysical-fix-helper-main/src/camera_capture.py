"""Supplied UGREEN capture program, adapted to publish only the clean image."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import imageio_ffmpeg
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAPTURES_DIR = PROJECT_ROOT / "data" / "captures"
CAMERA_NAME = "UGREEN Camera 1080P"
WIDTH, HEIGHT = 1920, 1080


def take_photo(output_root: Path = CAPTURES_DIR, *, camera_name: str = CAMERA_NAME,
               resolution: tuple[int, int] | None = (WIDTH, HEIGHT), camera_label: str | None = None) -> dict:
    """Open the external camera on each call, save one capture, then release it."""
    if sys.platform != "win32":
        raise RuntimeError("This camera capture currently requires Windows.")
    if not camera_name.strip():
        raise ValueError("Choose a camera first")
    if resolution is not None and (len(resolution) != 2 or min(resolution) < 2):
        raise ValueError("Camera resolution must contain positive width and height")
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    # Publish only after the clean image and metadata are complete.
    with TemporaryDirectory(prefix=".capture-", dir=output_root) as temporary:
        staging = Path(temporary)
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "dshow", "-rtbufsize", "64M",
        ]
        if resolution is not None:
            command.extend(["-video_size", f"{resolution[0]}x{resolution[1]}"])
        if (camera_label or camera_name) == CAMERA_NAME and resolution == (WIDTH, HEIGHT):
            # Preserve the known working UGREEN mode. Other cameras negotiate
            # their own format rather than being forced to support MJPEG.
            command.extend(["-framerate", "30", "-vcodec", "mjpeg"])
        command.extend([
            "-i", f"video={camera_name}",
            # Discard the first second to allow exposure to settle.
            "-ss", "1", "-frames:v", "1", "-q:v", "2", "-update", "1",
            str(staging / "original.jpg"),
        ])
        try:
            subprocess.run(
                command, check=True, capture_output=True, timeout=20,
                encoding="utf-8", errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"Camera capture timed out: {camera_name}. Check its connection and close other camera apps."
            ) from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"Could not capture {camera_name}. Check its connection, selected resolution, "
                f"and other apps using it.\n{error.stderr.strip()}"
            ) from error

        captured_at = datetime.now().astimezone()
        destination = output_root / (
            captured_at.strftime("%Y%m%d-%H%M%S-%f") + "-" + uuid4().hex[:8]
        )
        with Image.open(staging / "original.jpg") as original:
            original.load()
            width, height = original.size
        result = {
            "camera": camera_name,
            "captured_at": captured_at.isoformat(),
            "image_width": width,
            "image_height": height,
            "origin": "top-left",
            "pixel_index_base": 0,
            "original_path": str(destination / "original.jpg"),
            "metadata_path": str(destination / "metadata.json"),
        }
        (staging / "metadata.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        staging.rename(destination)
    return result


def main() -> int:
    try:
        result = take_photo()
    except (RuntimeError, OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
