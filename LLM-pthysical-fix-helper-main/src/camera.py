"""Reuse the supplied named-camera capture; model inputs never have a grid."""
import asyncio
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw

from . import camera_capture
from .models import BoundingBox, ImageFrame


class Camera:
    def __init__(self, artifacts: Path, fixture: Path | None = None, *,
                 camera_name: str = camera_capture.CAMERA_NAME,
                 resolution: tuple[int, int] | None = (camera_capture.WIDTH, camera_capture.HEIGHT)):
        self.artifacts = artifacts
        self.fixture = fixture
        self.camera_name, self.resolution = camera_name, resolution
        self.camera_label = camera_name

    async def capture_and_process(self) -> ImageFrame:
        return await asyncio.to_thread(self._capture)

    def _capture(self) -> ImageFrame:
        if self.fixture is None:
            result = camera_capture.take_photo(self.artifacts / "captures",
                camera_name=self.camera_name, resolution=self.resolution, camera_label=self.camera_label)
            path = Path(result["original_path"])
        else:
            directory = self.artifacts / "captures" / uuid4().hex
            directory.mkdir(parents=True)
            path = directory / "original.png"
            with Image.open(self.fixture) as image:
                image.convert("RGB").save(path)
        # Read dimensions from the exact encoded image, never a configured guess.
        with Image.open(path) as image:
            image.load()
            width, height = image.size
        if width < 2 or height < 2:
            raise ValueError("Captured image is too small to localize a component")
        return ImageFrame(path, width, height)


def save_debug(frame: ImageFrame, bbox: BoundingBox | None, label: str | None) -> Path:
    path = frame.path.with_name("debug_bbox.png")
    with Image.open(frame.path) as source:
        debug = source.convert("RGB").copy()
    if bbox is not None:
        bbox.validate_image(frame.width, frame.height)
        draw = ImageDraw.Draw(debug)
        draw.rectangle((bbox.x1, bbox.y1, bbox.x2, bbox.y2), outline="red", width=3)
        draw.text((bbox.x1, bbox.y1), label or "target", fill="red")
    debug.save(path)
    return path
