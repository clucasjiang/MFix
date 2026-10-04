"""The Fixpoint laser rig as this app's camera and pointer.

The rig keeps its camera open for laser tracking, so the assistant takes its
photos through the rig instead of opening the camera itself. Each photo is
the tracker's own view (laser off, normally exposed), so a returned box is
already in tracker pixels: it is forwarded unchanged and the rig steers the
dot until the camera sees it inside the box.
"""
import asyncio
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .models import ImageFrame
from .pointer import PointingFailed

ROOT = Path(__file__).resolve().parent.parent
# fixpoint/ and laser_tracker.py live next to this repository unless FIXPOINT_ROOT says otherwise.
FIXPOINT_ROOT = Path(os.environ.get("FIXPOINT_ROOT", ROOT.parent)).resolve()


class RigError(RuntimeError):
    """The laser rig could not start."""


def _import_fixpoint() -> None:
    if str(FIXPOINT_ROOT) not in sys.path:
        sys.path.insert(0, str(FIXPOINT_ROOT))
    try:
        import fixpoint  # noqa: F401
    except ImportError as error:
        raise RigError(f"The fixpoint package was not found in {FIXPOINT_ROOT}. "
                           "Set FIXPOINT_ROOT to the folder that contains fixpoint/.") from error


class LaserRig:
    """Camera, gimbal, tracking controller and debug window for one app run.

    Construction blocks for a few seconds (the ESP32 resets when its port opens).
    """

    def __init__(self, *, port: str = "COM6", camera_index: int = 1, calibration: Path | None = None,
                 sim: bool = False, window: bool = True, start: tuple[float, float] = (90.0, 144.0),
                 photo_size: tuple[int, int] | None = (1920, 1080)):
        _import_fixpoint()
        from fixpoint.controller import Controller
        from fixpoint.detector import LaserDetector
        from fixpoint.rig import Rig

        if sim:
            from fixpoint.sim import SimCamera, SimGimbal, SimWorld
            world = SimWorld()
            camera = SimCamera(world).start()
            gimbal = SimGimbal(world)
            self.label = "Simulated laser rig"
        else:
            from fixpoint.camera import Camera
            from fixpoint.gimbal import SerialGimbal
            try:
                camera = Camera(camera_index).start()
            except RuntimeError as error:
                raise RigError(f"{error} Close laser_tracker.py and other camera apps.") from error
            try:
                gimbal = SerialGimbal(port)
            except Exception as error:
                camera.stop()
                raise RigError(f"Could not open the gimbal on {port}: {error}. "
                                   "Close laser_tracker.py and any serial monitor.") from error
            self.label = f"Fixpoint laser · {port} · camera {camera_index}"
        if calibration is None:
            calibration = FIXPOINT_ROOT / ("calibration_sim.json" if sim else "calibration.json")
        self.camera, self.gimbal = camera, gimbal
        self.rig = Rig(camera, gimbal, LaserDetector())
        self.controller = Controller(self.rig, camera.size, calibration, start,
                                     log_path=None if sim else FIXPOINT_ROOT / "aim_log.csv")
        # Tracking keeps its own resolution; only the assistant's photos use
        # photo_size. The simulator has no stills and uses tracking frames.
        self.photo_size = tuple(camera.size) if sim or photo_size is None else tuple(photo_size)
        self.controller.photo_size = self.photo_size
        self.last_photo_size: tuple[int, int] | None = None  # pixels the latest box refers to
        self.controller.start()
        self._stop_window = threading.Event()
        self._window = None
        if window:
            import laser_tracker
            # Space is push-to-talk in the assistant, so it must not toggle the laser there.
            self._window = threading.Thread(
                target=laser_tracker.run_window, args=(self.controller, self.rig),
                kwargs={"stop": self._stop_window, "space_toggles_laser": False},
                name="tracker-window", daemon=True)
            self._window.start()

    def close(self) -> None:
        self._stop_window.set()
        if self._window is not None:
            self._window.join(timeout=2.0)
        try:
            self.controller.shutdown()
        finally:
            try:
                self.gimbal.close()  # laser off
            finally:
                self.camera.stop()


class RigCamera:
    """Assistant camera backed by the rig: a clean, full-resolution photo."""

    fixture = None
    external = True  # the UI cannot switch this camera or its resolution

    def __init__(self, rig: LaserRig, artifacts: Path, timeout: float = 30.0):
        self.rig, self.artifacts, self.timeout = rig, Path(artifacts), timeout
        self.resolution = rig.photo_size
        self.camera_name = self.camera_label = rig.label

    async def capture_and_process(self) -> ImageFrame:
        future = self.rig.controller.request("snapshot")
        try:
            image = await asyncio.wait_for(asyncio.wrap_future(future), self.timeout)
        except asyncio.TimeoutError:
            raise RuntimeError("The laser rig did not deliver a photo; it may still be calibrating.") from None
        return await asyncio.to_thread(self._save, image)

    def _save(self, image) -> ImageFrame:
        import cv2
        directory = self.artifacts / "captures" / f"{datetime.now():%Y%m%d-%H%M%S-%f}-{uuid4().hex[:8]}"
        directory.mkdir(parents=True)
        # JPEG 95 keeps fine detail at about a tenth of a 1080p PNG's size,
        # which is uploaded with every model request.
        path = directory / "original.jpg"
        if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not save the rig photo to {path}")
        height, width = image.shape[:2]
        self.rig.last_photo_size = (width, height)
        return ImageFrame(path, width, height)


def to_tracker(box, photo_size, view_size):
    """A box in photo pixels as a Rect in the tracker's pixels."""
    from fixpoint.aim import Rect
    from fixpoint.camera import view_in_photo
    scale, x0, y0 = view_in_photo(photo_size, view_size)
    x1, y1, x2, y2 = box
    return Rect((x1 - x0) * scale, (y1 - y0) * scale, (x2 - x0) * scale, (y2 - y0) * scale)


def middle(x1: float, y1: float, x2: float, y2: float):
    """The middle half of a box (at least 8 px across, or the whole box if smaller).

    Stopping as soon as the dot touches the box can leave it on the box's edge,
    where it reads as pointing at the neighbor. People point at the middle.
    """
    from fixpoint.aim import Rect

    def span(low, high):
        center, size = (low + high) / 2, high - low
        half = max(size / 4, min(size / 2, 4.0))
        return center - half, center + half

    left, right = span(x1, x2)
    top, bottom = span(y1, y2)
    return Rect(left, top, right, bottom)


# A dot that stops this close to the box (tracker pixels) still marks the part.
NEAR_MISS_PX = 25.0


def near(box, point, margin: float = NEAR_MISS_PX) -> bool:
    """Whether a point is inside the box or within `margin` pixels of it."""
    x, y = point
    return box.x1 - margin <= x <= box.x2 + margin and box.y1 - margin <= y <= box.y2 + margin


class RigPointer:
    """Pointer adapter: steer the dot to the middle of the box and report when the camera sees it.

    Boxes arrive in the pixels of the latest photo (1920x1080 by default) and
    are converted to the tracker's 640x480 view, the middle of that photo.
    The controller blinks the dot wherever it stops, before the aim completes.
    """

    def __init__(self, rig: LaserRig, cancel_timeout: float = 10.0):
        self.rig, self.cancel_timeout = rig, cancel_timeout
        self._pending = None
        self._box = None

    async def send_bbox(self, x1: int, y1: int, x2: int, y2: int) -> None:
        view_size = tuple(self.rig.camera.size)
        self._box = to_tracker((x1, y1, x2, y2), self.rig.last_photo_size or view_size, view_size)
        box = self._box
        self._pending = self.rig.controller.request("aim", middle(box.x1, box.y1, box.x2, box.y2))

    async def wait_until_complete(self) -> None:
        from fixpoint.aim import AimStatus
        if self._pending is None:
            raise RuntimeError("No target was sent to the laser rig")
        try:
            result = await asyncio.wrap_future(self._pending)
        except RuntimeError as error:  # refused before moving, e.g. not calibrated
            raise PointingFailed(str(error)) from error
        if result.status == AimStatus.VALID:
            return
        if result.status == AimStatus.PREDICTED:
            # Aimed by the calibration, but the camera cannot see the dot there
            # (a dark hole, glare, an edge). The laser is still on the target.
            return
        if result.detection is not None and near(self._box, result.detection.xy):
            return  # stopped around the part: off the middle, or just outside a tiny box
        raise PointingFailed(f"{result.status.value}, {result.message.rstrip('.')}")

    async def cancel(self) -> bool:
        """Stop aiming; True once the controller is idle and the servos have stopped."""
        controller = self.rig.controller
        controller.submit("clear")
        if self._pending is not None:
            self._pending.cancel()  # skipped if it has not started yet
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.cancel_timeout
        while loop.time() < deadline:
            if controller.idle and self.rig.gimbal.arrival_time() <= time.perf_counter():
                return True
            await asyncio.sleep(0.05)
        return False
