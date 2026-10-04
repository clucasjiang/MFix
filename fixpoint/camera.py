"""Threaded webcam capture with locked settings for laser detection."""

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

# DirectShow treats 0.25 as manual exposure and 0.75 as automatic.
DSHOW_MANUAL_EXPOSURE = 0.25
DSHOW_AUTO_EXPOSURE = 0.75


def view_in_photo(photo_size: tuple[int, int], view_size: tuple[int, int]) -> tuple[float, float, float]:
    """Where a camera's lower-resolution view sits inside its photo.

    Webcams make other aspect ratios by cropping the middle of the sensor: the
    4:3 640x480 view of a 1920x1080 camera is its middle 1440x1080, scaled
    (measured to within about 1 px on the rig camera).
    Returns (scale, x0, y0) with view = (photo - (x0, y0)) * scale.
    """
    photo_width, photo_height = photo_size
    view_width, view_height = view_size
    crop_width = min(photo_width, photo_height * view_width / view_height)
    crop_height = min(photo_height, photo_width * view_height / view_width)
    return view_width / crop_width, (photo_width - crop_width) / 2, (photo_height - crop_height) / 2


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    t: float  # perf_counter() when the frame arrived
    seq: int


class FrameSource:
    """Keeps the newest frame and lets threads wait for later ones."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame: Frame | None = None
        self._seq = 0
        self.fps = 0.0

    def _publish(self, image: np.ndarray, t: float) -> None:
        with self._condition:
            if self._frame is not None:
                interval = t - self._frame.t
                if interval > 0:
                    self.fps = 0.9 * self.fps + 0.1 / interval if self.fps else 1.0 / interval
            self._seq += 1
            self._frame = Frame(image, t, self._seq)
            self._condition.notify_all()

    def latest(self) -> Frame | None:
        with self._condition:
            return self._frame

    def wait_frame(self, after_seq: int = 0, not_before: float = 0.0, timeout: float = 1.0) -> Frame | None:
        """Return the first frame newer than `after_seq` that arrived at or after `not_before`."""
        deadline = time.perf_counter() + timeout
        with self._condition:
            while True:
                frame = self._frame
                if frame is not None and frame.seq > after_seq and frame.t >= not_before:
                    return frame
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def frames(self, count: int, not_before: float, timeout: float = 1.5) -> list[Frame]:
        """Collect `count` consecutive frames that all arrived at or after `not_before`."""
        frames: list[Frame] = []
        seq = 0
        deadline = time.perf_counter() + timeout
        while len(frames) < count:
            frame = self.wait_frame(seq, not_before, max(0.0, deadline - time.perf_counter()))
            if frame is None:
                break
            frames.append(frame)
            seq = frame.seq
        return frames


class Camera(FrameSource):
    def __init__(
        self,
        index: int = 1,
        width: int = 640,
        height: int = 480,
        exposure: float = -8.0,
        gain: float = 0.0,
        backend: int = cv2.CAP_DSHOW,
    ) -> None:
        super().__init__()
        self.index = index
        self._requested_size = (width, height)
        self.exposure = exposure
        self._gain = gain
        self._backend = backend
        self._capture: cv2.VideoCapture | None = None
        self._capture_lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self.size = (width, height)

    def start(self) -> "Camera":
        capture, image = self._open(self._requested_size, self.exposure)
        self.size = (image.shape[1], image.shape[0])
        self._capture = capture
        self._running = True
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()
        return self

    def _open(self, size: tuple[int, int], exposure: float, fourcc: str | None = None):
        """Open the camera at `size` with every automatic adjustment locked."""
        width, height = size
        # Asking for the format in the open call is fast (about 0.3 s); changing
        # it on an open DirectShow stream rebuilds the stream (1.5-2.5 s).
        params = [cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc)] if fourcc else []
        params += [cv2.CAP_PROP_FRAME_WIDTH, width, cv2.CAP_PROP_FRAME_HEIGHT, height]
        capture = cv2.VideoCapture(self.index, self._backend, params)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera {self.index}.")
        # Laser detection compares frames over time, so anything the camera
        # adjusts on its own (exposure, gain, white balance, focus) is noise.
        capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, DSHOW_MANUAL_EXPOSURE)
        if not capture.set(cv2.CAP_PROP_EXPOSURE, exposure):
            print("Warning: camera did not accept manual exposure.")
        capture.set(cv2.CAP_PROP_GAIN, self._gain)
        capture.set(cv2.CAP_PROP_AUTO_WB, 0)
        capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        ok, image = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Camera {self.index} opened but returned no frames.")
        return capture, image

    def still(self, size: tuple[int, int], exposure: float, count: int = 2, settle_frames: int = 1) -> np.ndarray:
        """One photo at another resolution, then back to tracking frames.

        The camera is reopened at `size` (MJPEG, so 1080p still runs at 30 fps)
        and then reopened for tracking with the same settings as before. Tracking
        frames pause meanwhile: about 2 s on the rig camera, mostly opening and
        releasing the device. The exposure applies from the first frame there;
        `settle_frames` more are skipped for safety before averaging `count`.
        """
        images = []
        with self._capture_lock:
            self._capture.release()
            try:
                photo, _ = self._open(size, exposure, "MJPG")
                try:
                    for _ in range(settle_frames + count):
                        ok, image = photo.read()
                        if ok and (image.shape[1], image.shape[0]) == tuple(size):
                            images.append(image)
                finally:
                    photo.release()
            finally:
                self._capture, _ = self._open(self.size, self.exposure)
        if len(images) < count:
            raise RuntimeError(f"Camera {self.index} gave no {size[0]}x{size[1]} photo.")
        return np.mean([image.astype(np.float32) for image in images[-count:]], axis=0).round().astype(np.uint8)

    def _run(self) -> None:
        while self._running:
            with self._capture_lock:
                ok, image = self._capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            self._publish(image, time.perf_counter())

    def set_exposure(self, value: float) -> float:
        with self._capture_lock:
            self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, DSHOW_MANUAL_EXPOSURE)
            self._capture.set(cv2.CAP_PROP_EXPOSURE, value)
            self.exposure = self._capture.get(cv2.CAP_PROP_EXPOSURE)
        return self.exposure

    def show_settings(self) -> None:
        """Open the driver's own settings dialog (DirectShow only)."""
        with self._capture_lock:
            self._capture.set(cv2.CAP_PROP_SETTINGS, 1)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._capture is not None:
            # Hand the camera back to other apps in automatic mode.
            self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, DSHOW_AUTO_EXPOSURE)
            self._capture.release()
