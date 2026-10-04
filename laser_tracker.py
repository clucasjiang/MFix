"""Point the gimbal laser into a rectangle selected on the live camera view."""

import argparse
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from fixpoint.aim import AimStatus, Rect
from fixpoint.controller import Controller, View
from fixpoint.detector import LaserDetector
from fixpoint.rig import Rig

WINDOW = "Fixpoint laser tracker"
HERE = Path(__file__).resolve().parent

# cv2.waitKeyEx codes for arrow keys on Windows.
KEY_LEFT, KEY_UP, KEY_RIGHT, KEY_DOWN = 2424832, 2490368, 2555904, 2621440

STATUS_COLORS = {
    AimStatus.MOVING: (0, 215, 255),
    AimStatus.VALID: (0, 200, 0),
    AimStatus.PREDICTED: (0, 140, 255),
    AimStatus.LOST: (0, 0, 255),
    AimStatus.FAILED: (0, 0, 255),
    AimStatus.UNREACHABLE: (0, 0, 255),
    AimStatus.CANCELLED: (180, 180, 180),
}
HELP = (
    "drag or click 2 corners: target   X clear   C calibrate   B find dot   L laser   Q quit",
    "arrows/WASD jog (shift x5)   [ ] exposure   T auto-exposure   M signal   G points   P snapshot",
)


class RectSelector:
    """Mouse input: drag a rectangle, or click its two corners. Right-click cancels."""

    def __init__(self) -> None:
        self.anchor: tuple[int, int] | None = None
        self.cursor: tuple[int, int] | None = None
        self._pressed_at: tuple[int, int] | None = None
        self._finished: Rect | None = None

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        self.cursor = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.anchor is None:
                self.anchor = self._pressed_at = (x, y)
            else:
                self._finish((x, y))
        elif event == cv2.EVENT_LBUTTONUP and self.anchor is not None and self._pressed_at is not None:
            # A drag finishes on release; a click leaves the anchor for a second click.
            if abs(x - self._pressed_at[0]) > 4 and abs(y - self._pressed_at[1]) > 4:
                self._finish((x, y))
            self._pressed_at = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.anchor = self._pressed_at = None

    def _finish(self, corner: tuple[int, int]) -> None:
        if corner[0] != self.anchor[0] and corner[1] != self.anchor[1]:
            self._finished = Rect.from_corners(self.anchor, corner)
        self.anchor = self._pressed_at = None

    def take(self) -> Rect | None:
        rect, self._finished = self._finished, None
        return rect


def text_panel(image: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]], top: int,
               scale: float = 0.45, line_height: int = 18) -> None:
    """Text lines on a translucent dark band so they read on any background."""
    bottom = min(image.shape[0], top + line_height * len(lines) + 6)
    band = image[top:bottom]
    band[:] = (band * 0.35).astype(np.uint8)
    for row, (text, color) in enumerate(lines):
        cv2.putText(image, text, (8, top + line_height * (row + 1) - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)


def draw_rect(image: np.ndarray, rect: Rect, color, thickness: int = 2) -> None:
    cv2.rectangle(image, (int(rect.x1), int(rect.y1)), (int(rect.x2), int(rect.y2)), color, thickness)


def draw_overlay(image: np.ndarray, view: View, selector: RectSelector, frame_t: float,
                 show_samples: bool, fps: float, exposure: float | None, fine: bool) -> None:
    if show_samples and view.samples is not None:
        for yaw, pitch, x, y in view.samples:
            cv2.circle(image, (int(x), int(y)), 2, (255, 120, 0), -1)

    if view.rect is not None:
        color = STATUS_COLORS.get(view.aim_status, (255, 255, 255))
        draw_rect(image, view.rect, color)
        cx, cy = view.rect.center.astype(int)
        cv2.drawMarker(image, (int(cx), int(cy)), color, cv2.MARKER_CROSS, 10, 1)

    if selector.anchor is not None and selector.cursor is not None:
        draw_rect(image, Rect.from_corners(selector.anchor, selector.cursor), (255, 255, 0), 1)

    if view.predicted is not None:
        px, py = view.predicted.astype(int)
        cv2.drawMarker(image, (int(px), int(py)), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 1)

    fresh = view.detection is not None and frame_t - view.detection_t < 0.3
    if fresh:
        dx, dy = int(round(view.detection.x)), int(round(view.detection.y))
        cv2.circle(image, (dx, dy), 10, (255, 255, 0), 2)
        if view.rect is not None:
            cx, cy = view.rect.center.astype(int)
            cv2.line(image, (dx, dy), (int(cx), int(cy)), (255, 255, 0), 1)

    dot = (f"dot {view.detection.x:.1f}, {view.detection.y:.1f} (snr {view.detection.snr:.0f})"
           if fresh else "dot not seen")
    lines = [
        (f"{view.status.upper()}   {view.message}", STATUS_COLORS.get(view.aim_status, (255, 255, 255))),
        (f"yaw {view.angles[0]:.2f}  pitch {view.angles[1]:.2f}   {dot}", (255, 255, 255)),
        (f"{fps:.0f} fps   latency {view.latency * 1000:.0f} ms   exposure "
         f"{exposure if exposure is not None else '-'}   noise {view.noise:.1f}   "
         f"{'fine' if fine else 'whole-degree'} firmware", (200, 200, 200)),
    ]
    if view.model_rms is not None:
        lines.append((f"calibration fit {view.model_rms:.1f} px, {view.px_per_degree:.1f} px per degree",
                      (200, 200, 200)))
    result = view.last_result
    if result is not None:
        error = f", {result.error_px:.1f} px from center" if result.error_px is not None else ""
        lines.append((f"last aim: {result.status.value} in {result.elapsed:.2f} s, "
                      f"{result.corrections} corrections, {result.blinks} blinks{error}", (200, 200, 200)))
    text_panel(image, lines, 0)
    text_panel(image, [(text, (220, 220, 220)) for text in HELP], image.shape[0] - 40, 0.4, 16)


def signal_view(detector: LaserDetector, shape: tuple[int, ...]) -> np.ndarray:
    """Heat map of the laser signal, scaled so 10x the noise level is full red."""
    signal = detector.last_signal
    if signal is None:
        return np.zeros(shape, np.uint8)
    scaled = np.clip(signal / (10.0 * max(detector.last_noise, 0.1)), 0, 1)
    return cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def run_window(controller: Controller, rig: Rig, *, target: Rect | None = None,
               stop: threading.Event | None = None, space_toggles_laser: bool = True) -> None:
    """Show the debug view and handle its keys until Q, closing the window, or `stop`.

    Call all of this from one thread: OpenCV windows belong to the thread that made them.
    """
    camera, gimbal = rig.camera, rig.gimbal
    selector = RectSelector()
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, selector.on_mouse)
    pending_target = target
    show_signal = show_samples = False
    display_gain = 1.0
    jog_step = 1.0
    laser_keys = ("l", " ") if space_toggles_laser else ("l",)

    try:
        while stop is None or not stop.is_set():
            frame = camera.latest()
            if frame is None:
                time.sleep(0.01)
                continue
            view = controller.view()
            if pending_target is not None and view.status in ("ready", "valid"):
                controller.submit("aim", pending_target)
                pending_target = None

            # Locked low exposure makes the image dark; brighten it for people.
            display_gain = 0.9 * display_gain + 0.1 * float(np.clip(110.0 / max(frame.image.mean(), 1.0), 1.0, 6.0))
            canvas = signal_view(rig.detector, frame.image.shape) if show_signal else \
                cv2.convertScaleAbs(frame.image, alpha=display_gain)
            draw_overlay(canvas, view, selector, frame.t, show_samples, camera.fps,
                         getattr(camera, "exposure", None), gimbal.fine)
            cv2.imshow(WINDOW, canvas)

            rect = selector.take()
            if rect is not None:
                controller.submit("aim", rect)

            key = cv2.waitKeyEx(1)
            if key != -1:
                char = chr(key).lower() if key < 256 else ""
                shift = key < 256 and chr(key).isupper()
                if char in ("q", "\x1b"):
                    break
                if key in (KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN) or char in ("w", "a", "s", "d"):
                    step = jog_step * (5 if shift else 1)
                    # Default directions match the mount: +yaw moves left, +pitch moves down.
                    dyaw = step if key == KEY_LEFT or char == "a" else -step if key == KEY_RIGHT or char == "d" else 0
                    dpitch = -step if key == KEY_UP or char == "w" else step if key == KEY_DOWN or char == "s" else 0
                    controller.submit("jog", dyaw, dpitch)
                elif char == "c":
                    controller.submit("calibrate")
                elif char == "b":
                    controller.submit("blink")
                elif char in laser_keys:
                    controller.submit("laser")
                elif char == "x":
                    controller.submit("clear")
                elif char == "[":
                    controller.submit("exposure", -1.0)
                elif char == "]":
                    controller.submit("exposure", 1.0)
                elif char == "t":
                    controller.submit("tune")
                elif char == "m":
                    show_signal = not show_signal
                elif char == "g":
                    show_samples = not show_samples
                elif char == "o" and hasattr(camera, "show_settings"):
                    camera.show_settings()
                elif char == "p":
                    folder = HERE / "snapshots"
                    folder.mkdir(exist_ok=True)
                    path = folder / f"fixpoint_{datetime.now():%Y%m%d_%H%M%S}.png"
                    cv2.imwrite(str(path), canvas)
                    print(f"Saved {path}")
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyWindow(WINDOW)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6", help="ESP32 serial port (default: COM6)")
    parser.add_argument("--camera", type=int, default=1, help="camera index (default: 1)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--exposure", type=float, default=-8.0,
                        help="manual exposure, log2 seconds; more negative is darker (default: -8)")
    parser.add_argument("--start", type=float, nargs=2, default=(90.0, 144.0), metavar=("YAW", "PITCH"),
                        help="angles to start from before calibration (default: 90 144)")
    parser.add_argument("--calibration", type=Path, default=HERE / "calibration.json")
    parser.add_argument("--log", type=Path, default=HERE / "aim_log.csv", help="CSV of every aim result")
    parser.add_argument("--target", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        help="aim at this rectangle once ready")
    parser.add_argument("--no-hold", action="store_true", help="do not re-aim if the dot drifts out")
    parser.add_argument("--legacy", action="store_true", help="force whole-degree commands")
    parser.add_argument("--sim", action="store_true", help="run against the built-in simulator")
    parser.add_argument("--sim-legacy", action="store_true", help="simulate the old whole-degree firmware")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sim or args.sim_legacy:
        from fixpoint.sim import SimCamera, SimGimbal, SimWorld

        world = SimWorld(size=(args.width, args.height), fine=not args.sim_legacy, exposure=args.exposure)
        camera = SimCamera(world).start()
        gimbal = SimGimbal(world)
        if args.calibration == HERE / "calibration.json":
            args.calibration = HERE / "calibration_sim.json"
    else:
        from fixpoint.camera import Camera
        from fixpoint.gimbal import SerialGimbal

        camera = Camera(args.camera, args.width, args.height, args.exposure).start()
        print(f"Opening {args.port}...")
        try:
            gimbal = SerialGimbal(args.port, force_legacy=args.legacy)
        except Exception as error:
            camera.stop()
            raise SystemExit(f"Could not open {args.port}: {error}\n"
                             "Close PlatformIO Serial Monitor and other gimbal scripts.") from error

    rig = Rig(camera, gimbal, LaserDetector())
    controller = Controller(rig, camera.size, args.calibration, tuple(args.start),
                            hold=not args.no_hold, log_path=args.log)
    controller.start()
    target = Rect.from_corners(args.target[:2], args.target[2:]) if args.target else None
    try:
        run_window(controller, rig, target=target)
    finally:
        controller.shutdown()
        gimbal.close()  # laser off
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
