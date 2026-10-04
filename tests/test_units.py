"""Fast checks that need no hardware and no real-time simulation."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from fixpoint.aim import AimResult, AimStatus, Rect
from fixpoint.controller import FLASH_BLINKS, Controller
from fixpoint.detector import LaserDetector
from fixpoint.gimbal import LASER_DEVICE, GimbalBase
from fixpoint.model import AngleModel
from fixpoint.sim import SimWorld, pwm_angle


class RecordingGimbal(GimbalBase):
    def __init__(self, fine: bool) -> None:
        super().__init__(fine)
        self.sent: list[bytes] = []

    def _write(self, data: bytes) -> None:
        self.sent.append(data)


class GimbalProtocolTest(unittest.TestCase):
    def test_fine_moves_send_centidegrees_high_byte_first(self):
        gimbal = RecordingGimbal(fine=True)
        self.assertEqual(gimbal.move_to(123.456, 45.0), (123.46, 45.0))
        self.assertEqual(gimbal.sent, [bytes((0x15, 0x05, 0x30, 0x3A)), bytes((0x15, 0x06, 0x11, 0x94))])

    def test_legacy_moves_round_to_whole_degrees(self):
        gimbal = RecordingGimbal(fine=False)
        self.assertEqual(gimbal.move_to(100.6, 90.0), (101.0, 90.0))
        self.assertEqual(gimbal.sent, [bytes((0x15, 0x01, 101))])  # pitch unchanged: not resent

    def test_soft_limits_apply_unless_disabled(self):
        gimbal = RecordingGimbal(fine=True)
        gimbal.limits = (80.0, 100.0, 80.0, 100.0)
        self.assertEqual(gimbal.move_to(120, 60), (100.0, 80.0))
        self.assertEqual(gimbal.move_to(120, 60, use_limits=False), (120.0, 60.0))

    def test_speed_is_fine_firmware_only(self):
        legacy = RecordingGimbal(fine=False)
        legacy.set_speed(120)
        self.assertEqual(legacy.sent, [])
        fine = RecordingGimbal(fine=True)
        fine.set_speed(120)
        self.assertEqual(fine.sent, [bytes((0x15, 0x07, 120))])

    def test_ramp_estimate_matches_firmware_speed(self):
        gimbal = RecordingGimbal(fine=True)
        gimbal.move_to(90, 90)
        gimbal.move_to(120, 90)
        t0 = gimbal._yaw.t0
        self.assertAlmostEqual(gimbal.position(t0 + 0.25)[0], 105.0)
        self.assertAlmostEqual(gimbal.arrival_time() - t0, 0.5)

    def test_legacy_pwm_ticks_are_coarse(self):
        # The old 10-bit timer cannot tell 101 and 102 degrees apart: only
        # 104 distinct positions exist across 0-180 degrees.
        self.assertEqual(pwm_angle(101, fine=False), pwm_angle(102, fine=False))
        self.assertEqual(len({pwm_angle(a, fine=False) for a in range(181)}), 104)
        self.assertLess(abs(pwm_angle(100.3, fine=True) - 100.3), 0.12)


class AngleModelTest(unittest.TestCase):
    def setUp(self):
        self.world = SimWorld()
        samples = []
        for yaw in np.linspace(65, 120, 8):
            for pitch in np.linspace(118, 165, 6):
                pixel = self.world.dot_pixel(yaw, pitch)
                if pixel is not None:
                    samples.append((yaw, pitch, *pixel))
        self.model = AngleModel.fit(np.array(samples))

    def test_fit_follows_the_true_geometry(self):
        self.assertLess(self.model.rms, 3.0)
        for yaw, pitch in ((80.3, 131.7), (101.2, 150.4), (92.0, 140.0)):
            error = np.hypot(*(self.model.to_pixel(yaw, pitch) - self.world.dot_pixel(yaw, pitch)))
            self.assertLess(error, 4.0)

    def test_inverse_round_trips(self):
        for yaw, pitch in ((80.3, 131.7), (101.2, 150.4)):
            angles = self.model.to_angles(*self.model.to_pixel(yaw, pitch))
            np.testing.assert_allclose(angles, (yaw, pitch), atol=0.01)

    def test_save_and_load(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "calibration.json"
            self.model.save(path)
            loaded = AngleModel.load(path)
        np.testing.assert_allclose(loaded.to_pixel(95, 140), self.model.to_pixel(95, 140))

    def test_covers_only_the_calibrated_area(self):
        center = self.model.samples[:, 2:4].mean(axis=0)
        self.assertTrue(self.model.covers(*center))
        self.assertFalse(self.model.covers(-200, -200))


class DetectorTest(unittest.TestCase):
    def setUp(self):
        self.world = SimWorld()
        self.rng = np.random.default_rng(0)

    def frame(self, pixel=None, gain: float = 1.0) -> np.ndarray:
        image = self.world.background * gain
        if pixel is not None:
            self.world._draw_dot(image, np.asarray(pixel, float), gain)
        image = image + self.world._noise_bank[self.rng.integers(6)]
        return np.clip(image, 0, 255).astype(np.uint8)

    def detector(self) -> LaserDetector:
        detector = LaserDetector()
        detector.set_reference([self.frame(), self.frame()])
        return detector

    def test_finds_dot_on_each_surface(self):
        detector = self.detector()
        surfaces = {"paper": (300, 250), "wood": (610, 200), "green board": (450, 320), "dark chip": (200, 320)}
        for name, pixel in surfaces.items():
            with self.subTest(surface=name):
                detection = detector.detect(self.frame(pixel))
                self.assertIsNotNone(detection)
                self.assertLess(np.hypot(detection.x - pixel[0], detection.y - pixel[1]), 1.0)

    def test_ignores_static_red_leds(self):
        detection = self.detector().detect(self.frame((300, 250)))
        self.assertLess(np.hypot(detection.x - 300, detection.y - 250), 1.0)
        self.assertEqual(detection.rivals, 0)

    def test_no_dot_means_no_detection(self):
        self.assertIsNone(self.detector().detect(self.frame()))

    def test_dot_vanishes_in_hole_and_glare(self):
        detector = self.detector()
        self.assertIsNone(detector.detect(self.frame((420, 150))))  # hole
        self.assertIsNone(detector.detect(self.frame((300, 120))))  # saturated glare

    def test_search_window_prefers_the_prediction(self):
        detector = self.detector()
        image = self.frame((300, 250))
        self.world._draw_dot(image_float := image.astype(np.float32), np.array((340.0, 250.0)), 1.0)
        image = np.clip(image_float, 0, 255).astype(np.uint8)
        detection = detector.detect(image, predicted=np.array((338.0, 252.0)), radius=50)
        self.assertLess(np.hypot(detection.x - 340, detection.y - 250), 1.0)
        self.assertEqual(detection.rivals, 1)

    def test_lighting_change_marks_reference_stale(self):
        detector = self.detector()
        self.assertLess(detector.changed_fraction(self.frame()), 0.01)
        self.assertGreater(detector.changed_fraction(self.frame(gain=1.6)), 0.03)


class ViewInPhotoTest(unittest.TestCase):
    def test_4_3_view_is_the_middle_of_a_16_9_photo(self):
        from fixpoint.camera import view_in_photo
        self.assertEqual(view_in_photo((1920, 1080), (640, 480)), (640 / 1440, 240.0, 0.0))
        self.assertEqual(view_in_photo((1280, 960), (640, 480)), (0.5, 0.0, 0.0))
        self.assertEqual(view_in_photo((640, 480), (640, 480)), (1.0, 0.0, 0.0))


class RectTest(unittest.TestCase):
    def test_corners_in_any_order(self):
        rect = Rect.from_corners((50, 80), (10, 20))
        self.assertEqual((rect.x1, rect.y1, rect.x2, rect.y2), (10, 20, 50, 80))
        self.assertTrue(rect.contains((10, 80)))
        self.assertFalse(rect.contains((9.9, 50)))


class FlashTest(unittest.TestCase):
    def setUp(self):
        self.gimbal = RecordingGimbal(fine=True)
        self.gimbal.set_laser(True)  # on target
        self.gimbal.sent.clear()
        rig = SimpleNamespace(gimbal=self.gimbal, on_frame=None)
        self.controller = Controller(rig, (640, 480), Path("unused.json"), (90.0, 90.0))

    def laser_commands(self) -> list[int]:
        return [data[2] for data in self.gimbal.sent if data[1] == LASER_DEVICE]

    def test_blinks_in_place_and_leaves_the_laser_on(self):
        with patch("fixpoint.controller.FLASH_INTERVAL", 0.0):
            self.controller._flash()
        self.assertEqual(self.laser_commands(), [0, 1] * FLASH_BLINKS)
        self.assertEqual(len(self.gimbal.sent), 2 * FLASH_BLINKS)  # no moves
        self.assertTrue(self.gimbal.laser_on)

    def test_cancel_stops_blinking_with_the_laser_on(self):
        self.controller._cancel.set()
        self.controller._flash()
        self.assertEqual(self.laser_commands(), [0, 1])
        self.assertTrue(self.gimbal.laser_on)

    def test_blinks_wherever_the_dot_stops(self):
        rect = Rect(10, 10, 30, 30)
        for status, blinks in ((AimStatus.VALID, True), (AimStatus.PREDICTED, True),
                               (AimStatus.FAILED, True), (AimStatus.LOST, True),
                               (AimStatus.UNREACHABLE, False), (AimStatus.CANCELLED, False)):
            with self.subTest(status=status):
                self.gimbal.sent.clear()
                outcome = AimResult(status, rect, None, (90.0, 90.0), 0, 0.1, 0)
                self.controller.aimer = SimpleNamespace(aim=lambda rect, cancel: outcome)
                self.controller.rect = rect
                with patch("fixpoint.controller.FLASH_INTERVAL", 0.0):
                    self.assertIs(self.controller._aim(), outcome)
                self.assertEqual(self.laser_commands(), [0, 1] * FLASH_BLINKS if blinks else [])


if __name__ == "__main__":
    unittest.main()
