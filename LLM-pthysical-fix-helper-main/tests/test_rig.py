"""Laser rig adapter: status mapping, cancel ACK, and full turns on the simulated rig."""
import asyncio
import shutil
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from src.models import BoundingBox, DiagnosticResponse, State
from src.operator import Operator
from src.pointer import PointingFailed
from src.rig import LaserRig, RigCamera, RigPointer, _import_fixpoint, middle

_import_fixpoint()
from fixpoint.aim import AimResult, AimStatus, Rect  # noqa: E402

TESTS = Path(__file__).resolve().parent


def artifacts_directory() -> Path:
    path = TESTS / ("test-artifacts-" + uuid4().hex)
    path.mkdir()
    return path


def remove_artifacts(path: Path) -> None:
    if path.parent == TESTS and path.name.startswith("test-artifacts-"):
        shutil.rmtree(path, ignore_errors=True)


def result(status: AimStatus, message: str = "") -> AimResult:
    return AimResult(status, Rect(1, 2, 3, 4), None, (90.0, 90.0), 0, 0.1, 0, message)


class FakeController:
    def __init__(self, outcome):
        self.outcome, self.requests, self.submitted, self.idle = outcome, [], [], True

    def request(self, name, *args):
        self.requests.append((name, args))
        future = Future()
        if isinstance(self.outcome, Exception):
            future.set_exception(self.outcome)
        else:
            future.set_result(self.outcome)
        return future

    def submit(self, name, *args):
        self.submitted.append(name)


def fake_rig(outcome):
    return SimpleNamespace(controller=FakeController(outcome), gimbal=SimpleNamespace(arrival_time=lambda: 0.0),
                           camera=SimpleNamespace(size=(640, 480)), label="Test rig", last_photo_size=None)


class RigPointerTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_hd_box_is_converted_to_tracker_pixels(self):
        rig = fake_rig(result(AimStatus.VALID))
        rig.last_photo_size = (1920, 1080)  # tracker view = middle 1440x1080, scaled by 4/9
        pointer = RigPointer(rig)
        await pointer.send_bbox(720, 360, 1200, 720)
        await pointer.wait_until_complete()
        name, (aimed,) = rig.controller.requests[0]
        for actual, expected in zip((aimed.x1, aimed.y1, aimed.x2, aimed.y2), (266.67, 200, 373.33, 280)):
            self.assertAlmostEqual(actual, expected, places=1)

    async def test_aims_at_the_middle_of_the_box(self):
        rig = fake_rig(result(AimStatus.VALID))
        pointer = RigPointer(rig)
        await pointer.send_bbox(100, 20, 140, 60)
        await pointer.wait_until_complete()
        self.assertEqual(rig.controller.requests[0], ("aim", (Rect(110, 30, 130, 50),)))

    def test_middle_of_small_boxes(self):
        self.assertEqual(middle(0, 0, 10, 6), Rect(1, 0, 9, 6))  # at least 8 px, never outside the box

    async def test_dot_on_the_box_but_off_the_middle_counts(self):
        from fixpoint.detector import Detection
        missed_middle = AimResult(AimStatus.FAILED, Rect(110, 30, 130, 50), Detection(104, 55, 20, 30),
                                  (90.0, 90.0), 10, 3.0, 0, "Could not settle inside")
        pointer = RigPointer(fake_rig(missed_middle))
        await pointer.send_bbox(100, 20, 140, 60)
        await pointer.wait_until_complete()

    async def test_dot_stopped_just_outside_a_tiny_box_counts(self):
        from fixpoint.detector import Detection
        for dot, counts in (((150, 40), True), ((170, 40), False)):  # box is 100-140 x 20-60
            with self.subTest(dot=dot):
                stopped = AimResult(AimStatus.FAILED, Rect(110, 30, 130, 50), Detection(*dot, 20, 30),
                                    (90.0, 90.0), 10, 3.0, 0, "Could not settle inside")
                pointer = RigPointer(fake_rig(stopped))
                await pointer.send_bbox(100, 20, 140, 60)
                if counts:
                    await pointer.wait_until_complete()
                else:
                    with self.assertRaises(PointingFailed):
                        await pointer.wait_until_complete()

    async def test_predicted_counts_as_complete(self):
        pointer = RigPointer(fake_rig(result(AimStatus.PREDICTED)))
        await pointer.send_bbox(10, 20, 30, 40)
        await pointer.wait_until_complete()

    async def test_known_failures_raise_pointing_failed(self):
        for outcome in (result(AimStatus.UNREACHABLE, "Target is outside the calibrated area."),
                        result(AimStatus.FAILED, "Gimbal is at its angle limit."),
                        RuntimeError("The pointer is not calibrated yet")):
            with self.subTest(outcome=outcome):
                pointer = RigPointer(fake_rig(outcome))
                await pointer.send_bbox(10, 20, 30, 40)
                with self.assertRaises(PointingFailed):
                    await pointer.wait_until_complete()

    async def test_cancel_acknowledges_only_when_idle(self):
        rig = fake_rig(result(AimStatus.VALID))
        rig.controller.idle = False
        pointer = RigPointer(rig, cancel_timeout=0.2)
        self.assertFalse(await pointer.cancel())
        self.assertEqual(rig.controller.submitted, ["clear"])
        rig.controller.idle = True
        self.assertTrue(await pointer.cancel())


class Recorder:
    def start(self): pass
    def stop(self): return Path("input.wav")
    def abort(self): pass


class Speech:
    def __init__(self): self.texts = []
    async def transcribe(self, audio): return "Which part should I check?"
    async def speak(self, text): self.texts.append(text)


class BoxModel:
    """Answers every turn with a fixed box and records what it was shown."""

    def __init__(self, rig=None):
        self.box, self.rig, self.frames, self.laser_during_photo = None, rig, [], []

    async def diagnose(self, transcript, frame, session):
        self.frames.append(frame)
        if self.rig is not None:
            self.laser_during_photo.append(self.rig.gimbal.laser_on)
        x1, y1, x2, y2 = self.box
        return DiagnosticResponse(speech="Check this part.", target_present=True, target_label="part",
                                  bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2), confidence=0.9, status="inspect")


async def turn(operator):
    operator.space_down()
    await asyncio.sleep(0.05)
    operator.space_up()
    await operator.wait_until_idle()


class OperatorFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_unreachable_target_returns_to_ready_with_feedback(self):
        temp = artifacts_directory()
        try:
            from src.camera import Camera
            image = temp / "board.png"
            import cv2
            cv2.imwrite(str(image), np.zeros((480, 640, 3), np.uint8))
            model = BoxModel()
            model.box = (5, 5, 20, 20)
            operator = Operator(Recorder(), Camera(temp, image), Speech(), model,
                                RigPointer(fake_rig(result(AimStatus.UNREACHABLE, "Target is outside the calibrated area."))),
                                notify=lambda text: None)
            await turn(operator)
            self.assertEqual(operator.state, State.READY)
            self.assertIsNone(operator.last_error)
            self.assertIn("could not mark part", operator.session.pending_feedback[0])
            await operator.close()
        finally:
            remove_artifacts(temp)


class SimulatedRigTests(unittest.IsolatedAsyncioTestCase):
    """Real controller, detector and aiming against the simulator (about 40 s)."""

    @classmethod
    def setUpClass(cls):
        cls.temp = artifacts_directory()
        cls.rig = LaserRig(sim=True, window=False, calibration=cls.temp / "calibration.json")
        deadline = time.time() + 20
        while cls.rig.controller.view().status not in ("uncalibrated", "no dot", "error") and time.time() < deadline:
            time.sleep(0.05)
        cls.rig.controller.request("calibrate").result(timeout=90)

    @classmethod
    def tearDownClass(cls):
        cls.rig.close()
        remove_artifacts(cls.temp)

    def setUp(self):
        self.assertIsNotNone(self.rig.controller.aimer, "simulated calibration failed")

    async def run_turn(self, box):
        model = BoxModel(self.rig)
        model.box = box
        operator = Operator(Recorder(), RigCamera(self.rig, self.temp), Speech(), model,
                            RigPointer(self.rig), notify=lambda text: None, pointer_timeout=20)
        await turn(operator)
        await operator.close()
        return operator, model

    async def test_turn_points_the_laser_into_the_box(self):
        operator, model = await self.run_turn((280, 230, 320, 260))
        self.assertIsNone(operator.last_error)
        self.assertEqual((model.frames[0].width, model.frames[0].height), (640, 480))
        self.assertEqual(model.laser_during_photo, [False])  # clean photo
        last = self.rig.controller.view().last_result
        self.assertEqual(last.status, AimStatus.VALID)
        self.assertTrue(Rect(280, 230, 320, 260).contains(last.detection.xy))
        self.assertTrue(self.rig.gimbal.laser_on)  # still marking the part

    async def test_hidden_dot_is_reported_complete(self):
        operator, _ = await self.run_turn((408, 138, 432, 162))  # the simulator's hole
        self.assertIsNone(operator.last_error)
        self.assertEqual(self.rig.controller.view().last_result.status, AimStatus.PREDICTED)

    async def test_unreachable_box_does_not_lock(self):
        operator, _ = await self.run_turn((1, 1, 12, 12))
        self.assertEqual(operator.state, State.STOPPED)  # closed normally, not ERROR
        self.assertIsNone(operator.last_error)
        self.assertTrue(any("could not mark" in item for item in operator.session.pending_feedback))


if __name__ == "__main__":
    unittest.main()
