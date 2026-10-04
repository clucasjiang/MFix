"""Full pipeline against the simulator, in real time (about a minute)."""

import time
import unittest

import numpy as np

from fixpoint.aim import AimStatus, Aimer, Rect
from fixpoint.calibration import calibrate
from fixpoint.detector import LaserDetector
from fixpoint.rig import Rig
from fixpoint.sim import SimCamera, SimGimbal, SimWorld


def visible_targets(world: SimWorld, count: int, size: float, rng: np.random.Generator) -> list[Rect]:
    """Random target rectangles centered on surfaces where the dot shows up."""
    width, height = world.size
    background = world.background.max(axis=2)
    targets = []
    while len(targets) < count:
        x, y = rng.uniform(0.12, 0.88) * width, rng.uniform(0.12, 0.88) * height
        patch = (slice(int(y - size), int(y + size)), slice(int(x - size), int(x + size)))
        if world.reflectance[patch].min() > 0.15 and background[patch].max() < 200:
            targets.append(Rect(x - size / 2, y - size / 2, x + size / 2, y + size / 2))
    return targets


class EndToEndTest(unittest.TestCase):
    def run_pipeline(self, fine: bool, target_size: float, count: int):
        world = SimWorld(fine=fine)
        camera = SimCamera(world).start()
        gimbal = SimGimbal(world)
        try:
            rig = Rig(camera, gimbal, LaserDetector())
            gimbal.move_to(90, 144, use_limits=False)
            time.sleep(1.5)
            gimbal.set_laser(True)
            latency = rig.measure_latency()
            self.assertIsNotNone(latency)
            self.assertLess(abs(latency - world.latency), 0.06)

            started = time.perf_counter()
            model = calibrate(rig, world.size)
            calibration_time = time.perf_counter() - started
            gimbal.limits = model.angle_bounds(margin=3.0)
            aimer = Aimer(rig, model)

            rng = np.random.default_rng(7)
            results = [aimer.aim(rect) for rect in visible_targets(world, count, target_size, rng)]
            hole = aimer.aim(Rect(408, 138, 432, 162))
            return model, calibration_time, results, hole
        finally:
            gimbal.close()
            camera.stop()

    def report(self, label, model, calibration_time, results, hole):
        valid = [r for r in results if r.status == AimStatus.VALID]
        times = [r.elapsed for r in valid]
        print(f"\n[{label}] calibration {calibration_time:.1f} s, {len(model.samples)} points, fit {model.rms:.2f} px")
        for r in results:
            print(f"  {r.status.value:9s} {r.elapsed:5.2f} s  corrections {r.corrections}  blinks {r.blinks}  "
                  f"error {r.error_px if r.error_px is None else round(r.error_px, 1)}  {r.message}")
        if times:
            print(f"  valid {len(valid)}/{len(results)}, median {np.median(times):.2f} s, max {max(times):.2f} s")
        print(f"  hole target: {hole.status.value} ({hole.message})")
        return valid

    def test_fine_firmware(self):
        model, calibration_time, results, hole = self.run_pipeline(fine=True, target_size=24, count=12)
        valid = self.report("fine firmware, 24 px targets", model, calibration_time, results, hole)
        self.assertLess(model.rms, 3.0)
        self.assertGreaterEqual(len(valid), 11)
        self.assertEqual(hole.status, AimStatus.PREDICTED)


if __name__ == "__main__":
    unittest.main()
