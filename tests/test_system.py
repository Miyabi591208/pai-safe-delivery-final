import unittest
from pathlib import Path
import numpy as np
from safe_delivery.perception import Classifier, make_data, probabilities
from safe_delivery.planning import plan, segment_free, limit_change
from safe_delivery.simulation import make_scene, run, setup, observe


class DeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Classifier(Path('models/color_mlp.npz'))

    def test_unseen_synthetic_colors(self):
        x, y = make_data(2028, 600)
        self.assertGreater(float(np.mean(probabilities(x, self.model.weights).argmax(1)==y)), .98)

    def test_gray_and_dark_rejected(self):
        for level in (0, 5, 100, 240):
            self.assertFalse(self.model.predict(np.full((160, 160, 3), level, np.uint8))['accepted'])

    def test_planner_has_clear_segments(self):
        obstacles = [[0., 0., .3, .6]]
        route = plan([-2, 0], [2, 0], obstacles)
        self.assertIsNotNone(route)
        self.assertTrue(all(segment_free(a, b, obstacles) for a, b in zip(route[:-1], route[1:])))

    def test_blocked_map_and_bad_start_rejected(self):
        self.assertIsNone(plan([-2, 0], [2, 0], [[0, 0, .3, 3.]]))
        self.assertIsNone(plan([0, 0], [2, 0], [[0, 0, .3, .6]]))

    def test_vector_acceleration_bound(self):
        command = np.zeros(2)
        for target in ([1, 1], [-1, -1], [0, 1], [0, 0]):
            new = limit_change(command, target, .0275)
            self.assertLessEqual(np.linalg.norm(new-command), .0275+1e-12)
            command = new

    def test_rendered_recognition_all_classes(self):
        for label in range(3):
            m, d = setup(make_scene(10+label, label))
            self.assertEqual(self.model.predict(observe(m, d))['label'], label)

    def test_physics_baseline_collision_detected(self):
        result, trace = run(make_scene(10, 0), 'direct', self.model)
        self.assertTrue(result['collision'])
        self.assertFalse(result['success'])
        self.assertGreater(len(trace['xy']), 20)

    def test_physics_smooth_delivers_with_lower_jerk(self):
        scene = make_scene(10, 0)
        raw, _ = run(scene, 'astar', self.model)
        smooth, trace = run(scene, 'smooth', self.model)
        self.assertTrue(smooth['success'])
        self.assertFalse(smooth['collision'])
        self.assertLessEqual(smooth['command_accel_max'], .55+1e-8)
        self.assertLess(smooth['jerk_rms_m_s3'], raw['jerk_rms_m_s3'])
        self.assertGreater(np.linalg.norm(trace['xy'][-1]-trace['xy'][0]), 4.)

    def test_rendered_unknown_cargo_does_not_start(self):
        scene = make_scene(10, 0)
        scene.cargo_rgb = [.5, .5, .5]
        result, trace = run(scene, 'smooth', self.model)
        self.assertEqual(result['reason'], 'perception_rejected')
        self.assertEqual(result['sim_time'], 0.)
        self.assertIsNone(trace)


if __name__ == '__main__':
    unittest.main()
