import sys
import os
import unittest
import numpy as np

# Ensure project root is on sys.path so tests can import local packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.metrics import mae, rmse


class TestMetrics(unittest.TestCase):
    def test_mae_zero(self):
        y = np.zeros((2, 3))
        self.assertEqual(mae(y, y), 0.0)

    def test_mae_simple(self):
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        self.assertAlmostEqual(mae(y_true, y_pred), 0.6666666666, places=6)

    def test_rmse_zero(self):
        y = np.zeros((4,))
        self.assertEqual(rmse(y, y), 0.0)

    def test_rmse_simple(self):
        y_true = np.array([0.0, 2.0])
        y_pred = np.array([1.0, 1.0])
        self.assertAlmostEqual(rmse(y_true, y_pred), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
