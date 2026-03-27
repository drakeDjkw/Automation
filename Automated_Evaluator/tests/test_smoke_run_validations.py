import os
import sys
import shutil
import unittest

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import torch
except Exception:
    torch = None

try:
    import pandas as pd
except Exception:
    pd = None

from run_validations import main as run_validations_main


@unittest.skipIf(torch is None or pd is None, "torch or pandas missing - skipping smoke test")
class TestSmokeRunValidations(unittest.TestCase):
    def setUp(self):
        # Clean results folder to ensure a fresh run
        self.results_dir = os.path.abspath('results')
        self.experiments_dir = os.path.join(self.results_dir, 'experiments')
        self.summary_csv = os.path.join(self.results_dir, 'summary.csv')

        if os.path.exists(self.experiments_dir):
            shutil.rmtree(self.experiments_dir)
        if os.path.exists(self.summary_csv):
            os.remove(self.summary_csv)
        # Prepare a compatible mock weights file for the mock model so at least one
        # experiment runs successfully during the smoke test.
        self.weights_path = os.path.join('models', 'convlstm_smoke_test.pth')
        try:
            import models.convlstm as mc
            model = mc.ConvLSTM()
            import torch
            torch.save(model.state_dict(), self.weights_path)
        except Exception:
            # If torch or model not available, tests will be skipped by decorator
            pass

        # Create a small mapping file mapping the basename to the mock model
        self.mapping_file = os.path.join('tests', 'mapping_temp.json')
        try:
            with open(self.mapping_file, 'w') as f:
                json = __import__('json')
                json.dump({
                    'convlstm_smoke_test': {'module': 'models.convlstm', 'class': 'ConvLSTM'}
                }, f)
        except Exception:
            pass

    def test_run_validations_produces_summary(self):
        # Run the full automation; pass the mapping to ensure our smoke weights are used
        run_validations_main(map_file=self.mapping_file)

        # Check that summary CSV exists and has at least one row
        self.assertTrue(os.path.exists(self.summary_csv), msg="summary.csv not found after running validations")

        df = pd.read_csv(self.summary_csv)
        self.assertTrue(len(df) >= 1, msg="summary.csv is empty; expected at least one experiment")

    def tearDown(self):
        # Don't remove results completely to preserve logs, but remove experiments to keep tests idempotent
        if os.path.exists(self.experiments_dir):
            shutil.rmtree(self.experiments_dir)
        # cleanup temporary files
        try:
            if os.path.exists(self.weights_path):
                os.remove(self.weights_path)
        except Exception:
            pass
        try:
            if os.path.exists(self.mapping_file):
                os.remove(self.mapping_file)
        except Exception:
            pass


if __name__ == '__main__':
    unittest.main()
