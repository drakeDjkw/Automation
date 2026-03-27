import os
import sys
import unittest

# ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import torch
except Exception:
    torch = None

from evaluate import run_evaluation


@unittest.skipIf(torch is None, "torch is not installed - skipping end-to-end evaluate test")
class TestEvaluateEndToEnd(unittest.TestCase):
    def test_run_evaluation_ok(self):
        # Use CPU for tests
        device = "cpu"

        # Create a temporary weights file matching our mock model
        import models.convlstm as mc

        model = mc.ConvLSTM()

        weights_path = os.path.join("models", "convlstm_test.pth")
        # Save state_dict so load_state_dict succeeds
        torch.save(model.state_dict(), weights_path)

        try:
            out = run_evaluation(weights_path, model_module="models.convlstm", model_class="ConvLSTM", device=device)
            self.assertIsInstance(out, dict)
            self.assertEqual(out.get("status"), "ok", msg=f"Expected status 'ok', got: {out}")
        finally:
            # Cleanup
            try:
                os.remove(weights_path)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
