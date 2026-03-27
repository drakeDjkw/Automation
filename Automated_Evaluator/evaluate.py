"""Flexible evaluation entrypoint.

This module exposes run_evaluation(...) which attempts to:
- import a model class from a module (e.g. models.convlstm.ConvLSTM)
- load weights from a .pth file
- (optionally) construct a dataloader using utils.dataset.RainfallDataset
- run the Evaluator and save results via utils.evaluator.save_experiment

The functions fail gracefully when model/dataset source code is missing and
return a descriptive result dictionary.
"""

import importlib
import os
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader

from utils.evaluator import Evaluator, save_experiment


def _camel_case(s: str) -> str:
    parts = s.replace("-", "_").split("_")
    return "".join(p.capitalize() for p in parts)


def run_evaluation(
    model_file: str,
    model_module: Optional[str] = None,
    model_class: Optional[str] = None,
    dataset_file: Optional[str] = "data/rainfall.npy",
    batch_size: int = 4,
    device: str = "cuda",
):
    """Run evaluation for a single weights file.

    Returns a dict with keys: status (ok|skipped|error), metrics (when ok), path (saved json), and reason.
    """
    model_file = os.path.abspath(model_file)

    base = os.path.splitext(os.path.basename(model_file))[0]
    if model_module is None:
        model_module = f"models.{base}"
    if model_class is None:
        model_class = _camel_case(base)

    config = {
        "model": model_class,
        "weights": model_file,
        "dataset": dataset_file,
        "batch_size": batch_size,
    }

    # Try to import the model module and class
    try:
        mod = importlib.import_module(model_module)
        ModelClass = getattr(mod, model_class)
    except Exception as e:
        return {
            "status": "skipped",
            "reason": f"Could not import model module/class: {model_module}.{model_class}: {e}",
            "config": config,
        }

    # Instantiate model and load weights
    try:
        model = ModelClass()
        state = torch.load(model_file, map_location="cpu")
        # handle both state_dict and straight model save
        if isinstance(state, dict) and not any(k.startswith("module") for k in state.keys()):
            try:
                model.load_state_dict(state)
            except Exception:
                # maybe saved full model
                model = state
        else:
            try:
                model.load_state_dict(state)
            except Exception:
                # fallback: try to load full model
                model = state
    except Exception as e:
        return {"status": "error", "reason": f"Failed to construct/load model: {e}", "config": config}

    # Try to build dataloader
    dataloader = None
    if dataset_file is not None:
        try:
            dataset_mod = importlib.import_module("utils.dataset")
            RainfallDataset = getattr(dataset_mod, "RainfallDataset")
            dataset = RainfallDataset(dataset_file)
            dataloader = DataLoader(dataset, batch_size=batch_size)
        except Exception as e:
            # proceed without dataloader; we'll save a skipped result if needed
            dataloader = None

    if dataloader is None:
        return {"status": "skipped", "reason": "Missing dataset/dataloader (utils.dataset.RainfallDataset not available)", "config": config}

    # Run evaluation
    try:
        # If device not available fall back to cpu
        use_device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        evaluator = Evaluator(model, dataloader, device=use_device)
        results = evaluator.run()
    except Exception as e:
        return {"status": "error", "reason": f"Evaluation failed: {e}", "config": config}

    # Save experiment
    try:
        path = save_experiment(results, config)
    except Exception as e:
        return {"status": "error", "reason": f"Failed to save experiment: {e}", "config": config, "metrics": results}

    return {"status": "ok", "path": path, "metrics": results, "config": config}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation for a model weights file")
    parser.add_argument("model_file", help="Path to .pth model weights")
    parser.add_argument("--model-module", help="Python module path that defines the model class (e.g. models.convlstm)")
    parser.add_argument("--model-class", help="Model class name (e.g. ConvLSTM)")
    parser.add_argument("--dataset-file", default="data/rainfall.npy", help="Path to dataset file")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    out = run_evaluation(
        args.model_file,
        model_module=args.model_module,
        model_class=args.model_class,
        dataset_file=args.dataset_file,
        batch_size=args.batch_size,
        device=args.device,
    )

    print(out)
