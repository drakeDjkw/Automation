"""Automate running validations for all weight files in `models/`.

This script will:
- scan `models/` for `*.pth` files
- for each file attempt to run `evaluate.run_evaluation(...)`
- collect per-experiment JSONs (saved by Evaluator/save_experiment)
- write a summary CSV to `results/summary.csv`

Note: This is conservative and will skip models/modules that can't be imported.
"""

import glob
import os
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from evaluate import run_evaluation


def discover_model_files(models_dir: str = "models") -> List[str]:
    pattern = os.path.join(models_dir, "*.pth")
    return sorted(glob.glob(pattern))


def build_summary(experiments_folder: str = "results/experiments", output_csv: str = "results/summary.csv"):
    records = []
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    if not os.path.exists(experiments_folder):
        print(f"No experiments folder found at {experiments_folder}")
        # write empty csv
        pd.DataFrame(columns=["id", "model", "MAE", "RMSE", "runtime"]).to_csv(output_csv, index=False)
        return

    for file in os.listdir(experiments_folder):
        if not file.endswith(".json"):
            continue
        path = os.path.join(experiments_folder, file)
        try:
            with open(path) as f:
                data = json.load(f)
            record = {
                "id": data.get("experiment_id"),
                "model": data.get("config", {}).get("model"),
                "MAE": data.get("metrics", {}).get("MAE"),
                "RMSE": data.get("metrics", {}).get("RMSE"),
                "runtime": data.get("metrics", {}).get("runtime_sec"),
            }
            records.append(record)
        except Exception as e:
            print(f"Failed to read experiment {path}: {e}")

    if records:
        df = pd.DataFrame(records)
        df = df.sort_values("RMSE") if "RMSE" in df.columns else df
        df.to_csv(output_csv, index=False)
        print(f"Wrote summary to {output_csv}")
    else:
        print("No experiment JSONs found; created empty summary")
        pd.DataFrame(columns=["id", "model", "MAE", "RMSE", "runtime"]).to_csv(output_csv, index=False)


def load_mapping(map_file: str) -> Dict[str, Dict[str, str]]:
    """Load optional mapping JSON that maps weights basename to module/class.

    Example:
    {
      "convlstm": {"module": "models.convlstm", "class": "ConvLSTM"}
    }
    """
    if not map_file or not os.path.exists(map_file):
        return {}
    try:
        with open(map_file) as f:
            return json.load(f)
    except Exception:
        return {}


def main(map_file: Optional[str] = None):
    model_files = discover_model_files()
    if not model_files:
        print("No .pth model files found in models/ - nothing to run")
        return

    mapping = load_mapping(map_file) if map_file else {}

    results = []
    for wf in model_files:
        base = os.path.splitext(os.path.basename(wf))[0]
        kwargs = mapping.get(base, {})
        print(f"Running evaluation for {wf} with mapping {kwargs}")
        out = run_evaluation(wf, model_module=kwargs.get("module"), model_class=kwargs.get("class"))
        results.append({"weights": wf, "result": out})
        print(out)

    # After all runs, build summary CSV from saved experiment JSONs
    build_summary()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run validations for models/*.pth")
    parser.add_argument("--mapping", help="Optional JSON mapping file for model module/class", default=None)
    args = parser.parse_args()

    main(map_file=args.mapping)
