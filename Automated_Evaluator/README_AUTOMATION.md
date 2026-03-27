Automation: validation and metric collection
=========================================

This project includes small automation to run model validations, save per-experiment JSONs, and create a `results/summary.csv` suitable for the Streamlit dashboard.

Files and how to use them
- `run_validations.py`: discover `models/*.pth` files and run evaluation for each via `evaluate.run_evaluation`. It saves per-experiment JSONs in `results/experiments/` (via `utils.evaluator.save_experiment`) and builds `results/summary.csv`.
- `evaluate.py`: exposes `run_evaluation(model_file, model_module=None, model_class=None, dataset_file=None, batch_size=4, device='cuda')`. You can call as a CLI to run a single file.
- `mapping.sample.json`: example mapping from weights basename to module/class when default inference doesn't match.

Example: run validations for all models

```bash
python3 run_validations.py
```

Example: use a mapping file

```bash
python3 run_validations.py --mapping mapping.sample.json
```

Notes
- You can override automatic inference of model module/class by providing `--mapping`.
- The project includes a simple mock model (`models/convlstm.py`) and a mock dataset (`utils/dataset.py`) so you can run a minimal end-to-end evaluation locally. For full evaluations, replace them with your real model/dataset implementations.
