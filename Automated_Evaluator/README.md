# Automated Evaluator

This repository contains tools to run and evaluate automated model validations and comparisons. It includes data loaders, metrics calculation, evaluation scripts, and test coverage for the project.

## Repository structure

- `compare.py` — utilities to compare model outputs / experiments.
- `dashboard.py` — (optional) dashboard or visualization runner.
- `evaluate.py` — script to run evaluation logic.
- `run_validations.py` — top-level script to run validations or example runs.
- `data/` — data used by the project (input / sample files).
- `models/` — model artifacts and model code.
  - `convlstm.pth`, `gan.pth` — pre-trained model weights (binary).
  - `convlstm.py` — model source code.
- `results/summary.csv` — output summary of recent runs.
- `utils/` — helper modules:
  - `dataset.py` — dataset utilities and loaders.
  - `evaluator.py` — orchestrates evaluation logic.
  - `metrics.py` — metric implementations.
- `tests/` — pytest test suite covering metrics and end-to-end smoke tests.

## Quickstart (macOS / zsh)

1. Create and activate a Python virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies. If your project has a `requirements.txt`, use it. If not, install the common dependencies used by this repository (adjust versions as needed):

```bash
# If a requirements file exists
pip install -r requirements.txt

# Otherwise install common packages used by ML projects
pip install numpy pandas pytest torch torchvision
```

3. Run validations / example run

```bash
python run_validations.py
```

Depending on the script arguments and implementation, you may need to pass additional flags. Check inline script help (e.g. `python evaluate.py -h`) for details.

## Running tests

Run the full test suite with pytest:

```bash
python -m pytest -q
```

There are unit tests in `tests/` including:

- `test_metrics.py` — metric unit tests
- `test_e2e_evaluate.py` — end-to-end evaluation tests
- `test_smoke_run_validations.py` — smoke tests for `run_validations.py`

## Results and models

- Generated run output is saved under `results/` (for example `results/summary.csv`).
- Pre-trained model weights are stored in `models/*.pth`. Keep these files under source control only if they are small and allowed by your project policy.

## Notes and troubleshooting

- If tests fail, run an individual test file for targeted output: `python -m pytest tests/test_metrics.py -q`.
- If you see import errors, ensure your virtual environment is activated and dependencies are installed.
- If GPU-specific code requires CUDA, configure the appropriate CUDA toolkit and compatible `torch` package.

## Contributing

If you'd like to contribute, please open issues or pull requests describing the change. Keep tests green and add tests for new functionality.

## License

Specify your project license here (e.g., MIT). If a license file exists, follow its terms.

---

This README is intended to give a quick onboarding guide. Update sections with project-specific command-line flags and dependency information as needed.
