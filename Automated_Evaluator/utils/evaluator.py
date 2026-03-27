import torch
import json
import os
import time
from utils.metrics import mae, rmse

class Evaluator:
    def __init__(self, model, dataloader, device="cuda"):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device

    def run(self):
        self.model.eval()

        mae_list, rmse_list = [], []

        start_time = time.time()

        with torch.no_grad():
            for x, y in self.dataloader:
                x, y = x.to(self.device), y.to(self.device)

                pred = self.model(x)

                y_np = y.cpu().numpy()
                pred_np = pred.cpu().numpy()

                mae_list.append(mae(y_np, pred_np))
                rmse_list.append(rmse(y_np, pred_np))

        results = {
            "MAE": sum(mae_list)/len(mae_list),
            "RMSE": sum(rmse_list)/len(rmse_list),
            "num_samples": len(mae_list),
            "runtime_sec": time.time() - start_time
        }

        return results


def save_experiment(results, config, save_dir="results/experiments"):
    os.makedirs(save_dir, exist_ok=True)

    exp_id = f"exp_{int(time.time())}"
    path = os.path.join(save_dir, exp_id + ".json")

    full_result = {
        "experiment_id": exp_id,
        "timestamp": time.ctime(),
        "config": config,
        "metrics": results
    }

    with open(path, "w") as f:
        json.dump(full_result, f, indent=4)

    return path