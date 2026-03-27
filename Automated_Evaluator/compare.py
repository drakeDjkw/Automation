import os
import json
import pandas as pd

def load_experiments(folder="results/experiments"):
    records = []

    for file in os.listdir(folder):
        if file.endswith(".json"):
            with open(os.path.join(folder, file)) as f:
                data = json.load(f)

                record = {
                    "id": data["experiment_id"],
                    "model": data["config"]["model"],
                    "MAE": data["metrics"]["MAE"],
                    "RMSE": data["metrics"]["RMSE"],
                    "runtime": data["metrics"]["runtime_sec"]
                }

                records.append(record)

    return pd.DataFrame(records)


df = load_experiments()
df = df.sort_values("RMSE")

print(df)

# Save summary
df.to_csv("results/summary.csv", index=False)