import os
import sys

# Setup base directory and ensure imports work
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(BASE_DIR)

from data.fetch_save_data import fetch_save
from model.MinMaxScaler_ import Min_Max_Scaler

# Configuration
DATASETS = [
    "breast_cancer",
    "parkinsons",
    "connectionist_bench",
    "spambase"
]
MODELS = ["linear", "svr"]

# Directory for writing binary data
DATA_BIN_DIR = os.path.join(BASE_DIR, "binary_data", "data")


def _save_binary(X, y, out_dir):
    """Write feature and target arrays to binary files in the specified directory."""
    os.makedirs(out_dir, exist_ok=True)
    for fname, arr in (("0-queries.in", X), ("0-relevance.in", y)):
        path = os.path.join(out_dir, fname)
        with open(path, 'wb') as f:
            f.write(arr.tobytes())


def process_raw():
    """Fetch each dataset and save raw feature/target arrays."""
    for ds in DATASETS:
        print(f"Converting raw data for: {ds}...")
        X, y = fetch_save(ds)
        out_dir = os.path.join(DATA_BIN_DIR, ds)
        _save_binary(X, y, out_dir)


def process_scaled():
    """Fetch, scale, and save arrays for each dataset-model combination."""
    for model in MODELS:
        for ds in DATASETS:
            print(f"Converting scaled data for: {ds} ({model})...")
            X, y = fetch_save(ds)
            X_scaled = Min_Max_Scaler(ds, model).scaler_x(X)
            out_dir = os.path.join(DATA_BIN_DIR, f"{ds}_{model}_scale")
            _save_binary(X_scaled, y, out_dir)


if __name__ == "__main__":
    process_raw()
    process_scaled()
