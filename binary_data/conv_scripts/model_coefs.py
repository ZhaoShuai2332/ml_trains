import os
import numpy as np

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PARAM_DIR = os.path.join(BASE_DIR, 'params')
BINARY_BASE = os.path.join(BASE_DIR, 'binary_data', 'params')

# Configuration
DATASETS = ['breast_cancer', 'parkinsons', 'connectionist_bench', 'spambase']
MODELS = {
    'linear': ['min_max_scaler', 'coef', 'intercept'],
    'svr': ['min_max_scaler', 'support_vectors', 'dual_coef', 'gamma']
}
SCALES = ['', '_scale']


def _write_binary(output_dir, name, arr):
    """Write NumPy array to binary file with given name."""
    path = os.path.join(output_dir, f"0_{name}")
    with open(path, 'wb') as f:
        f.write(arr.tobytes())


def _handle_min_max(npz, dataset, output_dir, suffix):
    # Extract scaler components
    for comp in ('min', 'scale', 'data_min', 'data_max'):
        _write_binary(output_dir, f"{comp}_{dataset}{suffix}", npz[comp])


def process_dataset(dataset):
    out_dir = os.path.join(BINARY_BASE, dataset)
    os.makedirs(out_dir, exist_ok=True)

    for model, params in MODELS.items():
        for suffix in SCALES:
            for param in params:
                filename = f"{param}_{dataset}{suffix}_{model}.npz" if param != 'min_max_scaler' else f"min_max_scaler_{dataset}_{model}.npz"
                filepath = os.path.join(PARAM_DIR, filename)
                data = np.load(filepath)

                if param == 'min_max_scaler':
                    _handle_min_max(data, dataset, out_dir, '')
                else:
                    key = param if param in data else list(data.keys())[0]
                    _write_binary(out_dir, f"{param}_{dataset}{suffix}", data[key])

        # Handle kernel_params for SVR
        if model == 'svr':
            for suffix in SCALES:
                kp_file = os.path.join(PARAM_DIR, f"kernel_params_{dataset}{suffix}_svr.npz")
                kp_data = np.load(kp_file)
                for comp in ('kernel', 'C', 'epsilon'):
                    _write_binary(out_dir, f"{comp}_{dataset}{suffix}", kp_data[comp])


if __name__ == '__main__':
    for ds in DATASETS:
        process_dataset(ds)
