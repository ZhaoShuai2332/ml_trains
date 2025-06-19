import os, sys
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

import numpy as np


class Min_Max_Scaler:
    def __init__(self, name):
        self.param_path = os.path.join(parent_dir, "params", f"min_max_scaler_{name}.npz")
        params_np = np.load(self.param_path)
        self.params = {
            "min": params_np["min"],
            "scale": params_np["scale"],
            "data_min": params_np["data_min"],
            "data_max": params_np["data_max"],
        }
        
    def scaler_x(self, x: np.ndarray):
        x_min = self.params["data_min"]
        x_scale = self.params["scale"]
        x_centered = x - x_min
        x_scaled = x_centered * x_scale
        
        return x_scaled

    