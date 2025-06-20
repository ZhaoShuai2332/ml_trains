"""
Module: linear_model_trainer

Provides utilities for training, evaluating, and persisting linear regression models
on specified UCI benchmark datasets.

Features
--------
- Linear_Train: fetches data, optionally scales features, trains LinearRegression,
  computes evaluation metrics, and saves model parameters.
- LinearModel: loads pre-trained coefficients and intercepts, applies scaling if used,
  and offers a predict method for inference on the original dataset.

Dependencies
------------
- numpy, pandas
- custom Min_Max_Scaler for manual scaling
- scikit-learn for data splits, regression, and metrics
- fetch_save function to download and save UCI datasets

Author: Shuai Zhao
Date: 2025-06-18
"""
import os
import sys
# Ensure project root is importable for custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # Numerical operations and array handling
import pandas as pd  # DataFrame structure
from model.MinMaxScaler_ import Min_Max_Scaler  # Custom scaler for manual scaling
from data.fetch_save_data import fetch_save  # Dataset fetching and saving utility
from sklearn.linear_model import LinearRegression  # Core regression model
from sklearn.preprocessing import MinMaxScaler  # Optional scaling from scikit-learn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # Evaluation metrics
from scripts.Fixpoint import parse_float_to_fixed_array

def Linear_Train(
    name: str,
    is_scale: bool = False,
    l: float = 0.0,
    r: float = 1.0
) -> None:
    """
    Train and evaluate a LinearRegression model on a given dataset.

    Parameters
    ----------
    name : str
        Identifier for the dataset (used by fetch_save).
    is_scale : bool, optional
        Whether to apply MinMax scaling to features before training.
    l : float, default=0.0
        Lower bound for MinMax scaling range.
    r : float, default=1.0
        Upper bound for MinMax scaling range.

    Side Effects
    ------------
    - Fetches features X and targets y via fetch_save(name).
    - Optionally scales X and saves scaler parameters to params/ folder.
    - Fits LinearRegression model and computes R², MSE, MAE metrics.
    - Persists model intercept and coefficients in .npz files under params/.
    """
    # Instantiate the regression model
    model = LinearRegression()
    # Load data
    X, y = fetch_save(name)

    if is_scale:
        # Apply scikit-learn MinMax scaling
        scaler = MinMaxScaler(feature_range=(l, r))
        X = scaler.fit_transform(X)
        # Persist scaling parameters for consistent future transformations
        np.savez(
            f"params/min_max_scaler_{name}.npz",
            min=scaler.min_,
            scale=scaler.scale_,
            data_min=scaler.data_min_,
            data_max=scaler.data_max_
        )

    # Train the model on full dataset
    print("Training the model...")
    model.fit(X, y)
    # Generate predictions for evaluation
    y_pred = model.predict(X).flatten()
    print("Done!")

    # Compute evaluation metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    # Display results
    print(f"\nModel Evaluation Metrics on {name} dataset:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}\n")

    # Ensure parameter directory exists
    os.makedirs("params", exist_ok=True)

    # Persist model parameters, with separate files for scaled vs. unscaled
    suffix = "_scale" if is_scale else ""
    np.savez(f"params/intercept_{name}{suffix}.npz", intercept=model.intercept_)
    np.savez(f"params/coef_{name}{suffix}.npz", coef=model.coef_)


class LinearModel:
    """
    Wrapper for loading a trained linear regression model and performing prediction.

    Upon initialization, retrieves feature matrix and target vector, applies optional
    manual scaling, and loads saved coefficients and intercept for inference.
    """
    def __init__(
        self,
        name: str,
        is_scale: bool = False,
        is_fixpoint: bool = False
    ):
        """
        Initialize the LinearModel with dataset features, targets, and parameters.

        Parameters
        ----------
        name : str
            Identifier used to fetch data and locate saved parameters.
        is_scale : bool, optional
            Whether the model was trained on scaled inputs.
        is_fixpoint: bool, optional
            Whether to use fixed-point arithmetic for predictions.
        """
        # Load raw data
        self.X, self.y = fetch_save(name)

        if is_scale:
            print("Scaling the data...")
            # Apply custom Min_Max_Scaler to the data
            scaler = Min_Max_Scaler(name)
            self.X = scaler.scaler_x(self.X)
            # Load scaled model parameters
            params = np.load(f"params/coef_{name}_scale.npz")
            self.coef_ = params["coef"].flatten()
            self.intercept_ = np.load(f"params/intercept_{name}_scale.npz")["intercept"]
        else:
            # Load unscaled model parameters
            params = np.load(f"params/coef_{name}.npz")
            self.coef_ = params["coef"].flatten()
            self.intercept_ = np.load(f"params/intercept_{name}.npz")["intercept"]
        if is_fixpoint:
            print("Converting to fixed-point...")
            self.X = parse_float_to_fixed_array(self.X)
            self.y = parse_float_to_fixed_array(self.y)
            self.coef_ = parse_float_to_fixed_array(self.coef_)
            self.intercept_ = parse_float_to_fixed_array(self.intercept_)

    def predict(self) -> np.ndarray:
        """
        Compute predictions for the loaded dataset using saved parameters.

        Returns
        -------
        np.ndarray
            Predicted target values, flattened to one-dimensional array.
        """
        # Compute linear combination: X @ coef + intercept
        bias_fp = self.intercept_.item() if isinstance(self.intercept_, np.ndarray) else self.intercept_
        res_list = []
        for x in self.X:
            res = np.dot(x, self.coef_) + bias_fp
            res_list.append(res)
        print("Done!")
        return np.array(res_list).flatten()

