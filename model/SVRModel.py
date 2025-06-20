""" 
Module: svr_model_trainer

Provides utilities for training, evaluating, and persisting Support Vector Regression models
on specified UCI benchmark datasets.

Features
--------
- SVR_Train: fetches data, optionally scales features, trains SVR model,
  computes evaluation metrics, and saves model parameters separately.
- SVRModel: loads pre-trained SVR parameters (support vectors, dual coefficients, etc.),
  applies scaling if used, and offers a predict method for inference on the original dataset.

Dependencies
------------
- numpy, pandas
- custom Min_Max_Scaler for manual scaling
- scikit-learn for data splits, SVR, and metrics
- fetch_save function to download and save UCI datasets
- Fixpoint module for fixed-point arithmetic support

Author: Shuai Zhao
Date: 2025-06-18
"""
import os
import sys
# Ensure project root is importable for custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np 
import pandas as pd  
from model.MinMaxScaler_ import Min_Max_Scaler  
from data.fetch_save_data import fetch_save  
from sklearn.svm import SVR  
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  
from scripts.Fixpoint import parse_float_to_fixed_array

def SVR_Train(
    name: str,
    is_scale: bool = False,
    l: float = 0.0,
    r: float = 1.0,
    kernel: str = 'linear',  # Set default kernel to 'linear'
    C: float = 1.0,
    gamma: str = 'scale',
    epsilon: float = 0.1
) -> None:
    """
    Train and evaluate a Support Vector Regression model on a given dataset.

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
    kernel : str, default='linear'
        Specifies the kernel type to be used in the algorithm.
    C : float, default=1.0
        Regularization parameter for SVR.
    gamma : str or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    epsilon : float, default=0.1
        Epsilon parameter in the epsilon-SVR model.

    Side Effects
    ------------
    - Fetches features X and targets y via fetch_save(name).
    - Optionally scales X and saves scaler parameters to params/ folder.
    - Fits SVR model and computes R², MSE, MAE metrics.
    - Persists SVR model parameters separately in .npz files under params/:
      support vectors, dual coefficients, intercept, gamma, and kernel parameters.
    """
    # Instantiate the Support Vector Regression model
    model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon, max_iter=10000)
    # Load data
    X, y = fetch_save(name)

    if is_scale:
        # Apply scikit-learn MinMax scaling
        scaler = MinMaxScaler(feature_range=(l, r))
        X = scaler.fit_transform(X)
        # Persist scaling parameters for consistent future transformations
        np.savez(
            f"params/min_max_scaler_{name}_svr.npz",
            min=scaler.min_,
            scale=scaler.scale_,
            data_min=scaler.data_min_,
            data_max=scaler.data_max_
        )

    # Train the model on full dataset
    print("Training the SVR model...")
    model.fit(X, y.ravel())
    # Generate predictions for evaluation
    y_pred = model.predict(X).flatten()
    print("Done!")

    # Compute evaluation metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print(f"\nSVR Model Evaluation Metrics on {name} dataset:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}\n")

    os.makedirs("params", exist_ok=True)

    suffix = "_scale" if is_scale else ""
    
    support_vectors = X[model.support_]
    np.savez(f"params/support_vectors_{name}{suffix}_svr.npz", support_vectors=support_vectors)
    
    dual_coef = model.dual_coef_.flatten()
    np.savez(f"params/dual_coef_{name}{suffix}_svr.npz", dual_coef=dual_coef)
    
    intercept = model.intercept_
    np.savez(f"params/intercept_{name}{suffix}_svr.npz", intercept=intercept)
    
    if hasattr(model, 'gamma') and model.gamma != 'scale' and model.gamma != 'auto':
        gamma_value = model.gamma
    else:
        if model.gamma == 'scale':
            gamma_value = 1.0 / (X.shape[1] * X.var())
        else:
            gamma_value = 1.0 / X.shape[1]
    np.savez(f"params/gamma_{name}{suffix}_svr.npz", gamma=gamma_value)
    
    # Save kernel type and other parameters
    np.savez(f"params/kernel_params_{name}{suffix}_svr.npz",  
             kernel=model.kernel, C=model.C, epsilon=model.epsilon)


class SVRModel:
    """
    Wrapper for loading a trained Support Vector Regression model and performing prediction.

    Upon initialization, retrieves feature matrix and target vector, applies optional
    manual scaling, and loads saved SVR model for inference.
    """
    def __init__(
        self,
        name: str,
        is_scale: bool = False,
        is_fixpoint: bool = False
    ):
        """
        Initialize the SVRModel with dataset features, targets, and trained model.

        Parameters
        ----------
        name : str
            Identifier used to fetch data and locate saved model.
        is_scale : bool, optional
            Whether the model was trained on scaled inputs.
        is_fixpoint: bool, optional
            Whether to use fixed-point arithmetic for predictions.
        """
        # Store fixpoint flag for use in predict method
        self.is_fixpoint = is_fixpoint
        
        # Load raw data
        self.X, self.y = fetch_save(name)

        if is_scale:
            print("Scaling the data...")
            # Apply custom Min_Max_Scaler to the data
            scaler = Min_Max_Scaler(name, 'svr')
            self.X = scaler.scaler_x(self.X)
            suffix = "_scale"
        else:
            suffix = ""
            
        # Load SVR model parameters from separate npz files
        support_vectors_data = np.load(f"params/support_vectors_{name}{suffix}_svr.npz")
        self.support_vectors = support_vectors_data["support_vectors"]
        
        dual_coef_data = np.load(f"params/dual_coef_{name}{suffix}_svr.npz")
        self.dual_coef = dual_coef_data["dual_coef"]
        
        intercept_data = np.load(f"params/intercept_{name}{suffix}_svr.npz")
        self.intercept = intercept_data["intercept"]
        
        gamma_data = np.load(f"params/gamma_{name}{suffix}_svr.npz")
        self.gamma = gamma_data["gamma"]
        
        kernel_params_data = np.load(f"params/kernel_params_{name}{suffix}_svr.npz", allow_pickle=True)
        self.kernel = str(kernel_params_data["kernel"])
        self.C = float(kernel_params_data["C"])
        self.epsilon = float(kernel_params_data["epsilon"])
             
        if is_fixpoint:
            print("Converting to fixed-point...")
            self.X = parse_float_to_fixed_array(self.X)
            self.y = parse_float_to_fixed_array(self.y)
            # Note: SVR model parameters cannot be directly converted to fixed-point
            # as they involve complex kernel computations

    def _linear_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute Linear kernel between X1 and X2.
        
        Parameters
        ----------
        X1 : np.ndarray
            First set of samples.
        X2 : np.ndarray
            Second set of samples.
            
        Returns
        -------
        np.ndarray
            Kernel matrix.
        """
        # Compute linear kernel: K(x1, x2) = x1 * x2^T
        return np.dot(X1, X2.T)
    
    def predict(self) -> np.ndarray:
        """
        Compute predictions for the loaded dataset using saved SVR parameters.
        
        Implements the SVM regression formula:
        f_svr(z) = Σ(α_j * K(x_j, z)) + b
        where α_j are dual coefficients, x_j are support vectors, and K is the kernel function.

        Returns
        -------
        np.ndarray
            Predicted target values, flattened to one-dimensional array.
            If is_fixpoint=True, returns array of FixedPoint objects.
        """
        if hasattr(self, 'X') and self.X is not None:
            # Convert fixed-point back to float for SVR prediction if needed
            if self.is_fixpoint:
                from scripts.Fixpoint import parse_fixed_to_float_array
                X_float = parse_fixed_to_float_array(self.X)
            else:
                X_float = self.X
            
            # Manual SVR prediction using saved parameters
            predictions = []
            
            for x_query in X_float:
                # Compute kernel values between query point and all support vectors
                if self.kernel == 'linear':
                    kernel_values = self._linear_kernel(x_query.reshape(1, -1), self.support_vectors)
                    kernel_values = kernel_values.flatten()
                else:
                    raise ValueError(f"Unsupported kernel type: {self.kernel}")
                
                # Compute prediction: Σ(α_j * K(x_j, z)) + b
                prediction = np.sum(self.dual_coef * kernel_values) + self.intercept
                predictions.append(prediction)
            
            predictions_array = np.array(predictions).flatten()
            
            # Convert to FixedPoint if this is a fixpoint model
            if self.is_fixpoint:
                predictions_array = parse_float_to_fixed_array(predictions_array)
            
            print("Done!")
            return predictions_array
        else:
            raise ValueError("No data available for prediction")
