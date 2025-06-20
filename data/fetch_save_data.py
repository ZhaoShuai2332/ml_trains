"""
Module: fetch_save_data

Provides utilities to download benchmark datasets from the UCI Machine Learning Repository,
preprocess them with label encoding, and save them locally as CSV files.

This module emphasizes:
- Reproducibility: all parameters and outputs are explicitly documented.
- Clarity: docstrings follow NumPy/Google style for easy reference.
- Robustness: basic error handling for missing datasets and directories.
- Data preprocessing: automatic label encoding for categorical features and targets.

Author: Shuai Zhao
Date: 2025-06-12
"""

import os
import sys

# ------------------------------------------------------------------------------
# Ensure the project root is on the PYTHONPATH so that ucimlrepo can be imported
# ------------------------------------------------------------------------------
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def fetch_data(id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download a dataset from the UCI ML Repository.

    Parameters
    ----------
    id : int
        Unique dataset identifier as specified by the UCI repository.

    Returns
    -------
    X : pandas.DataFrame
        DataFrame of explanatory variables (features) in original format.
    y : pandas.DataFrame
        DataFrame of the response variable (targets) in original format.

    Notes
    -----
    - The fetch_ucirepo function uses HTTP to retrieve data; verify network access.
    - Column names and data types are preserved exactly as provided by UCI.
    - No preprocessing is applied at this stage.
    """
    # Retrieve the repository response object
    res_data = fetch_ucirepo(id=id)
    
    # Extract feature matrix and target vector
    X = res_data.data.features
    y = res_data.data.targets
    
    return X, y


def save_data(X: pd.DataFrame, y: pd.DataFrame, name: str):
    """
    Preprocess and save feature and target DataFrames to CSV files in dataset-specific directories.

    This function applies label encoding to categorical features and targets, then saves
    the processed data to organized directory structure under ./data/{name}/.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix to preprocess and write.
    y : pandas.DataFrame
        Target vector to preprocess and write.
    name : str
        Dataset name used for creating subdirectory and filenames.

    Raises
    ------
    OSError
        If the data directory cannot be created or written to.

    Notes
    -----
    - Categorical features (object dtype) are automatically label-encoded.
    - Target variables are also label-encoded if they contain categorical data.
    - Original data is preserved by creating copies before processing.
    - Files are saved without row indices for clean downstream analysis.
    """
    # Create dataset-specific directory if it does not exist
    os.makedirs(f"data/{name}", exist_ok=True)
    
    # Create copies to avoid modifying original data
    X_processed = X.copy()
    y_processed = y.copy()
    
    # Encode categorical features using label encoding
    label_encoders = {}
    for column in X_processed.columns:
        if X_processed[column].dtype == 'object':
            le = LabelEncoder()
            X_processed[column] = le.fit_transform(X_processed[column].astype(str))
            label_encoders[column] = le
    
    # Encode target variable if categorical
    if y_processed.iloc[:, 0].dtype == 'object':
        le_target = LabelEncoder()
        y_processed.iloc[:, 0] = le_target.fit_transform(y_processed.iloc[:, 0].astype(str))
        label_encoders['target'] = le_target
    
    # Save preprocessed data to CSV files in dataset-specific directory
    X_processed.to_csv(f"data/{name}/X.csv", index=False)
    y_processed.to_csv(f"data/{name}/y.csv", index=False)


def fetch_save(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch a named dataset by its human‐readable key, preprocess it, and save locally.
    
    This function serves as the main entry point for dataset acquisition. It checks
    for existing processed data first, and only fetches from UCI if needed.

    Supported dataset_name keys and their UCI IDs:
      - "breast_cancer"       → 17  (Wisconsin Breast Cancer Database)
      - "parkinsons"          → 174 (Parkinsons Disease Dataset)
      - "connectionist_bench" → 151 (Connectionist Bench Dataset)
      - "spambase"            → 94  (Spambase Dataset)

    Parameters
    ----------
    dataset_name : str
        Key identifying which dataset to fetch, preprocess, and save.

    Returns
    -------
    X : pandas.DataFrame
        Preprocessed feature matrix loaded from CSV.
    y : pandas.DataFrame
        Preprocessed target vector loaded from CSV.

    Raises
    ------
    KeyError
        If the provided dataset_name is not in the supported list.
    """
    # Mapping from friendly names to UCI numeric IDs
    id_dict = {
        "breast_cancer": 17,
        "parkinsons": 174,
        "connectionist_bench": 151,
        "spambase": 94,
    }
    
    # Validate the dataset_name
    if dataset_name not in id_dict:
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. "
            f"Valid options: {list(id_dict.keys())}"
            )
    
    # Check if preprocessed data already exists locally
    if os.path.exists(f"data/{dataset_name}/X.csv") and os.path.exists(f"data/{dataset_name}/y.csv"):
        print(f"Data for {dataset_name} already exists")
        print(f"Loading data for {dataset_name} from ./data/{dataset_name}")
        X = pd.read_csv(f"data/{dataset_name}/X.csv")
        y = pd.read_csv(f"data/{dataset_name}/y.csv")
    else:
        # Fetch raw data from UCI, preprocess, and save
        X, y = fetch_data(id_dict[dataset_name])
        save_data(X, y, dataset_name)
        
        # Load the preprocessed data that was just saved
        X = pd.read_csv(f"data/{dataset_name}/X.csv")
        y = pd.read_csv(f"data/{dataset_name}/y.csv")
        
        # Inform the user of successful completion
        print(
            f"Saved '{dataset_name}' → "
            f"data/{dataset_name}/X.csv & data/{dataset_name}/y.csv"
        )
    
    return X.values, y.values
