import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

from data.fetch_save_data import fetch_save
from model.MinMaxScaler_ import Min_Max_Scaler

dataset_list = ["breast_cancer", "parkinsons", "connectionist_bench", "spambase"]

def save_data_binary():
    for name in dataset_list:
        print(f"Starting to convert: {name}...")
        X, y = fetch_save(name)
        path = os.path.join(parent_dir, "binary_data", "data", name)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "0-queries.in"), 'wb') as f:
            f.write(X.tobytes())
        with open(os.path.join(path, "0-relevance.in"), 'wb') as f:
            f.write(y.tobytes())

def save_data_binary_scale():
    model_list = ["svr", "linear"]
    for model_name in model_list:
        for name in dataset_list:
            print(f"Starting to convert: {name} in {model_name} with scale...")
            scaler = Min_Max_Scaler(name, model_name)
            X,y = fetch_save(name)
            X = scaler.scaler_x(X)
            path = os.path.join(parent_dir, "binary_data", "data", f"{name}_{model_name}_scale")
            if not os.path.exists(path):
                os.makedirs(path)
            with open(os.path.join(path, "0-queries.in"), 'wb') as f:
                f.write(X.tobytes())
            with open(os.path.join(path, "0-relevance.in"), 'wb') as f:
                f.write(y.tobytes())

if __name__ == "__main__":
    save_data_binary()
    save_data_binary_scale()
            

        