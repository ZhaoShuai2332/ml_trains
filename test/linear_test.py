import os,sys
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from model.LinearModel import LinearModel
from scripts.run_proms import parse_args
import numpy as np
def LinearTest():
    args = parse_args()
    name = args.name 
    is_scale = args.scale
    is_gridsearch = args.gridsearch
    print(is_gridsearch)

    linear = LinearModel(name, is_scale)
    preds = linear.predict()
    if is_gridsearch:
        t_list = np.linspace(0.001, 1.000, 1000)
    else:
        np.random.seed(42)
        t_list = np.random.uniform(0.001, 1.000, 1000)
    
    

if __name__ == "__main__":
    LinearTest()