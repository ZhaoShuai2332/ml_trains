import os,sys
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from model.SVRModel import SVRModel
from scripts.run_proms import parse_args
import numpy as np
from scripts.calu import f_logr_pred, convert_t, f_logr_fix
from scripts.Fixpoint import parse_fixed_to_float_array, parse_float_to_fixed_array

def SVRTest():
    args = parse_args()
    name = args.name
    is_scale = args.scale
    is_gridsearch = args.gridsearch
    
    if is_gridsearch:
        t_list = np.linspace(0.01, 0.99, 99)
    else:
        np.random.seed(42)
        t_list = np.random.uniform(0.01, 0.99, 99)

    # Float Point condition
    svr = SVRModel(name, is_scale)
    preds = svr.predict()
    logr_list = np.array([f_logr_pred(preds, t) for t in t_list])

    # FixPoint condition
    logr_t_list = convert_t(t_list)
    logr_t_list = parse_float_to_fixed_array(logr_t_list)
    svr_fix = SVRModel(name, is_scale, is_fixpoint=True)
    preds_fix = svr_fix.predict()
    print(preds_fix)
    logr_list_fix = np.array([f_logr_fix(preds_fix, t) for t in logr_t_list])
    fix_float = parse_fixed_to_float_array(preds_fix)

    # Print result
    print(np.max(np.abs(fix_float - preds)))
    print(np.array([np.sum(logr_list_fix[i] == logr_list[i]) for i in range(0, 99)]))

if __name__ == "__main__":
    SVRTest()