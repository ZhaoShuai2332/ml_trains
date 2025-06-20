import os, sys
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from scripts.Fixpoint import FixedPoint
import numpy as np


def sign_bit(x: np.ndarray) -> np.ndarray:
    '''Extract MSB sign bit: 0 if x >= 0, 1 if x < 0.'''
    arr = np.asarray(x)
    bits = arr.dtype.itemsize * 8
    uint_type = f'uint{bits}'
    arr_uint = arr.view(uint_type)
    return ((arr_uint >> (bits - 1)) & 1).astype(int)

def sign_bit_fix(x: FixedPoint) -> int:
    '''Extract MSB sign bit: 0 if x >= 0, 1 if x < 0.'''
    if not isinstance(x, FixedPoint):
        raise TypeError("The input must be a FixedPoint object")
    sign_mask = x.sign_bit
    
    return 1 if (x.raw & sign_mask) else 0
    

def logistic_pred(linear_pred: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    '''
    Compute sigmoid probabilities and binary decision using threshold t.
    Returns tuple (probabilities, binary_preds).
    '''
    prob = 1 / (1 + np.exp(-linear_pred))
    binary = sign_bit(prob - t)
    return binary


def f_logr_pred(linear_pred: np.ndarray, t: float) -> np.ndarray:
    '''
    Compute binary decision based on linear_pred and logit threshold log(t/(1-t)).
    Returns binary_preds.
    '''
    with np.errstate(divide='ignore'):
        thresh = np.log(t / (1.0 - t))
    return sign_bit(linear_pred - thresh)

def f_logr_fix(linear_pred: np.ndarray, tresh: FixedPoint) -> np.ndarray:
    '''
    Compute binary decision based on linear_pred and logit threshold log(t/(1-t)).
    Returns binary_preds.
    '''
    result = []
    for pred in linear_pred:
        if not isinstance(pred, FixedPoint):
            raise TypeError("Each element in linear_pred must be a FixedPoint object")
        result.append(sign_bit_fix(pred - tresh))
    return np.array(result)

def convert_t(t_list : np.ndarray) -> np.ndarray:
    '''
    Convert t_list to logit threshold.
    '''
    return np.log(t_list / (1.0 - t_list))




