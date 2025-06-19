import numpy as np


def sign_bit(x: np.ndarray) -> np.ndarray:
    '''Extract MSB sign bit: 0 if x >= 0, 1 if x < 0.'''
    arr = np.asarray(x)
    bits = arr.dtype.itemsize * 8
    uint_type = f'uint{bits}'
    arr_uint = arr.view(uint_type)
    return ((arr_uint >> (bits - 1)) & 1).astype(int)


def logistic_pred(linear_pred: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    '''
    Compute sigmoid probabilities and binary decision using threshold t.
    Returns tuple (probabilities, binary_preds).
    '''
    # print(linear_pred)
    prob = 1 / (1 + np.exp(-linear_pred))
    # print(prob)
    binary = sign_bit(prob - t)
    # binary = sign_bit(prob - 0.5)
    return prob, binary


def f_logr_pred(linear_pred: np.ndarray, t: float) -> np.ndarray:
    '''
    Compute binary decision based on linear_pred and logit threshold log(t/(1-t)).
    Returns binary_preds.
    '''
    with np.errstate(divide='ignore'):
        thresh = np.log(t / (1.0 - t))
    return sign_bit(linear_pred - thresh)

def convert_t(t_list : np.ndarray) -> np.ndarray:
    '''
    Convert t_list to logit threshold.
    '''
    return np.log(t_list / (1.0 - t_list))




