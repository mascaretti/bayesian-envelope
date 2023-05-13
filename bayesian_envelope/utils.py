import numpy as np

def col_max_positive(a: np.ndarray) -> np.ndarray:
    '''
    Change the sign according to the maximum value
    '''
    a = a * np.sign(a[np.abs(a) == np.max(np.abs(a))])
    return a

def sign_update(M: np.ndarray) -> np.ndarray:
    '''
    Change the sign of the columns
    '''
    return np.apply_along_axis(func1d=col_max_positive, axis=0, arr=M)
