import numpy as np

def sum_prod(X, V):
    if len(X) == 0:
        return np.array([])
    X_array = np.array(X)
    V_array = np.array(V)
    res = np.sum(X_array @ V_array, axis=0)

    return res