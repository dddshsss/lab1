import numpy as np

def binarize(M, threshold=0.5):
    M = np.array(M)

    binary_matrix = (M > threshold).astype(int)

    return binary_matrix