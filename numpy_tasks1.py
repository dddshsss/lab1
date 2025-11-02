import numpy as np

def sum_prod(X, V):
    if len(X) == 0:
        return np.array([])
    X_array = np.array(X)
    V_array = np.array(V)
    res = np.sum(X_array @ V_array, axis=0)

    return res

#Тесты
import pytest
import numpy as np
from fromnum1 import sum_prod

class TestSumProd:
    def test_single_pair(self):
        X = [np.array([[1, 2], [3, 4]])]
        V = [np.array([[1], [2]])]
        result = sum_prod(X, V)
        expected = np.array([[5], [11]])
        assert np.allclose(result, expected)

    def test_multiple_pairs(self):
        X = [np.array([[1, 0], [0, 1]]), np.array([[2, 0], [0, 2]])]
        V = [np.array([[1], [2]]), np.array([[3], [4]])]
        result = sum_prod(X, V)
        expected = np.array([[7], [10]])
        assert np.allclose(result, expected)

    def test_3x3_matrices(self):
        X = [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])]
        V = [np.array([[1], [0], [2]])]
        result = sum_prod(X, V)
        expected = np.array([[7], [16], [25]])
        assert np.allclose(result, expected)

    def test_empty_input(self):
        result = sum_prod([], [])
        assert result.shape == (0,)

    def test_large_matrices(self):
        n, p = 50, 5
        X = [np.ones((n, n)) for _ in range(p)]
        V = [np.ones((n, 1)) for _ in range(p)]
        result = sum_prod(X, V)
        expected = np.full((n, 1), n * p)
        assert np.allclose(result, expected)
#Запускать через pytest fromnum1_pytest.py -v