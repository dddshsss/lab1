3 задача 

import numpy as np

def get_unique_rows(matrix):
    if matrix.ndim != 2:
        raise ValueError("Матрица должна быть двумерной")
    
    return [np.unique(matrix[i, :]) for i in range(matrix.shape[0])]

def get_unique_columns(matrix):
    if matrix.ndim != 2:
        raise ValueError("Матрица должна быть двумерной")
    
    return [np.unique(matrix[:, j]) for j in range(matrix.shape[1])]

#Тесты 

import pytest
import numpy as np
from fromnum3 import get_unique_rows, get_unique_columns


class TestUniqueRows:
    """Тесты для функции get_unique_rows - получения уникальных элементов по строкам"""
    
    def test_basic_matrix(self):
        """Тест базового случая с дубликатами в строках"""
        matrix = np.array([
            [1, 2, 2, 3],      # уникальные: 1, 2, 3
            [4, 4, 5, 5],      # уникальные: 4, 5
            [6, 7, 7, 8]       # уникальные: 6, 7, 8
        ])
        result = get_unique_rows(matrix)
        expected = [
            np.array([1, 2, 3]),
            np.array([4, 5]),
            np.array([6, 7, 8])
        ]
        for res, exp in zip(result, expected):
            assert np.array_equal(res, exp), f"Ожидалось {exp}, получено {res}"

    def test_all_unique_elements(self):
        """Тест матрицы со всеми уникальными элементами"""
        matrix = np.array([
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90]
        ])
        result = get_unique_rows(matrix)
        expected = [
            np.array([10, 20, 30]),
            np.array([40, 50, 60]),
            np.array([70, 80, 90])
        ]
        for res, exp in zip(result, expected):
            assert np.array_equal(res, exp)

    def test_duplicates_in_rows(self):
        """Тест строк с полными дубликатами"""
        matrix = np.array([
            [5, 5, 5],         # уникальные: 5
            [8, 8, 8],         # уникальные: 8
            [9, 9, 9]          # уникальные: 9
        ])
        result = get_unique_rows(matrix)
        expected = [
            np.array([5]),
            np.array([8]),
            np.array([9])
        ]
        for res, exp in zip(result, expected):
            assert np.array_equal(res, exp)

    def test_negative_numbers(self):
        """Тест с отрицательными числами и нулями"""
        matrix = np.array([
            [-5, 0, -5],       # уникальные: -5, 0
            [0, 10, 0],        # уникальные: 0, 10
            [-3, -3, 7]        # уникальные: -3, 7
        ])
        result = get_unique_rows(matrix)
        expected = [
            np.array([-5, 0]),
            np.array([0, 10]),
            np.array([-3, 7])
        ]
        for res, exp in zip(result, expected):
            assert np.array_equal(res, exp)
    
    def test_single_element_matrix(self):
        """Тест матрицы с одним элементом"""
        matrix = np.array([[15]])
        result = get_unique_rows(matrix)
        expected = [np.array([15])]
        assert len(result) == 1
        assert np.array_equal(result[0], expected[0])
    
    def test_floats_and_ints(self):
        """Тест со смешанными типами данных (целые и вещественные числа)"""
        matrix = np.array([
            [2, 3.5, 2, 3.5],      # уникальные: 2.0, 3.5
            [4.0, 4, 4.0, 8]       # уникальные: 4.0, 8.0
        ])
        result = get_unique_rows(matrix)
        expected = [
            np.array([2., 3.5]),
            np.array([4., 8.])
        ]
        for res, exp in zip(result, expected):
            assert np.array_equal(res, exp)


class TestUniqueColumns:
    """Тесты для функции get_unique_columns - получения уникальных элементов по столбцам"""
    
    def test_basic_matrix(self):
        """Тест базового случая с дубликатами в столбцах"""
        matrix = np.array([
            [1, 2, 2, 3],
            [4, 4, 5, 5],
            [6, 7, 7, 8]
        ])
        result = get_unique_columns(matrix)
        expected = [
            np.array([1, 4, 6]),    # столбец 0: 1,4,6
            np.array([2, 4, 7]),    # столбец 1: 2,4,7
            np.array([2, 5, 7]),    # столбец 2: 2,5,7
            np.array([3, 5, 8])     # столбец 3: 3,5,8
        ]
        for res, exp in zip(result, expected):
            assert np.array_equal(res, exp)

    def test_all_unique_elements_columns(self):
        """Тест матрицы со всеми уникальными элементами в столбцах"""
        matrix = np.array([
            [10, 40, 70],
            [20, 50, 80],
            [30, 60, 90]
        ])
        result = get_unique_columns(matrix)
        expected = [
            np.array([10, 20, 30]),
            np.array([40, 50, 60]),
            np.array([70, 80, 90])
        ]
        for res, exp in zip(result, expected):
            assert np.array_equal(res, exp)

    def test_duplicates_in_columns(self):
        """Тест столбцов с полными дубликатами"""
        matrix = np.array([
            [5, 8, 12],
            [5, 8, 12],
            [5, 8, 12]
        ])
        result = get_unique_columns(matrix)
        expected = [
            np.array([5]),
            np.array([8]),
            np.array([12])
        ]
        for res, exp in zip(result, expected):
            assert np.array_equal(res, exp)

    def test_single_column_matrix(self):
        """Тест матрицы с одним столбцом"""
        matrix = np.array([[2], [4], [4], [6]])
        result = get_unique_columns(matrix)
        expected = [np.array([2, 4, 6])]
        assert len(result) == 1, "Должен быть один столбец в результате"
        assert np.array_equal(result[0], expected[0])
    
    def test_string_elements(self):
        """Тест со строковыми элементами"""
        matrix = np.array([
            ['apple', 'banana', 'banana'],
            ['cherry', 'cherry', 'date']
        ])
        result = get_unique_columns(matrix)
        expected = [
            np.array(['apple', 'cherry']),
            np.array(['banana', 'cherry']),
            np.array(['banana', 'date'])
        ]
        for res, exp in zip(result, expected):
            assert np.array_equal(res, exp)


class TestEdgeCases:
    """Тесты граничных случаев и обработки ошибок"""
    
    def test_empty_matrix(self):
        """Тест пустой матрицы"""
        matrix = np.array([]).reshape(0, 0)result_rows = get_unique_rows(matrix)
        result_cols = get_unique_columns(matrix)
        assert result_rows == [], "Для пустой матрицы должен возвращаться пустой список"
        assert result_cols == [], "Для пустой матрицы должен возвращаться пустой список"
    
    def test_one_dimensional_array(self):
        """Тест обработки одномерного массива (должна быть ошибка)"""
        matrix = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="Матрица должна быть двумерной"):
            get_unique_rows(matrix)
        with pytest.raises(ValueError, match="Матрица должна быть двумерной"):
            get_unique_columns(matrix)
    
    def test_three_dimensional_array(self):
        """Тест обработки трехмерного массива (должна быть ошибка)"""
        matrix = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        with pytest.raises(ValueError, match="Матрица должна быть двумерной"):
            get_unique_rows(matrix)