import numpy as np

def binarize(M, threshold=0.5):
    M = np.array(M)
    binary_matrix = (M > threshold).astype(int)
    return binary_matrix


#Тесты
import pytest
import numpy as np

def binarize(M, threshold=0.5):
    # Преобразуем в numpy массив
    M = np.array(M)
    # Создаем бинарную матрицу: 1 если значение > порога, иначе 0
    binary_matrix = (M > threshold).astype(int)
    return binary_matrix

# Тесты для функции binarize
def test_basic_functionality():
    # Проверка базовой работы функции
    matrix = np.array([[0.1, 0.6], [0.4, 0.9]])
    result = binarize(matrix, 0.5)
    expected = np.array([[0, 1], [0, 1]])
    assert np.array_equal(result, expected)

def test_custom_threshold():
    # Проверка с разными пороговыми значениями
    matrix = np.array([[0.3, 0.7], [0.5, 0.8]])
    result_low = binarize(matrix, 0.2)
    result_high = binarize(matrix, 0.6)
    
    assert np.array_equal(result_low, np.array([[1, 1], [1, 1]]))
    assert np.array_equal(result_high, np.array([[0, 1], [0, 1]]))

def test_all_zeros_and_ones():
    # Проверка крайних случаев
    zeros_matrix = np.array([[0.1, 0.2], [0.3, 0.4]])
    ones_matrix = np.array([[0.6, 0.7], [0.8, 0.9]])
    
    zeros_result = binarize(zeros_matrix, 0.5)
    ones_result = binarize(ones_matrix, 0.5)
    
    assert np.array_equal(zeros_result, np.array([[0, 0], [0, 0]]))
    assert np.array_equal(ones_result, np.array([[1, 1], [1, 1]]))

def test_list_input():
    # Проверка работы с обычным списком
    list_input = [[0.1, 0.9], [0.4, 0.6]]
    result = binarize(list_input, 0.5)
    expected = np.array([[0, 1], [0, 1]])
    assert np.array_equal(result, expected)

def test_default_threshold():
    # Проверка значения по умолчанию
    matrix = np.array([[0.4, 0.6], [0.5, 0.5]])
    result = binarize(matrix)  # без указания порога
    expected = np.array([[0, 1], [0, 0]])
    assert np.array_equal(result, expected)

def test_single_element():
    # Проверка с одним элементом
    single = np.array([[0.7]])
    result = binarize(single, 0.5)
    assert np.array_equal(result, np.array([[1]]))

# Запуск тестов
if name == "__main__":
    # Демонстрация работы
    print("Примеры работы функции binarize:")
    
    test_matrix = np.array([[0.1, 0.6, 0.3],
                           [0.8, 0.4, 0.9]])
    print(f"Исходная матрица:\n{test_matrix}")
    print(f"Бинаризация с порогом 0.5:\n{binarize(test_matrix, 0.5)}")
    
    # Запуск pytest
    pytest.main([__file__, "-v"])