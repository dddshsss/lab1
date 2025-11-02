import numpy as np

def chess(m, n, a, b):

    i_indices, j_indices = np.ogrid[:m, :n]
    chess_pattern = (i_indices + j_indices) % 2
    
    result = np.where(chess_pattern == 0, a, b)
    return result

#Тесты
import numpy as np
import pytest

def chess(rows, cols, value1, value2):
    # Создаем шахматную доску с чередующимися значениями
    board = np.zeros((rows, cols), dtype=type(value1))
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                board[i, j] = value1
            else:
                board[i, j] = value2
    return board

# Тесты с использованием pytest
def test_chess_2x2():
    """Тест шахматной доски 2x2"""
    result = chess(2, 2, 1, 0)
    expected = np.array([[1, 0], [0, 1]])
    assert np.array_equal(result, expected)

def test_chess_3x3():
    """Тест шахматной доски 3x3"""
    result = chess(3, 3, 1, 0)
    expected = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    assert np.array_equal(result, expected)

def test_chess_1x1():
    """Тест матрицы 1x1"""
    result = chess(1, 1, 5, 10)
    expected = np.array([[5]])
    assert np.array_equal(result, expected)

def test_chess_1x5():
    """Тест горизонтальной матрицы 1x5"""
    result = chess(1, 5, 1, 0)
    expected = np.array([[1, 0, 1, 0, 1]])
    assert np.array_equal(result, expected)

def test_chess_5x1():
    """Тест вертикальной матрицы 5x1"""
    result = chess(5, 1, 1, 0)
    expected = np.array([[1], [0], [1], [0], [1]])
    assert np.array_equal(result, expected)

def test_chess_4x4_custom_values():
    """Тест большой матрицы с пользовательскими значениями"""
    result = chess(4, 4, 7, 3)
    expected = np.array([[7, 3, 7, 3], [3, 7, 3, 7], [7, 3, 7, 3], [3, 7, 3, 7]])
    assert np.array_equal(result, expected)

def test_chess_float_values():
    """Тест с вещественными числами"""
    result = chess(2, 3, 1.5, 2.5)
    expected = np.array([[1.5, 2.5, 1.5], [2.5, 1.5, 2.5]])
    assert np.allclose(result, expected)

def test_chess_boolean_values():
    """Тест с булевыми значениями"""
    result = chess(2, 2, True, False)
    expected = np.array([[True, False], [False, True]])
    assert np.array_equal(result, expected)
    assert result.dtype == bool

# Демонстрация работы
if name == "__main__":
    print("Примеры шахматных досок:")
    
    # Пример 2x2
    board_2x2 = chess(2, 2, 1, 0)
    print(f"2x2:\n{board_2x2}")
    
    # Пример 3x3
    board_3x3 = chess(3, 3, 'A', 'B')
    print(f"3x3:\n{board_3x3}")
    
    # Запуск тестов
    print("\nЗапуск тестов...")
    pytest.main([__file__, "-v"])