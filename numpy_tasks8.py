import numpy as np
import matplotlib.pyplot as plt

def one_hot_encoding(labels):
    num_classes = np.max(labels) + 1
    encoding = np.zeros((len(labels), num_classes), dtype=int)
    encoding[np.arange(len(labels)), labels] = 1
    return encoding

# Тесты
def test_one_hot_basic():
    """Тест базового функционала"""
    labels = np.array([0, 2, 3, 0])
    result = one_hot_encoding(labels)
    expected = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0], 
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    assert np.array_equal(result, expected)

def test_one_hot_single_class():
    """Тест с одним классом"""
    labels = np.array([0, 0, 0])
    result = one_hot_encoding(labels)
    expected = np.array([[1], [1], [1]])
    assert np.array_equal(result, expected)

def test_one_hot_sequential():
    """Тест с последовательными классами"""
    labels = np.array([0, 1, 2])
    result = one_hot_encoding(labels)
    expected = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    assert np.array_equal(result, expected)

def test_one_hot_large_gap():
    """Тест с большим разрывом в классах"""
    labels = np.array([0, 5, 2])
    result = one_hot_encoding(labels)
    expected = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0]
    ])
    assert np.array_equal(result, expected)

if name == "__main__":
    # Демонстрация работы
    print("Демонстрация one-hot-encoding:")
    
    # Тест из примера
    vector = np.array([0, 2, 3, 0])
    encoded = one_hot_encoding(vector)
    print("Тест 1:")
    print(f"Вход: {vector}")
    print(f"Выход:\n{encoded}")
    print()
    
    # Дополнительные тесты
    test_cases = [
        [1, 0, 1],
        [0, 0, 0],
        [4, 2, 1, 0, 3]
    ]
    
    for i, test in enumerate(test_cases, 2):
        print(f"Тест {i}:")
        print(f"Вход: {test}")
        print(f"Выход:\n{one_hot_encoding(test)}")
        print()
    
    # Запуск pytest тестов
    import pytest
    pytest.main([__file__, "-v"])