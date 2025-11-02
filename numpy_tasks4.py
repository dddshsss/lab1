import numpy as np
import matplotlib.pyplot as plt


def analyze_normal_matrix(m, n, mean=0, std=1):
    if m <= 0 or n <= 0:
        raise ValueError("Размеры матрицы должны быть положительными числами")
    if std <= 0:
        raise ValueError("Стандартное отклонение должно быть положительным")
    
    matrix = np.random.normal(loc=mean, scale=std, size=(m, n))

    print(f"Матрица {m}x{n} из нормального распределения N({mean}, {std}):")
    print(matrix)
    print("\n" + "="*50 + "\n")

    column_means = np.mean(matrix, axis=0)
    column_variances = np.var(matrix, axis=0)

    print("СТАТИСТИКА ПО СТОЛБЦАМ:")
    for i in range(n):
        print(f"Столбец {i+1}: мат. ожидание = {column_means[i]:.4f}, "
              f"дисперсия = {column_variances[i]:.4f}")

    print("\n" + "="*50 + "\n")

    row_means = np.mean(matrix, axis=1)
    row_variances = np.var(matrix, axis=1)

    print("СТАТИСТИКА ПО СТРОКАМ:")
    for i in range(m):
        print(f"Строка {i+1}: мат. ожидание = {row_means[i]:.4f}, "
              f"дисперсия = {row_variances[i]:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].set_title("Распределение значений по СТРОКАМ")
    for i in range(m):
        axes[0, 0].hist(matrix[i, :], alpha=0.7, label=f"Строка {i+1}",
                        bins=15, density=True, histtype='stepfilled')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel("Значения")
    axes[0, 0].set_ylabel("Плотность вероятности")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Распределение значений по СТОЛБЦАМ")
    for j in range(n):
        axes[0, 1].hist(matrix[:, j], alpha=0.7, label=f"Столбец {j+1}", 
                        bins=15, density=True, histtype='stepfilled')
    axes[0, 1].legend()
    axes[0, 1].set_xlabel("Значения")
    axes[0, 1].set_ylabel("Плотность вероятности")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(matrix.flatten(), bins=20, alpha=0.7, density=True, 
                   color='green', histtype='stepfilled')
    axes[1, 0].set_title("Общее распределение всех значений матрицы")
    axes[1, 0].set_xlabel("Значения")
    axes[1, 0].set_ylabel("Плотность вероятности")
    axes[1, 0].grid(True, alpha=0.3)

    x = np.linspace(mean - 4*std, mean + 4*std, 100)
    y = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean)/std)**2)
    axes[1, 1].plot(x, y, 'r-', linewidth=2, label='Теоретическое N(μ,σ²)')
    axes[1, 1].hist(matrix.flatten(), bins=20, alpha=0.5, density=True, 
                   color='blue', label='Данные матрицы')
    axes[1, 1].set_title("Сравнение с теоретическим распределением")
    axes[1, 1].set_xlabel("Значения")
    axes[1, 1].set_ylabel("Плотность вероятности")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return matrix, column_means, column_variances, row_means, row_variances


def run_demonstration():
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ: СТАНДАРТНОЕ НОРМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ N(0, 1)")
    print("="*80)
    analyze_normal_matrix(3, 4)
    
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ: НОРМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ N(2, 0.5)")
    print("="*80)
    analyze_normal_matrix(3, 4, mean=2, std=0.5)
    
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ: НОРМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ N(-1, 2)")
    print("="*80)
    analyze_normal_matrix(4, 3, mean=-1, std=2)


if name == "__main__":
    run_demonstration()

#Тесты

import pytest
import numpy as np
import sys
import os

# Добавляем путь к директории с основной программой
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import analyze_normal_matrix


class TestAnalyzeNormalMatrix:
    """Тесты для функции analyze_normal_matrix"""
    
    def test_basic_functionality(self):
        """Тест базовой функциональности с стандартными параметрами"""
        m, n = 3, 4
        matrix, col_means, col_vars, row_means, row_vars = analyze_normal_matrix(m, n)
        
        # Проверка размеров возвращаемых массивов
        assert matrix.shape == (m, n)
        assert col_means.shape == (n,)assert col_vars.shape == (n,)
        assert row_means.shape == (m,)
        assert row_vars.shape == (m,)
        
        # Проверка, что матрица содержит числа
        assert np.all(np.isfinite(matrix))
    
    def test_custom_parameters(self):
        """Тест с пользовательскими параметрами распределения"""
        m, n, mean, std = 2, 3, 5.0, 2.0
        matrix, col_means, col_vars, row_means, row_vars = analyze_normal_matrix(m, n, mean, std)
        
        # Проверка размеров
        assert matrix.shape == (m, n)
        
        # Проверка, что средние значения близки к ожидаемым (из-за случайности допускаем некоторую погрешность)
        overall_mean = np.mean(matrix)
        assert abs(overall_mean - mean) < 2.0  # Допустимая погрешность
    
    def test_invalid_inputs(self):
        """Тест обработки невалидных входных данных"""
        # Отрицательные размеры матрицы
        with pytest.raises(ValueError):
            analyze_normal_matrix(-1, 5)
        
        with pytest.raises(ValueError):
            analyze_normal_matrix(3, -2)
        
        # Нулевые размеры матрицы
        with pytest.raises(ValueError):
            analyze_normal_matrix(0, 5)
        
        # Отрицательное стандартное отклонение
        with pytest.raises(ValueError):
            analyze_normal_matrix(3, 4, std=-1.0)
    
    def test_statistics_calculation(self):
        """Тест корректности вычисления статистик"""
        # Создаем детерминированную матрицу для тестирования
        test_matrix = np.array([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]])
        
        # Подменяем случайную генерацию на детерминированную матрицу
        original_normal = np.random.normal
        np.random.normal = lambda loc, scale, size: test_matrix
        
        try:
            matrix, col_means, col_vars, row_means, row_vars = analyze_normal_matrix(2, 3)
            
            # Проверка вычисления средних по столбцам
            expected_col_means = np.array([2.5, 3.5, 4.5])
            assert np.allclose(col_means, expected_col_means)
            
            # Проверка вычисления средних по строкам
            expected_row_means = np.array([2.0, 5.0])
            assert np.allclose(row_means, expected_row_means)
            
        finally:
            # Восстанавливаем оригинальную функцию
            np.random.normal = original_normal
    
    def test_output_types(self):
        """Тест типов возвращаемых значений"""
        matrix, col_means, col_vars, row_means, row_vars = analyze_normal_matrix(2, 2)
        
        # Проверка типов данных
        assert isinstance(matrix, np.ndarray)
        assert isinstance(col_means, np.ndarray)
        assert isinstance(col_vars, np.ndarray)
        assert isinstance(row_means, np.ndarray)
        assert isinstance(row_vars, np.ndarray)
        
        # Проверка типа данных элементов
        assert matrix.dtype in [np.float32, np.float64]
        assert col_means.dtype in [np.float32, np.float64]


if name == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v"])