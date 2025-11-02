import numpy as np
import matplotlib.pyplot as plt


def analyze_time_series(series, window_size):
    mean = np.mean(series)
    variance = np.var(series)
    std = np.std(series)
    local_maxima = []
    local_minima = []
    
    for i in range(1, len(series) - 1):
        if series[i] > series[i-1] and series[i] > series[i+1]:
            local_maxima.append((i, series[i]))
        if series[i] < series[i-1] and series[i] < series[i+1]:
            local_minima.append((i, series[i]))
    
    moving_average = np.convolve(series, np.ones(window_size)/window_size, mode='valid')
    
    return {
        'mean': mean,
        'variance': variance,
        'std': std,
        'local_maxima': local_maxima,
        'local_minima': local_minima,
        'moving_average': moving_average
    }


np.random.seed(42)
t = np.linspace(0, 4*np.pi, 50)
test_series = np.sin(t) + 0.2 * np.random.normal(size=len(t))
window_size = 3
results = analyze_time_series(test_series, window_size)

print("Результаты:")
print(f"Среднее: {results['mean']:.3f}")
print(f"Дисперсия: {results['variance']:.3f}")
print(f"СКО: {results['std']:.3f}")
print(f"Локальных максимумов: {len(results['local_maxima'])}")
print(f"Локальных минимумов: {len(results['local_minima'])}")

#Тесты
import numpy as np
import pytest


def analyze_time_series(series, window_size):
    mean = np.mean(series)
    variance = np.var(series)
    std = np.std(series)
    local_maxima = []
    local_minima = []
    
    for i in range(1, len(series) - 1):
        if series[i] > series[i-1] and series[i] > series[i+1]:
            local_maxima.append((i, series[i]))
        if series[i] < series[i-1] and series[i] < series[i+1]:
            local_minima.append((i, series[i]))
    
    moving_average = np.convolve(series, np.ones(window_size)/window_size, mode='valid')
    
    return {
        'mean': mean,
        'variance': variance,
        'std': std,
        'local_maxima': local_maxima,
        'local_minima': local_minima,
        'moving_average': moving_average
    }


class TestTimeSeriesAnalysis:
    def test_basic_statistics(self):
        series = np.array([1, 2, 3, 4, 5])
        results = analyze_time_series(series, 3)
        assert results['mean'] == 3.0
        assert results['variance'] == 2.0
        assert results['std'] == pytest.approx(1.4142, 0.001)
    

    def test_local_extrema(self):
        series = np.array([1, 3, 2, 4, 1, 5, 2])
        results = analyze_time_series(series, 3)
        
        assert len(results['local_maxima']) == 3
        assert len(results['local_minima']) == 2
        
        max_values = [val for _, val in results['local_maxima']]
        min_values = [val for _, val in results['local_minima']]
        
        assert 3 in max_values
        assert 4 in max_values
        assert 5 in max_values
        assert 2 in min_values
        assert 1 in min_values
    

    def test_moving_average(self):
        series = np.array([1, 2, 3, 4, 5, 6])
        results = analyze_time_series(series, 3)
        expected_ma = np.array([2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(results['moving_average'], expected_ma)
        assert len(results['moving_average']) == len(series) - 2
    

    def test_constant_series(self):
        series = np.array([5, 5, 5, 5, 5])
        results = analyze_time_series(series, 2)
        assert results['mean'] == 5.0
        assert results['variance'] == 0.0
        assert results['std'] == 0.0
        assert len(results['local_maxima']) == 0
        assert len(results['local_minima']) == 0
    

    def test_single_extremum(self):
        series = np.array([1, 2, 1])
        results = analyze_time_series(series, 2)
        assert len(results['local_maxima']) == 1
        assert len(results['local_minima']) == 0
        assert results['local_maxima'][0][1] == 2

#Запускать через pytest numpy_tasks7.py -v