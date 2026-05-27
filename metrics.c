/**
 * @file metrics.c
 * @brief Реализация математического аппарата Силуэтного анализа.
 * * Силуэтный анализ требует вычисления попарных расстояний между точками внутри
 * одного кластера, а также расстояний до точек из ближайшего альтернативного кластера.
 * Данный процесс имеет вычислительную сложность O(N^2), поэтому оптимизирован внутренними циклами.
 */

#include "metrics.h"
#include <stdlib.h>
#include <math.h>

/**
 * @brief Внутренняя функция для расчета классического евклидова расстояния (с корнем).
 * * Для силуэтного анализа нам нужны точные метрические расстояния (а не их квадраты),
 * так как они усредняются линейно согласно классической формуле Силуэта.
 */
static double get_euclidean_dist(const double* p1, const double* p2, int n_features) {
    double sum = 0.0;
    for (int i = 0; i < n_features; i++) {
        double diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

double calculate_silhouette(const double* data, const int* labels, int n_points, int n_features, int k) {
    double total_silhouette = 0.0;
    
    // Выделяем и обнуляем массив под размеры кластеров
    int* cluster_sizes = (int*)calloc(k, sizeof(int));
    for (int i = 0; i < n_points; i++) {
        cluster_sizes[labels[i]]++;
    }

    // Проходим по каждой точке датасета для расчета её персонального силуэта
    for (int i = 0; i < n_points; i++) {
        int my_cluster = labels[i];
        
        // Если в кластере всего одна точка, её силуэт по определению равен 0
        if (cluster_sizes[my_cluster] <= 1) {
            continue;
        }

        // Динамический массив для накопления сумм расстояний от точки i до всех кластеров c
        double* avg_dists = (double*)calloc(k, sizeof(double));

        // Вычисляем расстояния от текущей точки до абсолютно всех остальных точек данных
        for (int j = 0; j < n_points; j++) {
            if (i == j) continue;
            avg_dists[labels[j]] += get_euclidean_dist(&data[i * n_features], &data[j * n_features], n_features);
        }

        // Параметр a: среднее расстояние до точек своего собственного кластера
        double a = avg_dists[my_cluster] / (cluster_sizes[my_cluster] - 1);

        // Параметр b: минимальное среднее расстояние до точек любого другого кластера
        double b = -1.0;
        for (int c = 0; c < k; c++) {
            if (c == my_cluster || cluster_sizes[c] == 0) continue;
            
            double avg_to_cluster_c = avg_dists[c] / cluster_sizes[c];
            if (b < 0 || avg_to_cluster_c < b) {
                b = avg_to_cluster_c;
            }
        }

        // Расчет силуэта текущей точки по стандартной формуле: s = (b - a) / max(a, b)
        double s = 0.0;
        if (a < b) {
            s = (b - a) / b;
        } else if (a > b) {
            s = (b - a) / a;
        }

        total_silhouette += s;

        // Освобождаем память временного массива расстояний для текущей точки
        free(avg_dists);
    }

    free(cluster_sizes);

    // Возвращаем среднее арифметическое значение силуэта по всем точкам выборки
    return total_silhouette / n_points;
}