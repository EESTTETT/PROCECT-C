/**
 * kmeans.c
 * Реализация алгоритма K-means и метода умного старта K-means++.
 * * Файл содержит математическую логику распределения точек по кластерам,
 * вычисления расстояний в многомерном пространстве и динамического пересчета
 * центров масс групп точек на каждой итерации.
 */

#include "kmeans.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/**
 * Вспомогательная функция вычисления квадрата Евклидова расстояния.
 * * Мы используем квадрат расстояния вместо обычного расстояния везде, где это возможно,
 * чтобы избежать ресурсоемкой операции извлечения квадратного корня (sqrt), 
 * что значительно ускоряет вычисления на больших объемах данных.
 */
static double get_dist_sq(const double* p1, const double* p2, int n_features) {
    double sum = 0.0;
    for (int i = 0; i < n_features; i++) {
        double diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sum;
}

void init_centroids_kmeans_pp(const double* data, int n_points, int n_features, int k, double* centroids) {
    // Шаг 1: Первый центроид выбирается абсолютно случайно из имеющихся точек данных.
    int first_idx = rand() % n_points;
    for (int f = 0; f < n_features; f++) {
        centroids[0 * n_features + f] = data[first_idx * n_features + f];
    }

    // Выделяем память под массив квадратов расстояний до ближайшего центроида для каждой точки
    double* d_sq = (double*)malloc(n_points * sizeof(double));

    // Шаг 2: Итерационно выбираем остальные k - 1 центроидов
    for (int c = 1; c < k; c++) {
        double sum_sq = 0.0;

        // Для каждой точки ищем минимальное расстояние до уже существующих центроидов
        for (int i = 0; i < n_points; i++) {
            double min_dist = -1.0;
            for (int j = 0; j < c; j++) {
                double dist = get_dist_sq(&data[i * n_features], &centroids[j * n_features], n_features);
                if (min_dist < 0 || dist < min_dist) {
                    min_dist = dist;
                }
            }
            d_sq[i] = min_dist;
            sum_sq += min_dist; // Накапливаем сумму для вероятностного распределения
        }

        // Реализация метода "рулетки" (Roulette Wheel Selection):
        // Выбираем случайную точку с вероятностью, пропорциональной квадрату её расстояния.
        double r = ((double)rand() / RAND_MAX) * sum_sq;
        double current_sum = 0.0;
        int next_centroid_idx = n_points - 1;

        for (int i = 0; i < n_points; i++) {
            current_sum += d_sq[i];
            if (current_sum >= r) {
                next_centroid_idx = i;
                break;
            }
        }

        // Копируем координаты выбранной точки в структуру центроидов
        for (int f = 0; f < n_features; f++) {
            centroids[c * n_features + f] = data[next_centroid_idx * n_features + f];
        }
    }

    // Освобождаем временную память, чтобы избежать утечек
    free(d_sq);
}

double run_kmeans(const double* data, int n_points, int n_features, int k, int max_iters, int* labels, double* centroids) {
    // Шаг 1: Инициализируем центроиды с помощью K-means++
    init_centroids_kmeans_pp(data, n_points, n_features, k, centroids);

    // Выделяем память для подсчета количества точек, зачисленных в каждый из кластеров
    int* cluster_sizes = (int*)malloc(k * sizeof(int));
    int changed = 1; // Флаг отслеживания изменений на текущей итерации
    int iter = 0;

    // Изначально помечаем все точки как неприсвоенные ни к одному кластеру
    for (int i = 0; i < n_points; i++) labels[i] = -1;

    // Основной цикл оптимизации K-means
    while (changed && iter < max_iters) {
        changed = 0;

        // ЭТАП Е-шага (Expectation): Привязка каждой точки к ближайшему центроиду
        for (int i = 0; i < n_points; i++) {
            double min_dist = -1.0;
            int best_cluster = 0;

            for (int j = 0; j < k; j++) {
                double dist = get_dist_sq(&data[i * n_features], &centroids[j * n_features], n_features);
                if (min_dist < 0 || dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }

            // Если точка поменяла свой кластер, фиксируем изменение структуры
            if (labels[i] != best_cluster) {
                labels[i] = best_cluster;
                changed = 1;
            }
        }

        // Если ни одна точка не изменила свой кластер, значит алгоритм сошелся досрочно
        if (!changed) break;

        // ЭТАП М-шага (Maximization): Обновление координат центроидов (пересчет центра масс)
        // Обнуляем старые координаты и счетчики размеров
        for (int j = 0; j < k; j++) {
            cluster_sizes[j] = 0;
            for (int f = 0; f < n_features; f++) {
                centroids[j * n_features + f] = 0.0;
            }
        }

        // Суммируем координаты всех точек, попавших в соответствующий кластер
        for (int i = 0; i < n_points; i++) {
            int cluster = labels[i];
            cluster_sizes[cluster]++;
            for (int f = 0; f < n_features; f++) {
                centroids[cluster * n_features + f] += data[i * n_features + f];
            }
        }

        // Делим накопленную сумму координат на количество точек в кластере (вычисляем среднее)
        for (int j = 0; j < k; j++) {
            if (cluster_sizes[j] > 0) {
                for (int f = 0; f < n_features; f++) {
                    centroids[j * n_features + f] /= cluster_sizes[j];
                }
            }
        }
        iter++;
    }

    // Вычисляем финальную внутреннюю инерцию (WSS — Within Cluster Sum of Squares)
    double wss = 0.0;
    for (int i = 0; i < n_points; i++) {
        wss += get_dist_sq(&data[i * n_features], &centroids[labels[i] * n_features], n_features);
    }

    // Чистим динамическую память за собой
    free(cluster_sizes);
    
    return wss;
}