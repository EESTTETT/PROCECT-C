/**
 * main.c
 * Ядро кластеризации.
 *  Этот файл является центральной точкой входа программы на C. Он координирует
 * загрузку данных от модуля Python, запускает исследовательский цикл подбора k 
 * от 2 до 10, анализирует значения силуэтов в режиме реального времени, 
 * автоматически принимает решение о выборе оптимальной структуры и выводит метрики.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "kmeans.h"
#include "metrics.h"
#include "dataset.h"

int main() {
    // Настройка системного генератора случайных чисел на основе текущего времени
    srand((unsigned int)time(NULL));

    // Параметры конфигурации эксперимента (должны строго совпадать с параметрами в Python)
    const int n_points = 2000;
    const int n_features = 2;
    const int max_iters = 150;

    // Выделение динамической памяти под структуры данных приложения
    double* data = (double*)malloc(n_points * n_features * sizeof(double));
    int* true_labels = (int*)malloc(n_points * sizeof(int));
    
    printf("[C] Запуск вычислительного ядра К-means...\n");

    // Загружаем данные из файла, созданного методом make_blobs в Python
    if (!load_csv_dataset("synthetic_input.csv", data, true_labels, n_points, n_features)) {
        printf("[C] Ошибка: Не удалось подготовить данные. Проверьте 'synthetic_input.csv'\n");
        free(data);
        free(true_labels);
        return -1;
    }
    printf("[C] Датасет на 2000 точек успешно импортирован\n");

    // Массивы для логирования процесса автоматического подбора оптимального числа кластеров K
    int k_history[9];
    double wss_history[9];
    double sil_history[9];
    int idx = 0;

    int best_k = 2;
    double max_silhouette = -2.0; // Ниже нижней границы силуэта для старта поиска максимума

    printf("[C] Анализ внутренней структуры данных (перебор К от 2 до 10)...\n");

    // ИССЛЕДОВАТЕЛЬСКИЙ ЦИКЛ: Метод локтя и Силуэтный анализ
    for (int test_k = 2; test_k <= 10; test_k++) {
        int* temp_labels = (int*)malloc(n_points * sizeof(int));
        double* temp_centroids = (double*)malloc(test_k * n_features * sizeof(double));

        // Вычисляем K-means и забираем Инерцию (WSS) для Метода локтя
        double wss = run_kmeans(data, n_points, n_features, test_k, max_iters, temp_labels, temp_centroids);
        
        // Рассчитываем силуэт для текущего числа кластеров
        double sil = calculate_silhouette(data, temp_labels, n_points, n_features, test_k);

        // Сохраняем результаты шага в массивы истории
        k_history[idx] = test_k;
        wss_history[idx] = wss;
        sil_history[idx] = sil;
        idx++;

        printf("   Тест k=%d | Инерция (метод локотя): %7.2f | Коэффициент силуэта: %.4f\n", test_k, wss, sil);

        // Логика автоматического выбора оптимального K: ищем глобальный максимум Силуэта
        if (sil > max_silhouette) {
            max_silhouette = sil;
            best_k = test_k;
        }

        free(temp_labels);
        free(temp_centroids);
    }

    // Сохраняем историю подбора K в файл для построения графиков локтя в Excel
    save_metrics_history("metrics_output.csv", k_history, wss_history, sil_history, 9);
    printf("[C] Метрики для графиков сохранены в 'metrics_output.csv'\n");

    printf("[C] Оптимальное число кластеров k = %d (наибольший силуэт = %.4f)\n", best_k, max_silhouette);

    // ФИНАЛЬНЫЙ СТАБИЛЬНЫЙ ПРОГОН: Вычисляем разметку на оптимальном числе кластеров
    int* final_predicted_labels = (int*)malloc(n_points * sizeof(int));
    double* final_centroids = (double*)malloc(best_k * n_features * sizeof(double));
    
    run_kmeans(data, n_points, n_features, best_k, max_iters, final_predicted_labels, final_centroids);

    // Печатаем итоговые центроиды
    printf("Координаты всех центроидов для k=%d:\n", best_k);
    for (int j = 0; j < best_k; j++) {
        printf("   Центроид %d: X = %.3f, Y = %.3f\n", j, final_centroids[j * 2 + 0], final_centroids[j * 2 + 1]);
    }

    // Экспортируем финальную разметку для визуализации разноцветного облака в Excel Web
    save_optimal_clusters("clusters_optimal.csv", data, true_labels, final_predicted_labels, n_points);
    printf("Результат работы программы записаны в файле 'clusters_optimal.csv'\n");

    // Полная зачистка всей кучи выделенной памяти перед завершением процесса
    free(data);
    free(true_labels);
    free(final_predicted_labels);
    free(final_centroids);

    return 0;
}