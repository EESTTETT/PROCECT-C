/**
 * dataset.h
 * Модуль ввода-вывода (I/O) и сохранения структурированных CSV файлов.
 * * Отвечает за интеграцию между Python (генератор make_blobs) и C, а также за адаптацию
 * выходных форматов чисел под языковую специфику парсеров Microsoft Excel Web.
 */

#ifndef DATASET_H
#define DATASET_H

#include <stdio.h>

/**
 * @brief Функция считывания данных из CSV файла, сгенерированного Python скриптом.
 * * @param filename Путь к файлу входных данных.
 * @param data Указатель на выделенный массив, куда запишутся координаты.
 * @param true_labels Указатель на массив для сохранения эталонных меток от make_blobs.
 * @param n_points Ожидаемое количество строк данных для считывания.
 * @param n_features Количество колонок признаков (например, 2).
 * @return int Возвращает 1 в случае успеха, 0 при ошибке открытия или структуры файла.
 */
int load_csv_dataset(const char* filename, double* data, int* true_labels, int n_points, int n_features);

/**
 * @brief Сохранение результатов итоговой кластеризации с адаптацией разделителей под Excel.
 * * @param filename Имя выходного CSV файла.
 * @param data Координаты точек.
 * @param true_labels Реальные эталонные метки.
 * @param predicted_labels Метки, предсказанные нашим K-means.
 * @param n_points Количество точек.
 */
void save_optimal_clusters(const char* filename, const double* data, const int* true_labels, const int* predicted_labels, int n_points);

/**
 * @brief Вспомогательная функция сохранения результатов подбора K (Локоть и Силуэт).
 */
void save_metrics_history(const char* filename, const int* k_values, const double* wss_values, const double* sil_values, int count);

#endif // DATASET_H