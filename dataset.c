/**
 * dataset.c
 * Реализация функций парсинга CSV-файлов и строкового преобразования локали
 */

#include "dataset.h"
#include <stdlib.h>
#include <string.h>

/**
 * Внутренняя функция форматирования вещественного числа.
 */
static void format_to_excel_style(FILE* fp, double value) {
    char buffer[64];
    sprintf(buffer, "%.4f", value);
    for (int i = 0; buffer[i]; i++) {
        if (buffer[i] == '.') {
            buffer[i] = ',';
        }
    }
    fprintf(fp, "%s", buffer);
}

int load_csv_dataset(const char* filename, double* data, int* true_labels, int n_points, int n_features) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Критическая ошибка: не удалось открыть файл данных '%s'\n", filename);
        return 0;
    }

    char line[256];
    // Пропускаем заголовок CSV файла (строка вида x;y;true_label)
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return 0;
    }

    int i = 0;
    // Построчно парсим файл, используя разделитель ';'
    while (fgets(line, sizeof(line), fp) && i < n_points) {
        char* token = strtok(line, ";");
        if (!token) continue;
        
        // Считываем  X (меняем обратно запятую на точку перед atof, если питон записал с запятой)
        for(int t=0; token[t]; t++) if(token[t] == ',') token[t] = '.';
        data[i * n_features + 0] = atof(token);

        // Считываем  Y
        token = strtok(NULL, ";");
        if (!token) continue;
        for(int t=0; token[t]; t++) if(token[t] == ',') token[t] = '.';
        data[i * n_features + 1] = atof(token);

        // Считываем истинную метку true_label
        token = strtok(NULL, ";");
        if (!token) continue;
        true_labels[i] = atoi(token);

        i++;
    }

    fclose(fp);
    return (i == n_points);
}

void save_optimal_clusters(const char* filename, const double* data, const int* true_labels, const int* predicted_labels, int n_points) {
    FILE* fp = fopen(filename, "w");
    if (!fp) return;

    // Пишем заголовок таблицы
    fprintf(fp, "x;y;true_label;predicted_label\n");

    for (int i = 0; i < n_points; i++) {
        format_to_excel_style(fp, data[i * 2 + 0]);
        fprintf(fp, ";");
        format_to_excel_style(fp, data[i * 2 + 1]);
        fprintf(fp, ";%d;%d\n", true_labels[i], predicted_labels[i]);
    }

    fclose(fp);
}

void save_metrics_history(const char* filename, const int* k_values, const double* wss_values, const double* sil_values, int count) {
    FILE* fp = fopen(filename, "w");
    if (!fp) return;

    fprintf(fp, "k;WSS_Inertia;Silhouette_Score\n");
    for (int i = 0; i < count; i++) {
        fprintf(fp, "%d;", k_values[i]);
        format_to_excel_style(fp, wss_values[i]);
        fprintf(fp, ";");
        format_to_excel_style(fp, sil_values[i]);
        fprintf(fp, "\n");
    }
    fclose(fp);
}