"""
Данный файл связывает Python с кодом на С.
В соответствии с требованиями, вся вычислительная логика перенесена в С, 
а данный скрипт выполняет только генерацию эталонного датасета (по тз пункт 3) и его финальную визуализацию.
"""

import os
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt  # Библиотека для отрисовки графиков

def run_integration_test():
    print("[Py] 1. генерация эталонной структуры через make_blobs для обучения ")
    n_samples = 2000  # количество точек
    n_features = 2
    centers_true = 3 # Генерируем 3 кластера
    
    # Генерируем облака точек с фиксированным random_state для воспроизводимости
    X, y_true = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers_true, 
                           cluster_std=0.8, random_state=42)

    # Сохраняем сгенерированные данные во входной CSV-файл для C программы
    input_filename = "synthetic_input.csv"
    with open(input_filename, "w") as f:
        f.write("x;y;true_label\n")
        for i in range(n_samples):
            f.write(f"{X[i,0]:.4f};{X[i,1]:.4f};{y_true[i]}\n")
    print(f"[Py] Входной файл '{input_filename}' подготовлен")

    # =========================================================================
    # ИНТЕРАКТИВНАЯ ПАУЗА ДЛЯ ВЫПОЛНЕНИЯ В С
    # =========================================================================
    print("\n" + "="*60)
    print(" Запустите C-ядра:")
    print("   gcc main.c kmeans.c metrics.c dataset.c -o kmeans_analyzer -lm")
    print("   ./kmeans_analyzer")
    print("="*60)
    
    input("\nПосле того как программа на C полностью отработает, нажмите ENTER здесь для вывода графика...")

    print("Математическая верификация результатов")
    output_filename = "clusters_optimal.csv"
    if not os.path.exists(output_filename):
        print("[Py] Ошибка: Файл результатов от С-ядра не найден.")
        return

    # Считываем предсказания, сделанные нашей программой на C
    y_pred = []
    with open(output_filename, "r") as f:
        lines = f.readlines()[1:] # пропускаем заголовок
        for line in lines:
            parts = line.strip().split(";")
            if len(parts) >= 4:
                y_pred.append(int(parts[3]))
    
    y_pred = np.array(y_pred)

    # Вычисляем индекс сходства структур (Adjusted Rand Index)
    ari_score = adjusted_rand_score(y_true, y_pred)
    
    print("--- ИТОГОВЫЙ ОТЧЕТ ТЕСТИРОВАНИЯ ---")
    print(f"Индекс сходства структуры: {ari_score:.4f}")
    if ari_score > 0.95:
        print("СТАТУС ТЕСТА: ПРОЙДЕН С ОТЛИЧИЕМ (Идеальное совпадение кластеров с make_blobs!)")
    elif ari_score > 0.80:
        print("СТАТУС ТЕСТА: УСПЕШНО (Допустимые незначительные погрешности на стыках границ)")
    else:
        print("СТАТУС ТЕСТА: ПРОВАЛЕН (Алгоритм застрял или структура нарушена)")
    print("----------------------------------")

    # =========================================================================
    # ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
    # =========================================================================
    print("Идёт визуализация через график")
    
    # Создаем окно для графика фиксированного размера
    plt.figure(figsize=(10, 7))
    
    # Немного уменьшили размер точек (s=30) и прозрачность (alpha=0.7), чтобы 2000 точек не сливались в кашу
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='rainbow', edgecolor='k', s=30, alpha=0.7)
    
    # Добавляем легенду (какой цвет за какой кластер отвечает)
    plt.legend(*scatter.legend_elements(), title="Кластеры (C-Core)", loc="upper right")
    
    # Сетка и подписи осей для академического вида
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("Признак X (Координата 1)", fontsize=12)
    plt.ylabel("Признак Y (Координата 2)", fontsize=12)
    
    # Формируем красивую информативную шапку графика
    detected_k = len(np.unique(y_pred))
    plt.title(f"Результат финальной кластеризации K-Means\n"
              f"Автовыбор числа кластеров: K = {detected_k} | Точность (ARI) = {ari_score:.4f}", 
              fontsize=14, fontweight='bold', pad=15)
    
    # Выводим окно с готовым графиком на экран
    print(".....")
    plt.show()

if __name__ == "__main__":
    run_integration_test()