# Главная цель, которая выполнится при вызове команды `make`
all:
	gcc main.c kmeans.c metrics.c dataset.c -o kmeans_analyzer -lm
	./kmeans_analyzer