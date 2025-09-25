import random

import numpy as np


def nearest_neighbor_all_starts(matrix):
    """
    Ejecuta el algoritmo del vecino más cercano desde todas las ciudades.
    Devuelve la mejor ruta y su costo.
    """
    n = len(matrix)

    def nearest_neighbor(matrix, start):
        visited = [False] * n
        route = [start]
        visited[start] = True
        current = start

        for _ in range(n - 1):
            next_city = None
            min_dist = float("inf")
            for j in range(n):
                if not visited[j] and matrix[current][j] < min_dist:
                    min_dist = matrix[current][j]
                    next_city = j
            route.append(next_city)
            visited[next_city] = True
            current = next_city

        # calcular costo total
        total_cost = 0
        for i in range(n):
            j = (i + 1) % n
            total_cost += matrix[route[i]][route[j]]
        return route, total_cost

    best_route = None
    best_cost = float("inf")

    for start in range(n):
        route, cost = nearest_neighbor(matrix, start)
        if cost < best_cost:
            best_cost = cost
            best_route = route

    return best_route, best_cost

N = 10
random.seed(42)
np.random.seed(42)

matrix = [[0 if i == j else random.randint(10, 100) for j in range(N)] for i in range(N)]

    # Hacemos simétrica la matriz
for i in range(N):
    for j in range(i+1, N):
        matrix[j][i] = matrix[i][j]



list = [2, 0, 8, 4, 6, 9, 7, 1, 5, 3]
def evalRoute(individual):
        total = 0
        for i in range(len(individual)):
            j = (i + 1) % len(individual)  # siguiente ciudad (ciclo cerrado)
            total += matrix[individual[i]][individual[j]]
        return total,
best_route, best_cost = nearest_neighbor_all_starts(matrix)
print("Mejor ruta greedy:", best_route)
print("Costo greedy:", best_cost)
print("Costo eval:", evalRoute(list))