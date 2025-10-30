import numpy as np
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt

N = 20

TAM_POBLACION = 250
NUM_GENERACIONES = 1000
PROB_CRUCE = 0.7
PROB_MUTACION = 0.2
NUM_SOLUCIONES_DESEADAS = 100

LIMITE_SUP_DISTANCIA = 300
LIMITE_SUP_TIEMPO = 1000


# Definiciones de clases
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(N), N)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Operadores genéticos
toolbox.register("mate", tools.cxOrdered)

# Mutación
def mutSwap(ind):
    '''
    Mutación por swap: intercambia dos ciudades en la ruta
    Retorna una tupla con el individuo mutado
    '''
    a, b = random.sample(range(len(ind)), 2)
    ind[a], ind[b] = ind[b], ind[a]
    return ind,

toolbox.register("mutate", mutSwap)

# Selector de NSGA-II
toolbox.register("select", tools.selNSGA2)


def main():
    '''
    Función principal para ejecutar el algoritmo genético en el TSP multi-objetivo
    '''
    # Semillas para la creacion de matrices
    random.seed(42)
    np.random.seed(42)
    
    # 1. Matriz de Distancia random
    matrix_dist = [[0 if i == j else random.randint(50, LIMITE_SUP_DISTANCIA) for j in range(N)] for i in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            matrix_dist[j][i] = matrix_dist[i][j]

    # 2. Matriz de Tiempo random que no se correlacione con la distancia
    matrix_tiempo = [[0 if i == j else random.randint(10, LIMITE_SUP_TIEMPO) for j in range(N)] for i in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            matrix_tiempo[j][i] = matrix_tiempo[i][j]

    def evalMOTSP(individual):
        '''
        Funcion de evaluación para el TSP multi-objetivo
        Retorna dos objetivos: distancia total y tiempo total
        '''
        distancia_total = 0
        tiempo_total = 0
        for i in range(len(individual)):
            j = (i + 1) % len(individual)  # siguiente ciudad
            
            ciudad_a = individual[i]
            ciudad_b = individual[j]
            
            distancia_total += matrix_dist[ciudad_a][ciudad_b]
            tiempo_total += matrix_tiempo[ciudad_a][ciudad_b]
        return distancia_total, tiempo_total

    toolbox.register("evaluate", evalMOTSP)
    
    toolbox.register("mate", tools.cxOrdered)

    pareto_front = tools.ParetoFront()

    pop = toolbox.population(n=TAM_POBLACION)

    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    pareto_front.update(pop)

    for gen in range(1, NUM_GENERACIONES + 1):
        
        # Generamos la descendencia (Offspring)
        # varAnd clona y aplica cruce y mutación
        offspring = algorithms.varAnd(pop, toolbox, PROB_CRUCE, PROB_MUTACION)

        # Evaluamos los individuos de la descendencia que han cambiado
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Selección de la siguiente generación (NSGA-II)
        # Se combinan padres (pop) y descendencia (offspring)
        # y 'selNSGA2' selecciona los 'TAM_POBLACION' mejores
        pop[:] = toolbox.select(pop + offspring, k=TAM_POBLACION)

        # Actualizamos el frente de Pareto
        pareto_front.update(pop)
        
        if gen % 50 == 0:
            print(f"Generación {gen}: {len(pareto_front)} soluciones en el frente.")

    frente_filtrado = tools.selNSGA2(pareto_front, k=NUM_SOLUCIONES_DESEADAS)
    
    puntos_dist = []
    puntos_tiempo = []

    i = 0
    for ind in frente_filtrado:
        fit = ind.fitness.values
        puntos_dist.append(fit[0])
        puntos_tiempo.append(fit[1])
        print(f"Solución {i+1}: Distancia={fit[0]:.2f}, Tiempo={fit[1]:.2f} \n    Ruta: {ind}\n")
        i += 1

    plt.figure(figsize=(10, 6))
    plt.scatter(puntos_dist, puntos_tiempo, facecolors='none', edgecolors='b')
    plt.xlabel("Objetivo 1: Distancia Total")
    plt.ylabel("Objetivo 2: Tiempo Total")
    plt.title("Frente de Pareto para MOTSP (NSGA-II)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
