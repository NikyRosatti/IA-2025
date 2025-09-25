import numpy as np
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt

N = 10




# ===========================
# Definiciones de clases
# ===========================
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(N), N)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# ===========================
# Operadores genéticos
# ===========================
toolbox.register("mate", tools.cxOrdered)

# Mutación por swap (intercambia dos posiciones)
def mutSwap(ind):
    a, b = random.sample(range(len(ind)), 2)
    ind[a], ind[b] = ind[b], ind[a]
    return ind,

toolbox.register("mutate", mutSwap)

toolbox.register("select", tools.selTournament, tournsize=3)

# ===========================
# Algoritmo principal
# ===========================
def main():
    random.seed(42)
    np.random.seed(42)
        
    
    matrix = [[0 if i == j else random.randint(10, 100) for j in range(N)] for i in range(N)]

    # Hacemos simétrica la matriz
    for i in range(N):
        for j in range(i+1, N):
            matrix[j][i] = matrix[i][j]

    # ===========================
    # Función de evaluación
    # ===========================
    def evalRoute(individual):
        total = 0
        for i in range(len(individual)):
            j = (i + 1) % len(individual)  # siguiente ciudad (ciclo cerrado)
            total += matrix[individual[i]][individual[j]]
        return total,
    toolbox.register("evaluate", evalRoute)

    pop = toolbox.population(n=1050)

    # Registro de estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: sum(f[0] for f in fits) / len(fits))
    stats.register("max", lambda fits: max(f[0] for f in fits))
    stats.register("min", lambda fits: min(f[0] for f in fits))

    # Hall of Fame (opcional, mejores individuos)
    hof = tools.HallOfFame(1)

    # Ejecutar algoritmo y recolectar log
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=20,
                                   stats=stats, halloffame=hof, verbose=True)

    # Extraer datos para graficar
    gen = log.select("gen")
    fit_max = log.select("max")
    fit_avg = log.select("avg")
    fit_min = log.select("min")

    # Graficar evolución del fitness
    plt.plot(gen, fit_max, label="Max")
    plt.plot(gen, fit_avg, label="Avg")
    plt.plot(gen, fit_min, label="Min")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.title("Evolución del Fitness (Max-Ones)")
    plt.legend()
    plt.grid()
    plt.show()

    
    print("Mejor individuo es:", hof[0])
    print("Fitness del mejor:", hof[0].fitness.values)

if __name__ == "__main__":
    main()
