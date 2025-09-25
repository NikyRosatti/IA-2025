from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt

# ===========================
# Definiciones de clases
# ===========================
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("atr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.atr_bool, 20)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ===========================
# Función de evaluación
# ===========================
def eval_maxones(individual):
    return sum(individual),
toolbox.register("evaluate", eval_maxones)

# ===========================
# Operadores genéticos
# ===========================
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# ===========================
# Algoritmo principal
# ===========================
def main():
    random.seed(42)
    pop = toolbox.population(n=50)

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
