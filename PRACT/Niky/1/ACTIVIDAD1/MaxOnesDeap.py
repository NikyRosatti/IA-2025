import random
from deap import base, creator, tools

# --- Parámetros ---
Generaciones = 25
Size_Poblacion = 30
long_individuo = int(input("Tamaño del individuo: "))
prob_crossover = 0.8
prob_mutacion = 0.2

# --- Definición del problema ---
# Fitness: queremos maximizar la cantidad de unos
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Gen: aleatorio entre 0 y 1
toolbox.register("attr_bool", random.randint, 0, 1)

# Individuo = lista de long_individuo genes
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=long_individuo)

# Población
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Función de fitness
def evalOneMax(individuo):
    return sum(individuo),   # OJO: debe ser una tupla

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxOnePoint)             # crossover de un punto
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05) # mutación: flip de bits con prob 0.05 cada gen
toolbox.register("select", tools.selRoulette)          # selección por ruleta (FPS)

# --- Algoritmo principal ---
def main():
    random.seed(42)
    poblacion = toolbox.population(n=Size_Poblacion)

    # Evaluar la población inicial
    fitnesses = list(map(toolbox.evaluate, poblacion))
    for ind, fit in zip(poblacion, fitnesses):
        ind.fitness.values = fit

    for gen in range(Generaciones):
        # Selección de padres
        offspring = toolbox.select(poblacion, len(poblacion))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for i in range(0, len(offspring), 2):
            if random.random() < prob_crossover:
                toolbox.mate(offspring[i], offspring[i+1])
                del offspring[i].fitness.values
                del offspring[i+1].fitness.values

        # Mutación
        for mut in offspring:
            if random.random() < prob_mutacion:
                toolbox.mutate(mut)
                del mut.fitness.values

        # Re-evaluar fitness de los hijos
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Reemplazo: la nueva población son los hijos
        poblacion[:] = offspring

        # Mostrar mejor individuo de la generación
        top1 = tools.selBest(poblacion, 1)[0]
        print(f"Gen {gen}: Mejor = {top1}, Fitness = {top1.fitness.values[0]}")

    # Mejor individuo final
    best = tools.selBest(poblacion, 1)[0]
    print("\nMejor individuo encontrado:", best)
    print("Fitness =", best.fitness.values[0])

if __name__ == "__main__":
    main()
