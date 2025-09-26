import random

#Constantes
POP_SIZE = 20 # Tamaño de la población
GENS = 100 # Número de generaciones
CXPB = 0.8 # Probabilidad de cruce
MUTPB = 0.05 # Probabilidad de mutación
GENOME_LENGTH = 20 # Longitud de cada individuo (número de bits)
# Genotipo


# Poblacion
def createIndivudual():
    return [random.randint(0,1) for _ in range(GENOME_LENGTH)]
def create_population():
    return [createIndivudual() for _ in range(POP_SIZE)]
# Evaluacion (Fitness)
def fitness(individual):
    return sum(individual)
# Evolucion: ...

# Seleccion
def selection(population):
    k = 5
    selected = []
    for _ in range(POP_SIZE):
        aspirantes = random.sample(population,k)
        winner = max(aspirantes, key=fitness)
        selected.append(winner)
    return selected
# Mutacion
def mutate(individual):
    for i in range(GENOME_LENGTH):
        if random.random() < MUTPB:
            individual[i] = 1 - individual[i] # Flip bit
    return individual
# Cruzamiento
def crossover(p1,p2):
    if random.random() < CXPB:
        point = random.randint(1, GENOME_LENGTH - 1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]
    return p1[:], p2[:]

def genetic_algorithm():
    population = create_population()
    for gen in range(GENS):
        # Evaluar y mostrar el mejor
        population.sort(key=fitness, reverse=True)
        print(f"Gen {gen}: Mejor = {population[0]} Fitness = {fitness(population[0])}")
        # Selección
        selected = selection(population)
        # Reproducción
        next_gen = []
    for i in range(0, POP_SIZE, 2):
        offspring1, offspring2 = crossover(selected[i], selected[i+1])
        next_gen.append(mutate(offspring1))
        next_gen.append(mutate(offspring2))
        population = next_gen
    return max(population, key=fitness)

best = genetic_algorithm()
print(f"Mejor individuo encontrado: {best}, Fitness = {fitness(best)}")