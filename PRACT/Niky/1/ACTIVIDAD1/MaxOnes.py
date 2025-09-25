# ACTIVIDAD 1
import random

Generaciones = 10
Size_Poblacion = 30
long_individuo = int(input("Tamaño del individuo: "))

def main():
    poblacion = crear_poblacion()

    for gen in range(Generaciones):
        # Ordenar población por fitness
        poblacion.sort(key=fitness, reverse=True)
        print(f"Gen {gen}: Mejor = {poblacion[0]} Fitness = {fitness(poblacion[0])}")

        # Nueva generación
        next_gen = []
        for _ in range(Size_Poblacion // 2):
            # Selección de 2 padres por ruleta
            p1 = ruleta_selection(poblacion)
            p2 = ruleta_selection(poblacion)

            # Crossover
            h1, h2 = crossover(p1, p2)

            # Mutación
            next_gen.append(mutacion(h1))
            next_gen.append(mutacion(h2))

        poblacion = next_gen  # reemplazar con la nueva población

    # Retornar el mejor individuo encontrado
    return max(poblacion, key=fitness)


def crear_poblacion():
    return [crear_individuo() for _ in range(Size_Poblacion)]

def crear_individuo():
    return [random.randint(0, 1) for _ in range(long_individuo)]


# Selección por ruleta (FPS)
def ruleta_selection(poblacion):
    fitnesses = [fitness(ind) for ind in poblacion]
    total = sum(fitnesses)
    r = random.uniform(0, total)
    acumulado = 0
    for individuo, f in zip(poblacion, fitnesses):
        acumulado += f
        if acumulado >= r:
            return individuo[:]


### FITNESS ###
def fitness(individuo):
    return sum(individuo)  # cantidad de unos


### Combinación de un punto
def crossover(p1, p2):
    if long_individuo < 2:
        return p1[:], p2[:]
    k = random.randint(1, long_individuo - 1)
    c1 = p1[:k] + p2[k:]
    c2 = p2[:k] + p1[k:]
    return c1, c2


# Mutación por intercambio
def mutacion(individuo):
    K1 = random.randint(0, long_individuo - 1)
    K2 = random.randint(0, long_individuo - 1)
    while K1 == K2:
        K2 = random.randint(0, long_individuo - 1)
    individuo[K1], individuo[K2] = individuo[K2], individuo[K1]
    return individuo


if __name__ == "__main__":
    best = main()
    print(f"Mejor individuo encontrado: {best}, con Fitness = {fitness(best)}")
