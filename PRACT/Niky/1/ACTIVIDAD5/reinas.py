# ACTIVIDAD 1
import random

Generaciones = 10
Size_Poblacion = 30
N = int(input("NRO REINAS: "))
reinas = [0] * N

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
            print(f"HOLA SOY P1: {p1}")
            print(f"HOLA SOY P2: {p2}")

            # Crossover
            h1, h2 = crossover(p1, p2)

            # Mutación
            next_gen.append(mutacion(h1))
            next_gen.append(mutacion(h2))

        poblacion = next_gen  # reemplazar con la nueva población

    # Retornar el mejor individuo encontrado
    return min(poblacion, key=fitness)


def crear_poblacion():
    return [crear_individuo() for _ in range(Size_Poblacion)]

def crear_individuo():
    return [random.randint(1, N) for _ in range(N)]


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
    # las reinas no pueden coincidir ni verticalmente, horizontalmente, diagonal
    # de acuerdo a la cantidad de amenazas
    cant_amenazas = 0
    value = 0
    #horizontal ej array[i] = 2 y array[i+1]=2
    #diagonal ej array[i] = loquesea y array[i+1] = loquesea -1 o array[i+1] = loquesea +1
    for i in range(N):
        for j in range(i, N):
            if (i!=j):
                if individuo[i] == individuo[j]:
                    cant_amenazas+=1
                    value += (cant_amenazas + 100)
                if individuo[i] == (individuo[j] - 1) or individuo[i] == (individuo[j] + 1):
                    cant_amenazas+=1
                    value += (cant_amenazas + 100)
                if individuo[i] != individuo[j]:
                    value += (cant_amenazas - 100)
                if individuo[i] != (individuo[j] - 1) or individuo[i] != (individuo[j] + 1):
                    cant_amenazas+=1
                    value += (cant_amenazas - 100)
    return value


### Combinación de un punto
def crossover(p1, p2):
    if N < 2:
        return p1[:], p2[:]
    k = random.randint(1, N - 1)
    c1 = p1[:k] + p2[k:]
    c2 = p2[:k] + p1[k:]
    return c1, c2


# Mutación por gen
def mutacion(individual):
    for i in range(N):
        if random.random() < 0.05:
            individual[i] = random.randint(1,N) # Flip bit
    return individual


if __name__ == "__main__":
    best = main()
    print(f"Mejor individuo encontrado: {best}, con Fitness = {fitness(best)}")
