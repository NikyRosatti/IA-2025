
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt

N = int(input())

 #individuo con menos amenazas es mejor
creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) #creo una clase FitnessMax que hereda de la clase base.fitness
creator.create("Individual", list, fitness= creator.FitnessMin) #creo una clase individuo que tiene como atributo la fitness

toolbox = base.Toolbox() #creo una caja de herramientas
toolbox.register("atr_int", random.randint, 0, N-1) #un atributo booleano de 0 a N-1
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.atr_int, N) #creo un individuo que es una lista de N atr_int
toolbox.register("population", tools.initRepeat, list, toolbox.individual) #creo una poblacion que es una lista de individuos

#Funcion de evaluacion, cuenta unos de cada individuo
def eval_maxones(individuo):
    # las reinas no pueden coincidir ni verticalmente, horizontalmente, diagonal
    # de acuerdo a la cantidad de amenazas
    cant_amenazas = 0
    #diagonal ej array[i] = loquesea y array[i+1] = loquesea -1 o array[i+1] = loquesea +1
    for i in range(N):
        for j in range(i+1, N):
            if individuo[i] == individuo[j]:
                cant_amenazas+=1
            if abs(individuo[i] - individuo[j]) == abs(i - j):
                cant_amenazas+=1
    return cant_amenazas,
toolbox.register("evaluate", eval_maxones)

#operadores geneticos
toolbox.register("mate", tools.cxTwoPoint) # Combinacion de un punto, pero corrige duplicados(preservando el orden relativo de los padres)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=N-1, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3) #funcion que selecciona 3 individuos
#Los individuos con menos conflictos tendrán más chances de reproducirse.

def main():
    random.seed(42)
    pop = toolbox.population(n=100)

    # Registro de estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: sum(f[0] for f in fits) / len(fits))
    stats.register("max", lambda fits: max(f[0] for f in fits))
    stats.register("min", lambda fits: min(f[0] for f in fits))

    # Hall of Fame (opcional, mejores individuos)
    hof = tools.HallOfFame(1)

    # Ejecutar algoritmo y recolectar log
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=50, lambda_=100,cxpb=0.5, mutpb=0.2, ngen=50,
                                   stats=stats, halloffame=hof, verbose=True)

    print("Mejor individuo es:", hof[0])
    print("Fitness del mejor:", hof[0].fitness.values)

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

if __name__ == "__main__":
    main()