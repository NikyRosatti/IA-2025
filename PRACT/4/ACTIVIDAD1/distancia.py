import numpy as np

def distancia_euclidiana(p1, p2):
    """
    Calcula la distancia euclidiana entre dos puntos p1 y p2.
    p1 y p2 deben ser arrays o listas de la misma dimensión.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.sqrt(np.sum((p1 - p2)**2))

def distancia_manhattan(p1, p2):
    """
    Calcula la distancia Manhattan entre dos puntos p1 y p2.
    p1 y p2 deben ser arrays o listas de la misma dimensión.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.sum(np.abs(p1 - p2))

print("Distancia manhattan", distancia_manhattan((3,2), (6,2)))