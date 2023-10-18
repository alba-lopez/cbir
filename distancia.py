import numpy as np

def dist_en_uso(a: np.array, b: np.array) -> float:
    d = euclidea(a, b) #cambiar: función de distancia a usar
    return d

def euclidea(a, b):
    d = np.linalg.norm(a - b)
    return d

