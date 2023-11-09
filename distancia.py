import numpy as np
import cv2

def dist_en_uso(a: np.array, b: np.array) -> float:
    d = euclidea(a, b) #cambiar: función de distancia a usar
    return d

def euclidea(a, b):
    d = np.linalg.norm(a - b)
    return d

def sift_similarity_score(descriptors1, descriptors2, threshold=0.5):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append(m)

    # Calcula una puntuación de similitud en función de la cantidad de buenas coincidencias
    similarity_score = len(good) / min(len(descriptors1), len(descriptors2))

    return similarity_score
