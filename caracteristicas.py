import numpy as np
import cv2
from distancia import *

def caract_en_uso(filename: str, metodo: str) -> np.array:
    if metodo == 'histograma':
        carac = histograma_rgb(filename, 255)
    elif metodo == 'sift':
        carac = sift(filename)
    elif metodo == 'cnn':
        carac = cnn(filename)
    return carac

def histograma_byn(img_path, bins):
    img = cv2.imread(img_path)
    hist, _ = np.histogram(img, bins)
    return hist

def histograma_rgb(img_path, bins):
    img = cv2.imread(img_path)
    
    red, _ = np.histogram(img[:,:,0], bins)
    green, _ = np.histogram(img[:,:,1], bins)
    blue, _  = np.histogram(img[:,:,2], bins)
    
    hist = np.array([red, green, blue])
    return hist

def sift(img_path):
    img = cv2.imread(img_path)

    # Transformar imagen de BGR a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Inicializar SIFT
    sift = cv2.SIFT_create()

    # Encontrar los puntos clave en la imagen
    _ , descriptors = sift.detectAndCompute(gray, mask=None)

    return descriptors

def cnn(img):
    pass
