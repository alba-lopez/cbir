import numpy as np
import cv2
from distancia import *

def caract_en_uso(filename: str, metodo: str, model = None) -> np.array:
    if metodo == 'histograma_rgb':
        carac = histograma_rgb(filename)
    elif metodo == 'histograma_lab':
        carac = histograma_lab(filename)
    elif metodo == 'sift':
        carac = sift(filename)
    elif metodo == 'cnn_pool5' or metodo == 'cnn_fc2':
        carac = cnn(filename, model)
    return carac


def histograma_rgb2(img_path, bins = 255):
    img = cv2.imread(img_path)
    
    red, _ = np.histogram(img[:,:,0], bins)
    green, _ = np.histogram(img[:,:,1], bins)
    blue, _  = np.histogram(img[:,:,2], bins)
    
    hist = np.array([red, green, blue])
    return hist


def histograma_rgb(filename, bins = 8, mask=None):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convierte la imagen a espacio RGB
    histogram = cv2.calcHist([img], [0, 1, 2], mask, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    return histogram


def histograma_lab2(img_path, bins = 255):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    l, _ = np.histogram(img[:,:,0], bins)
    a, _ = np.histogram(img[:,:,1], bins)
    b, _  = np.histogram(img[:,:,2], bins)
    
    hist = np.array([l, a, b])
    return hist


def histograma_lab(filename, bins = 8, mask=None):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # Convierte la imagen a espacio L*a*b*.
    histogram = cv2.calcHist([img], [0, 1, 2], mask, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    return histogram


def sift(img_path):
    img = cv2.imread(img_path)
    # Transformar imagen de BGR a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Inicializar SIFT
    sift = cv2.SIFT_create()
    # Encontrar los puntos clave en la imagen
    _ , descriptors = sift.detectAndCompute(gray, mask=None)
    return descriptors


def cnn(filename: str, model) -> np.array:
    img = cv2.imread(filename)
    # Redimensionar la imagen a 224x224 píxeles
    img = cv2.resize(img, (224, 224))
    # Preprocesamiento
    x = np.expand_dims(img, axis=0)  # batch x width x height x channels
    x = x.astype(np.float32)
    x = x - x.mean()
    x = x / x.std()
    features = model.predict(x)
    features = features.flatten()
    return features
