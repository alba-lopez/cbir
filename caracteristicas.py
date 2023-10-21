import numpy as np
import cv2

def caract_en_uso(img: str, metodo: str) -> np.array:
    if metodo == 'histograma':
        h = histograma_rgb(img, 255)
    elif metodo == 'sift':
        h = sift(img)
    elif metodo == 'cnn':
        h = cnn(img)
    return h

def histograma_byn(img, bins):
    h, b = np.histogram(img, bins)
    return h

def histograma_rgb(img, bins):
    red, b = np.histogram(img[:,:,0], bins)
    green, b = np.histogram(img[:,:,1], bins)
    blue, b  = np.histogram(img[:,:,2], bins)
    h = np.array([red, green, blue])
    return h

def sift(img):
    pass

def cnn(img):
    pass