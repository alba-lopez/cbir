import numpy as np
import cv2

def caract_en_uso(img: str) -> np.array:
    h = histograma_byn(img, 255) #cambiar: función de características a usar
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