
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from typing import List
import cv2
import numpy as np

from caracteristicas import caract_en_uso
from distancia import dist_en_uso

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] ='sqlite:///imgs.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Imagen(db.Model):
    id = db.Column(db.Integer, primary_key = True, autoincrement = True)
    nombre = db.Column(db.String, unique = True, nullable = False)
    foto = db.Column(db.BLOB())
    caract = db.Column(db.String)

with app.app_context():
    db.create_all()

def leer_img(filename: str) -> np.array:
    """Recibe la ubi de una imagen y devuelve sus componentes BGR"""
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    return img

def calcular_caract(filename: str) -> np.array:
    """Recibe la ubi de una imagen y devuelve su vector de características"""
    img = leer_img(filename)
    caract = caract_en_uso(img) #funcion para calcular vector
    return caract

def file_to_binary(filename: str):
    """Recibe la ubi de una imagen y devuelve su contenido en binario"""
    with open(filename, 'rb') as file:
        blob_data = file.read()
    return blob_data

def subir_imag(filename: str) -> None:  
    """Recibe la ubi de una imagen, calcula sus características, y las almacena en la BD"""
    n = Imagen.query.filter_by(nombre = filename).first() #comprobamos que no haya ninguna img con mismo nombre
    if not n:
        blob = file_to_binary(filename)
        c = repr(calcular_caract(filename).tolist())
        img = Imagen(nombre = filename, foto = blob, caract = c)
        db.session.add(img)
        db.session.commit()
        print("Added!")
    else: 
        print("La imagen ya está almacenada")

def take_first(elem):
    return elem[0]

def buscar_similares(img: str) -> List[str]:
    """Recibe una imagen, calcula sus características, y busca imágenes similares en la BD en base a sus características"""
    caract_new = calcular_caract(img)
    tupla = []
    imgs = Imagen.query.select() #### corregir
    for i in imgs:
        nombre = i.nombre
        d = dist_en_uso(caract_new, i.caract)
        tupla += [(nombre, d)]
    tupla_ordenada = sorted(tupla, reverse= True, key=take_first)
    print(tupla_ordenada)


@app.route('/')
def index():
    lista = []
    return render_template('form.html', lista_imags = lista)

@app.route('/nueva/')
def nuevas_imags():
    imgs = ["templates\static\pug.jpeg", "templates\static\palmera.jpg"]
    for i in imgs:
        subir_imag(i)
    return "Up to date"

@app.route('/buscar/')
def buscar_imags():
    img = "templates\static\pug.jpeg"
    buscar_similares(img)

if __name__ == '__main__':
    app.run(debug = True)

