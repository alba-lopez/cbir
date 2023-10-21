
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from typing import List
import cv2
import numpy as np
import json
from werkzeug.utils import secure_filename
import os

from caracteristicas import caract_en_uso
from distancia import dist_en_uso

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] ='sqlite:///imgs.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.static_folder = 'static'

db = SQLAlchemy(app)

class Imagen(db.Model):
    id = db.Column(db.Integer, primary_key = True, autoincrement = True)
    nombre = db.Column(db.String, unique = True, nullable = False)
    foto = db.Column(db.BLOB())
    caract = db.Column(db.String)
    #caract_sift = db.Column(db.String)
    #caract_cnn = db.Column(db.String)

with app.app_context():
    db.create_all()

def leer_img(filename: str) -> np.array:
    """Recibe la ubi de una imagen y devuelve sus componentes BGR"""
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    return img

def calcular_caract(filename: str, metodo: str) -> np.array:
    """Recibe la ubi de una imagen y devuelve su vector de características"""
    img = leer_img(filename)
    caract = caract_en_uso(img, metodo) #funcion para calcular vector
    return caract

def file_to_binary(filename: str):
    """Recibe la ubi de una imagen y devuelve su contenido en binario"""
    with open(filename, 'rb') as file:
        blob_data = file.read()
    return blob_data

def subir_imag(filename: str) -> None:  
    """Recibe el nombre de una imagen, calcula sus características, y las almacena en la BD"""
    n = Imagen.query.filter_by(nombre = filename).first() #comprobamos que no haya ninguna img con mismo nombre
    if not n:
        path = os.path.join('static', filename)
        print(path)
        blob = file_to_binary(path)
        hist = repr(calcular_caract(path, 'histograma').tolist())
        img = Imagen(nombre = filename, foto = blob, caract = hist)
        db.session.add(img)
        db.session.commit()
        print("Added!")
    else: 
        print("La imagen ya está almacenada")

def take_first(elem):
    return elem[0]

def buscar_similares(filename: str, metodo: str) -> List[str]:
    """Recibe el nombre de una imagen, calcula sus características, y busca imágenes similares en la BD en base a sus características"""
    path = os.path.join('static', filename)
    caract_new = calcular_caract(path, metodo)
    tupla = []
    imgs = Imagen.query.all() 
    for i in imgs:
        nombre = i.nombre
        if i.nombre != filename:
            d = dist_en_uso(caract_new, np.array(json.loads(i.caract)))
            tupla += [(nombre, d)]
    tupla_ordenada = sorted(tupla, reverse= True, key=take_first)
    print(tupla_ordenada)
    return [t[0] for t in tupla_ordenada]

@app.route('/')
def index():
    lista = []
    return render_template('form.html', lista_imags = lista)

@app.route('/', methods=['POST'])
def upload_imag():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    else: 
        filename = secure_filename(file.filename)
        if 'metodo' in request.form:
            metodo = request.form['metodo']
            if metodo in ['histograma', 'sift', 'cnn']:
                imgs = buscar_similares(filename, metodo)[:20]
                return render_template('form.html', filename = filename, imgs = imgs)
        return render_template('form.html', filename = filename)

@app.route('/display/<filename>')
def display_imag(filename):
    #return redirect(url_for('static', filename= filename), code=301)
    return send_from_directory('static', filename)

@app.route('/nueva/')
def nuevas_imags():
    #imgs = ["pug.jpeg", "palmera.jpg", "aguacate.png", "astronauta.jpg", "casette.png", "encefalograma.jpg", "globo.jpg"]
    contenido = os.listdir(app.static_folder)
    for i in contenido:
        if os.path.isfile(os.path.join(app.static_folder, i)):
            subir_imag(i)
    return "Up to date"


if __name__ == '__main__':
    app.run(debug = True)

