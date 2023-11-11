
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from typing import List
import numpy as np
import json
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

from caracteristicas import caract_en_uso
from distancia import *

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] ='sqlite:///imgs.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.static_folder = 'static2'

db = SQLAlchemy(app)

class Imagen(db.Model):
    id = db.Column(db.Integer, primary_key = True, autoincrement = True)
    nombre = db.Column(db.String, unique = True, nullable = False)
    foto = db.Column(db.BLOB())
    caract_rgb = db.Column(db.String)
    caract_lab = db.Column(db.String)
    caract_sift = db.Column(db.String)
    caract_pool5 = db.Column(db.String)
    caract_fc2 = db.Column(db.String)

with app.app_context():
    db.create_all()

def file_to_binary(filename: str):
    """Recibe la ubi de una imagen y devuelve su contenido en binario"""
    with open(filename, 'rb') as file:
        blob_data = file.read()
    return blob_data

def subir_imag(filename: str, model_pool5, model_fc2) -> None:  
    """Recibe el nombre de una imagen, calcula sus características, y las almacena en la BD"""
    img_consulta = Imagen.query.filter_by(nombre = filename).first() #comprobamos que no haya ninguna img con mismo nombre
    if not img_consulta:
        path = os.path.join('static2', filename)
        blob = file_to_binary(path)
        hist_rgb = repr(caract_en_uso(path, 'histograma_rgb').tolist())
        hist_lab = repr(caract_en_uso(path, 'histograma_lab').tolist())
        sift = repr(caract_en_uso(path, 'sift').tolist())
        cnn_pool5 = repr(caract_en_uso(path, 'cnn_pool5', model_pool5).tolist())
        cnn_fc2 = repr(caract_en_uso(path, 'cnn_fc2', model_fc2).tolist())
        img = Imagen(nombre = filename, foto = blob, caract_rgb = hist_rgb, caract_lab = hist_lab, caract_sift = sift, caract_pool5 = cnn_pool5, caract_fc2 = cnn_fc2)
        db.session.add(img)
        db.session.commit()
        print("Added!")
    else: 
        print("La imagen ya está almacenada")

def buscar_similares(filename: str, metodo: str) -> List[str]:
    """Recibe el nombre de una imagen, calcula sus características, y busca imágenes similares en la BD en base a sus características"""
    query_img_path = os.path.join('static2', filename)

    model = None
    if metodo == 'cnn_pool5':
        base_model = VGG16(weights='imagenet', include_top=False)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
    elif metodo == 'cnn_fc2':
        base_model = VGG16(weights='imagenet', include_top=True)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
    query_descriptors = np.float32(caract_en_uso(query_img_path, metodo, model))

    imgs_similares = []
    imgs = Imagen.query.all() 
    for img in imgs:
        if img.nombre != filename:
            if metodo == 'histograma_rgb':
                dist = euclidea(query_descriptors, np.array(json.loads(img.caract_rgb)))
                imgs_similares += [(img.nombre, dist)]
            if metodo == 'histograma_lab':
                dist = euclidea(query_descriptors, np.array(json.loads(img.caract_lab)))
                imgs_similares += [(img.nombre, dist)]
            elif metodo == 'sift':
                img_descriptors = np.float32(json.loads(img.caract_sift))
                similarity_score = sift_similarity_score(img_descriptors, query_descriptors, 0.5)
                imgs_similares.append((img.nombre, similarity_score))
            if metodo == 'cnn_pool5':
                dist = euclidea(query_descriptors, np.array(json.loads(img.caract_pool5)))
                imgs_similares += [(img.nombre, dist)]
            if metodo == 'cnn_fc2':
                dist = euclidea(query_descriptors, np.array(json.loads(img.caract_fc2)))
                imgs_similares += [(img.nombre, dist)]
    
    if metodo == 'histograma_rgb' or metodo == 'histograma_lab' or metodo == 'cnn_pool5' or metodo == 'cnn_fc2':                   
        imgs_similares_ordenada = sorted(imgs_similares, key=lambda x: x[1])
    elif metodo == 'sift':
        imgs_similares_ordenada = sorted(imgs_similares, key=lambda x: x[1], reverse=True) 

    return [t[0] for t in imgs_similares_ordenada]

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
            if metodo in ['histograma_rgb', 'histograma_lab', 'sift', 'cnn_pool5', 'cnn_fc2']:
                imgs = buscar_similares(filename, metodo)[:20]
                return render_template('form.html', filename = filename, imgs = imgs)
        return render_template('form.html', filename = filename)

@app.route('/display/<filename>')
def display_imag(filename):
    #return redirect(url_for('static', filename= filename), code=301)
    return send_from_directory('static2', filename)

@app.route('/nueva/')
def nuevas_imags():
    #imgs = ["pug.jpeg", "palmera.jpg", "aguacate.png", "astronauta.jpg", "casette.png", "encefalograma.jpg", "globo.jpg"]
    base_model = VGG16(weights='imagenet', include_top=False)
    model_pool5 = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

    base_model = VGG16(weights='imagenet', include_top=True)
    model_fc2 = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
    
    contenido = os.listdir('static2')
    for i in contenido:
        if os.path.isfile(os.path.join('static2', i)):
            subir_imag(i, model_pool5, model_fc2)
    return "Up to date"


if __name__ == '__main__':
    app.run(debug = True)

