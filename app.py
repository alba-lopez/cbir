
from flask import Flask, render_template


app = Flask(__name__)

def subir_imag(): 
    pass

@app.route('/')
def index():
    lista = []
    return render_template('form.html', lista_imags = lista)


if __name__ == '__main__':
    app.run(debug = True)
