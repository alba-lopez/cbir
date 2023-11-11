
# Estructura de directorios: 
cbir/

    app.py : API Flask que implementa el sistema CBIR
    caracteristicas.py : fichero con funciones que generan las características
    distancia.py : fichero con funciones que comparan las características

    
    instance/
        imgs.db : BD de características (pesa demasiado, y no está en GitHub. Se genera como se menciona en 1., o se puede descargar generada)
        
    static/ : BD de imágenes WangDataSet

    templates/
        form.html : template de página web recuperada desde el navegador
        
    knn_pruebas : implementación de knn con el histograma, y evaluación de rendimiento
    knn_pruebas_sift : implementación de sift, y evaluación de rendimiento
    knn_pruebas_cnn : implementación de knn con el cnn, y evaluación de rendimiento

# Para ejecutar: 
desde la carpeta cbir:
cbir> python app.py


1. Generar BD: acceder a http://127.0.0.1:5000/nueva/ desde el navegador (o descomprimir imgs.zip)
2. Buscar imágenes similares a otra: acceder a http://127.0.0.1:5000/, seleccionar parámetros y 'submit'


# Para consultar tablas
doble click en .\sqlite-tools-win32-x86-3410200\sqlite3.exe

sqlite> .open './cbir/instance/imgs.db'

sqlite> .tables

sqlite> SELECT * FROM imagen;
