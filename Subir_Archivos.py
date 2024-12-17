from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
# Inicializar la aplicación Flask
app = Flask(__name__)

# Crear carpeta temporal para guardar imágenes
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Cargar el modelo entrenado
modelo = tf.keras.models.load_model("modelo_cifar10.h5")
class_names = ['avión', 'auto', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = ""
    imagen_path = None

    if request.method == "POST":
        img = request.files["imagen"]
        if img:
            # Guardar la imagen en la carpeta UPLOAD_FOLDER
            imagen_path = os.path.join(UPLOAD_FOLDER, img.filename)
            img.save(imagen_path)

            # Preprocesar la imagen
            img_cargada = image.load_img(imagen_path, target_size=(32, 32))
            img_array = image.img_to_array(img_cargada) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predecir
            prediccion = modelo.predict(img_array)
            clase_predicha = class_names[np.argmax(prediccion)]
            resultado = f"La imagen corresponde a: {clase_predicha}"

    return render_template("index.html", resultado=resultado, imagen=imagen_path)

if __name__ == "__main__":
    app.run(debug=True)
