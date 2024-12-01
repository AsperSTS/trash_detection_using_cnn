import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image #type: ignore
from PIL import Image, ImageTk

# Cargar el modelo guardado
model = tf.keras.models.load_model("waste_classification_model_8.keras")#"modelo_clasificacion_basura.keras")

# Definir las clases
classes = ["biologico", "desechos", "metal", "papel", "plasticoYtextil", "vidrio"]

def predict_image():
    # Obtener la ruta de la imagen
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Cargar y preprocesar la imagen
    img = image.load_img(file_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Hacer la predicción
    predictions = model.predict(img_array)
    print(predictions)
    predicted_class = classes[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Mostrar la imagen
    img = Image.open(file_path)
    img = img.resize((128, 128))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

    # Mostrar el resultado
    result_label.config(text=f"Predicción: {predicted_class}\nConfianza: {confidence:.2f}%")

# Crear la ventana principal
window = tk.Tk()
window.title("Clasificador de Basura")

# Botón para seleccionar la imagen
upload_button = tk.Button(window, text="Seleccionar Imagen", command=predict_image)
upload_button.pack(pady=20)

# Etiqueta para mostrar la imagen
image_label = tk.Label(window)
image_label.pack()

# Etiqueta para mostrar el resultado
result_label = tk.Label(window, text="")
result_label.pack(pady=20)

# Iniciar la interfaz gráfica
window.mainloop()