import os
import numpy as np
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img # type: ignore
from glob import glob

# Configuración
dataset_path = 'step1_dataset_joined'
output_path = 'step2_dataset_balanced'
target_class_count = 2500  # Número objetivo de imágenes por clase

# Generador de aumentos de datos
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)

# Crear directorio de salida si no existe
os.makedirs(output_path, exist_ok=True)

def balancear_clases(clase, imagenes):
    """Equilibra la cantidad de imágenes para una clase específica."""
    class_output_dir = os.path.join(output_path, clase)
    os.makedirs(class_output_dir, exist_ok=True)

    # Si la clase tiene menos imágenes que el objetivo, generar imágenes adicionales
    if len(imagenes) < target_class_count:
        dif = target_class_count - len(imagenes)
        print(f"Generando {dif} imágenes para la clase {clase}")
        
        for i in range(dif):
            # Selecciona una imagen aleatoria
            img_path = np.random.choice(imagenes)
            img = load_img(img_path)  # Cargar imagen
            img_array = img_to_array(img)  # Convertir a array
            img_array = np.expand_dims(img_array, 0)  # Expansión para usar en datagen

            # Generar imagen aumentada
            for batch in datagen.flow(img_array, batch_size=1):
                new_img = batch[0].astype('uint8')
                new_img_path = os.path.join(class_output_dir, f"{clase}_aug_{i}.jpg")
                save_img(new_img_path, new_img)
                break  # Solo un aumento por iteración

    # Si la clase tiene más imágenes que el objetivo, eliminar algunas para equilibrar
    elif len(imagenes) > target_class_count:
        excess = len(imagenes) - target_class_count
        print(f"Reduciendo {excess} imágenes de la clase {clase}")
        
        # Seleccionar aleatoriamente imágenes a eliminar
        images_to_remove = np.random.choice(imagenes, excess, replace=False)
        for img_path in images_to_remove:
            os.remove(img_path)  # Eliminar imagen original

def balance_dataset():
    """Balancea todas las clases del dataset."""
    for clase in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, clase)
        if not os.path.isdir(class_dir):
            continue
        
        # Listar todas las imágenes de la clase
        imagenes = glob(os.path.join(class_dir, "*.jpg"))
        
        # Balancear imágenes de la clase
        balancear_clases(clase, imagenes)

# Ejecutar balanceo de dataset
balance_dataset()
print("Balanceo de clases completo.")
