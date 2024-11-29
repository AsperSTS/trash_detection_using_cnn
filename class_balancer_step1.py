import os
import numpy as np
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img #type: ignore
from glob import glob
import cv2

# Configuración
dataset_path = 'step1_dataset_joined'
output_path = 'step2_dataset_balanced'
target_class_count = 4140

# Generador de aumentos de datos con ajustes para preservar calidad
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.2,
    fill_mode='reflect',  # Cambiado a 'reflect' para mejor manejo de bordes
    dtype=np.uint8  # Asegurar que no hay pérdida de precisión
)

# Crear directorio de salida si no existe
os.makedirs(output_path, exist_ok=True)

def load_image_high_quality(img_path):
    """Carga la imagen manteniendo la calidad original."""
    return cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def save_image_high_quality(img_path, img):
    """Guarda la imagen con alta calidad."""
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])

def balancear_clases(clase, imagenes):
    """Equilibra la cantidad de imágenes manteniendo la calidad."""
    class_output_dir = os.path.join(output_path, clase)
    os.makedirs(class_output_dir, exist_ok=True)
    
    # Copiar imágenes originales con máxima calidad
    print(f"Copiando {len(imagenes)} imágenes originales de la clase {clase}")
    for i, img_path in enumerate(imagenes):
        dest_path = os.path.join(class_output_dir, f"{clase}_original_{i}.jpg")
        shutil.copy2(img_path, dest_path)
    
    current_count = len(imagenes)
    if current_count < target_class_count:
        dif = target_class_count - current_count
        print(f"Generando {dif} imágenes aumentadas para la clase {clase}")
        
        for i in range(dif):
            # Seleccionar imagen aleatoria
            img_path = np.random.choice(imagenes)
            
            # Cargar imagen con alta calidad
            img = load_image_high_quality(img_path)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, 0)
            
            # Generar imagen aumentada preservando calidad
            for batch in datagen.flow(img_array, batch_size=1):
                new_img = batch[0].astype('uint8')
                new_img_path = os.path.join(class_output_dir, f"{clase}_aug_{i}.jpg")
                save_image_high_quality(new_img_path, new_img)
                break

def balance_dataset():
    """Balancea todas las clases del dataset."""
    print("Iniciando proceso de balanceo con preservación de calidad...")
    
    for clase in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, clase)
        if not os.path.isdir(class_dir):
            continue
        
        # Listar imágenes de la clase
        imagenes = glob(os.path.join(class_dir, "*.jpg"))
        print(f"\nProcesando clase {clase} - {len(imagenes)} imágenes encontradas")
        
        balancear_clases(clase, imagenes)

if __name__ == "__main__":
    balance_dataset()
    print("\nBalanceo de clases completado con preservación de calidad.")



# import os
# import numpy as np
# import shutil
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img # type: ignore
# from glob import glob

# # Configuración
# dataset_path = 'step1_dataset_joined'
# output_path = 'step2_dataset_balanced'
# target_class_count = 4140  # Número objetivo de imágenes por clase

# # Generador de aumentos de datos
# datagen = ImageDataGenerator(
#     rotation_range=30,          # Aumentar rango de rotación
#     width_shift_range=0.2,      # Aumentar shift
#     height_shift_range=0.2,     # Aumentar shift
#     zoom_range=0.2,             # Añadir zoom
#     horizontal_flip=True,
#     vertical_flip=True,         # Añadir flip vertical
#     shear_range=0.2,            # Añadir transformación de cizallamiento
#     fill_mode='nearest'
# )

# # Crear directorio de salida si no existe
# os.makedirs(output_path, exist_ok=True)

# # def balancear_clases(clase, imagenes):
# #     """Equilibra la cantidad de imágenes para una clase específica."""
# #     class_output_dir = os.path.join(output_path, clase)
# #     os.makedirs(class_output_dir, exist_ok=True)

# #     # Si la clase tiene menos imágenes que el objetivo, generar imágenes adicionales
# #     if len(imagenes) < target_class_count:
# #         dif = target_class_count - len(imagenes)
# #         print(f"Generando {dif} imágenes para la clase {clase}")
        
# #         for i in range(dif):
# #             # Selecciona una imagen aleatoria
# #             img_path = np.random.choice(imagenes)
# #             img = load_img(img_path)  # Cargar imagen
# #             img_array = img_to_array(img)  # Convertir a array
# #             img_array = np.expand_dims(img_array, 0)  # Expansión para usar en datagen

# #             # Generar imagen aumentada
# #             for batch in datagen.flow(img_array, batch_size=1):
# #                 new_img = batch[0].astype('uint8')
# #                 new_img_path = os.path.join(class_output_dir, f"{clase}_aug_{i}.jpg")
# #                 save_img(new_img_path, new_img)
# #                 break  # Solo un aumento por iteración

# #     # Si la clase tiene más imágenes que el objetivo, eliminar algunas para equilibrar
# #     elif len(imagenes) > target_class_count:
# #         excess = len(imagenes) - target_class_count
# #         print(f"Reduciendo {excess} imágenes de la clase {clase}")
        
# #         # Seleccionar aleatoriamente imágenes a eliminar
# #         images_to_remove = np.random.choice(imagenes, excess, replace=False)
# #         for img_path in images_to_remove:
# #             os.remove(img_path)  # Eliminar imagen original
# def balancear_clases(clase, imagenes):
#     class_output_dir = os.path.join(output_path, clase)
#     os.makedirs(class_output_dir, exist_ok=True)
    
#     # Primero copiar todas las imágenes originales
#     for img_path in imagenes:
#         shutil.copy2(img_path, class_output_dir)
    
#     # Luego generar imágenes adicionales si es necesario
#     current_count = len(imagenes)
#     if current_count < target_class_count:
#         dif = target_class_count - current_count
#         print(f"Generando {dif} imágenes para la clase {clase}")
        
#         for i in range(dif):
#             img_path = np.random.choice(imagenes)
#             img = load_img(img_path)
#             img_array = img_to_array(img)
#             img_array = np.expand_dims(img_array, 0)
            
#             for batch in datagen.flow(img_array, batch_size=1):
#                 new_img = batch[0].astype('uint8')
#                 new_img_path = os.path.join(class_output_dir, f"{clase}_aug_{i}.jpg")
#                 save_img(new_img_path, new_img)
#                 break
# def balance_dataset():
#     """Balancea todas las clases del dataset."""
#     for clase in os.listdir(dataset_path):
#         class_dir = os.path.join(dataset_path, clase)
#         if not os.path.isdir(class_dir):
#             continue
        
#         # Listar todas las imágenes de la clase
#         imagenes = glob(os.path.join(class_dir, "*.jpg"))
        
#         # Balancear imágenes de la clase
#         balancear_clases(clase, imagenes)

# # Ejecutar balanceo de dataset
# balance_dataset()
# print("Balanceo de clases completo.")
