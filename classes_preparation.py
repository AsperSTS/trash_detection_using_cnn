import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageOps

def encontrar_resolucion_minima(directorio):
  
  resolucion_minima = None
  file_name = ""
  for archivo in os.listdir(directorio):
    ruta_archivo = os.path.join(directorio, archivo)
    if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
      try:
        with Image.open(ruta_archivo) as imagen:
          ancho, alto = imagen.size
          if resolucion_minima is None or (ancho, alto) < resolucion_minima:
            resolucion_minima = (ancho, alto)
            file_name = archivo
      except IOError:
        print(f"No se pudo abrir la imagen: {ruta_archivo}")

  return resolucion_minima, file_name
def renombrar_imagenes(directorio_origen, directorio_destino, base_name):

  Path(directorio_destino).mkdir(parents=True, exist_ok=True)

  # Obtenemos la lista de archivos en el directorio de origen
  archivos = os.listdir(directorio_origen)

  # Contador para el número de imagen
  contador = 1

  for archivo in archivos:
    ruta_origen = os.path.join(directorio_origen, archivo)

    # Verificamos si el archivo es una imagen 
    if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
      nombre_nuevo = f"{base_name}{contador}{os.path.splitext(archivo)[1]}"
      ruta_destino = os.path.join(directorio_destino, nombre_nuevo)

      os.rename(ruta_origen, ruta_destino)

      contador += 1

def graph_count(classes, classes_count):
  # Crear la figura
  plt.figure(figsize=(16, 9))

  # Definir una lista de colores (añade más colores si tienes más clases)
  colores = ['skyblue', 'lightcoral', 'lightblue', 'lightgreen', 'peachpuff', 'hotpink']

  # Iterar sobre las clases y los colores, y graficar las barras con colores diferentes
  for i, clase in enumerate(classes):
      plt.bar(clase, classes_count[i], color=colores[i % len(colores)])

  # Añadir título y etiquetas
  plt.title('Conteo de imagenes por clase')
  plt.xlabel('Clases')
  plt.ylabel('Conteo de clase')

  plt.show()

def count_img_in_directory(directorio):
  Path(directorio).mkdir(parents=True, exist_ok=True)
  archivos = os.listdir(directorio)
  contador_archivos = 0
  for archivo in archivos:
    ruta_elemento = os.path.join(directorio, archivo)
    if os.path.isfile(ruta_elemento):
      contador_archivos += 1

  return contador_archivos
    


def redimensionar_con_relleno(imagen_ruta, size=(256, 256)):
    """
    Redimensiona la imagen a la resolución deseada sin deformarla.
    Aplica un relleno para mantener la proporción.
    """
    with Image.open(imagen_ruta) as img:
        # Convertir la imagen a RGB para evitar errores al guardar en JPEG
        img = img.convert("RGB")
        
        # Asegurarse de que la imagen esté en orientación horizontal
        if img.width < img.height:
            img = img.rotate(90, expand=True)
        
        # Redimensionar manteniendo la proporción y agregar relleno
        img.thumbnail(size, resample=3)
        delta_w = size[0] - img.width
        delta_h = size[1] - img.height
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        
        # Agregar relleno y guardar la imagen redimensionada
        img_padded = ImageOps.expand(img, padding, fill='white')
        return img_padded
def _gamma(v, gamma_val):
    # Paso 1: Dividimos cada valor de nuestro canal v por el valor de quantizacion
    normalized_v = v / 255.0
    # Paso 2: Elevamos al valor de gama
    corrected_v = np.power(normalized_v, gamma_val)
    # Paso 3: Multiplicamos por 255 para volver al rango de 0 a 255
    corrected_v = np.uint8(corrected_v * 255)    
    # Retornamos el valor con el operador gamma aplicado
    return corrected_v
  
def apply_clahe(v_channel, clip_limit=1.5, tile_grid_size=(4,4)):

  # Crear un objeto CLAHE
  clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

  # Aplicar CLAHE a la imagen
  v_channel_clahe = clahe.apply(v_channel)

  return v_channel_clahe
def preprocess_image(img):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[..., 2] = apply_clahe(img_hsv[..., 2], clip_limit=0.7, tile_grid_size=(8,8))
    img_hsv[..., 2] = _gamma(img_hsv[..., 2], 1)
    
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    
    img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), sigmaX=0.5)
    
    
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    img_sobel = cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))

    # res_stacked = np.hstack((img, img_bgr))
    # return [img_sobel, res_stacked]
    
    # img_multi_channel = np.concatenate((img_bgr, img_sobel), axis=2)
    return img_bgr, img_sobel

def procesar_directorio(directorio_origen, directorio_destino, size=(256, 256)):
    """
    Procesa todas las imágenes en el directorio de origen:
    - Asegura orientación horizontal
    - Redimensiona y añade relleno para que tengan la misma resolución.
    """
    Path(directorio_destino).mkdir(parents=True, exist_ok=True)
    archivos = os.listdir(directorio_origen)
    for archivo in archivos:
        if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            ruta_origen = os.path.join(directorio_origen, archivo)
            # img = preprocess_image(cv2.imread(ruta_origen))
            # img_procesada = redimensionar_con_relleno(ruta_origen, size)
            # ruta_destino = os.path.join(directorio_destino, archivo)
            # img_procesada.save(ruta_destino, format="JPEG")


if __name__ == "__main__":

  # WE COUNT THE IMAGES PER CLASS TO ADD SYNTETIC IMAGES IN THE FUTURE
  classes = np.array(["biologico", "desechos", "metal", "papel", "plasticoYtextil", "vidrio"])
  classes_count = np.array([])
  for i in range(0,6):
    classes_count = np.append(classes_count, count_img_in_directory(f"dataset_joined/{classes[i]}"))
  print(classes_count)
  
  # graph_count(classes, classes_count)
  
  # NOW WE CAN JUST RENAME THE FILES AND STORE IN ANOTHER DIRECTORY
  
  # Clase para renombrar los archivos y mandarlos a una carpeta nueva para la preparacion
  # for i in range(0,6):
  #   renombrar_imagenes(f"dataset/{classes[i]}", f"dataset_joined/{classes[i]}", f"{classes[i]}")
  
  
  for i in range(0,6):
    print(encontrar_resolucion_minima(f"dataset_joined/{classes[i]}"))
  
  # for clase in classes:
  #   procesar_directorio(f"dataset_joined/{clase}", f"dataset_normalized/{clase}")
    
    
    
  # img_1, img_2 = preprocess_image(cv2.imread("dataset_joined/plasticoYtextil/plasticoYtextil2450.jpg"))

  # np.savetxt("foo.csv", img_1, delimiter=",")
  # # for i in range(0,2):
  # #   cv2.imshow("Sobel",img_4[i])
  # #   cv2.waitKey(0)