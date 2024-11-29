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
    


def redimensionar_con_relleno(img, size=(160, 160)):
    """
    Redimensiona la imagen a la resolución deseada sin deformarla.
    Aplica un relleno para mantener la proporción, usando OpenCV.
    Esta función soporta imágenes de 1 canal (escala de grises) o de 3 canales (RGB).
    """
    # Verificar si la imagen es en escala de grises o a color
    if len(img.shape) == 3:  # Imagen RGB (3 canales)
        h, w, _ = img.shape
    elif len(img.shape) == 2:  # Imagen en escala de grises (1 canal)
        h, w = img.shape
    else:
        raise ValueError("La imagen no tiene el formato adecuado")

    # Calcular las proporciones de redimensionado
    aspect_ratio = w / h
    target_w, target_h = size

    if aspect_ratio > 1:  # Imagen más ancha que alta
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    else:  # Imagen más alta que ancha o cuadrada
        new_h = target_h
        new_w = int(target_h * aspect_ratio)
    
    # Redimensionar la imagen manteniendo la proporción
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # Calcular el relleno necesario para ajustar a la resolución target
    delta_w = target_w - new_w
    delta_h = target_h - new_h
    
    # Verificar el número de canales de la imagen original
    if len(img.shape) == 3:  # Imagen RGB (3 canales)
        # Crear un fondo blanco de las dimensiones target (3 canales)
        img_padded = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
        # Insertar la imagen redimensionada en el centro del fondo
        img_padded[delta_h // 2: delta_h // 2 + new_h, delta_w // 2: delta_w // 2 + new_w] = img_resized
    elif len(img.shape) == 2:  # Imagen en escala de grises (1 canal)
        # Crear un fondo blanco de las dimensiones target (1 canal)
        img_padded = np.full((target_h, target_w), 255, dtype=np.uint8)
        # Insertar la imagen redimensionada en el centro del fondo
        img_padded[delta_h // 2: delta_h // 2 + new_h, delta_w // 2: delta_w // 2 + new_w] = img_resized
    
    return img_padded
  
def redimensionar_sin_relleno(img, size=(160, 160)):
    """
    Redimensiona la imagen a 100x100 sin deformarla, 
    utilizando interpolación bicúbica. 
    No aplica relleno.
    """

    # Redimensionar la imagen usando interpolación bicúbica
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    
    return img_resized
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
# def preprocess_image(img):

#     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     # img_hsv[..., 2] = apply_clahe(img_hsv[..., 2], clip_limit=0.7, tile_grid_size=(8,8))
#     img_hsv[..., 2] = _gamma(img_hsv[..., 2], 0.9)
    
#     img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    
#     img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), sigmaX=0.5)
    
    
#     # img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     # sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
#     # sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
#     # img_sobel = cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))

#     # res_stacked = np.hstack((img, img_bgr))
#     # return [img_sobel, res_stacked]
    
#     # img_multi_channel = np.concatenate((img_bgr, img_sobel), axis=2)
#     return img_bgr ##, img_sobel

def preprocess_image(img):
    # Convertir a HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Aplicar CLAHE y gamma en el canal V
    img_hsv[..., 2] = apply_clahe(img_hsv[..., 2], clip_limit=0.7, tile_grid_size=(6,6))
    img_hsv[..., 2] = _gamma(img_hsv[..., 2], 1.1)
    
    # Volver a BGR
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    
    # Aplicar suavizado adaptativo
    img_bgr = cv2.bilateralFilter(img_bgr, 5, 30, 30)
    
    return img_bgr
def preprocess_image_with_edges(img):
    """
    Preprocesa una imagen RGB incorporando información de bordes
    mientras mantiene los 3 canales de color.
    """
    # Convertir a HSV para el ajuste de brillo/contraste
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[..., 2] = apply_clahe(img_hsv[..., 2], clip_limit=0.7, tile_grid_size=(8,8))
    img_hsv[..., 2] = _gamma(img_hsv[..., 2], 0.9)
    
    # Volver a BGR y aplicar suavizado
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), sigmaX=0.5)
    
    # Detectar bordes en cada canal por separado
    edges_b = cv2.Canny(img_bgr[..., 0], 100, 200)
    edges_g = cv2.Canny(img_bgr[..., 1], 100, 200)
    edges_r = cv2.Canny(img_bgr[..., 2], 100, 200)
    
    # Combinar los bordes detectados
    edges_combined = cv2.max(cv2.max(edges_b, edges_g), edges_r)
    
    # Crear una máscara de bordes de 3 canales
    edges_mask = cv2.cvtColor(edges_combined, cv2.COLOR_GRAY2BGR)
    
    # Mezclar la imagen original con los bordes detectados
    # Alpha = 0.75 significa 75% imagen original, 30% bordes
    alpha = 0.90
    enhanced_img = cv2.addWeighted(img_bgr, alpha, edges_mask, 1-alpha, 0)
    
    return enhanced_img
# def procesar_directorio(directorio_origen, directorio_destino, clase, size=(160, 160)):
#     Path(directorio_destino).mkdir(parents=True, exist_ok=True)
#     archivos = os.listdir(directorio_origen)

#     contador = 1


#     for archivo in archivos:
#         if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
#             ruta_origen = os.path.join(directorio_origen, archivo)

#             # img_bgr = preprocess_image(cv2.imread(ruta_origen))
#             # img_bgr = preprocess_image_with_edges(cv2.imread(ruta_origen))
#             # img_bgr = cv2.imread(ruta_origen)
#             # Guardar imagen BGR redimensionada
#             img_bgr = redimensionar_sin_relleno(cv2.imread(ruta_origen), size)
#             ruta_destino_bgr = os.path.join(directorio_destino, f"{clase}{contador}.jpg")
#             cv2.imwrite(ruta_destino_bgr, img_bgr)
#             contador += 1

#             # Guardar imagen Sobel redimensionada
# #             img_sobel = redimensionar_sin_relleno(img_sobel, size)
# #             ruta_destino_sobel = os.path.join(directorio_destino, f"{clase}{contador}.jpg")
# #             cv2.imwrite(ruta_destino_sobel, img_sobel, [cv2.IMWRITE_JPEG_QUALITY, 80])
# #             contador += 1
# # # 
def procesar_directorio(directorio_origen, directorio_destino, clase, size=(160, 160)):
    Path(directorio_destino).mkdir(parents=True, exist_ok=True)
    archivos = os.listdir(directorio_origen)
    
    for idx, archivo in enumerate(archivos, 1):
        if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            ruta_origen = os.path.join(directorio_origen, archivo)
            
            # Leer y preprocesar
            img_original = cv2.imread(ruta_origen)
            if img_original is None:
                continue
                
            # Preprocesar antes de redimensionar
            img_processed = preprocess_image(img_original)
            
            # Redimensionar manteniendo aspecto
            img_resized = redimensionar_con_relleno(img_processed, size)
            
            # Guardar con calidad optimizada
            ruta_destino = os.path.join(directorio_destino, f"{clase}{idx}.jpg")
            cv2.imwrite(ruta_destino, img_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
if __name__ == "__main__":

	# WE COUNT THE IMAGES PER CLASS TO ADD SYNTETIC IMAGES IN THE FUTURE
	classes = np.array(["biologico", "desechos", "metal", "papel", "plasticoYtextil", "vidrio"])
	classes_count = np.array([])
	for i in range(0,6):
		classes_count = np.append(classes_count, count_img_in_directory(f"step2_dataset_balanced/{classes[i]}"))
	print(classes_count)

	# graph_count(classes, classes_count)

	# NOW WE CAN JUST RENAME THE FILES AND STORE IN ANOTHER DIRECTORY

	# Clase para renombrar los archivos y mandarlos a una carpeta nueva para la preparacion
	# for i in range(0,6):
	#   renombrar_imagenes(f"dataset/{classes[i]}", f"dataset_joined/{classes[i]}", f"{classes[i]}")


	# for i in range(0,6):
	#   print(encontrar_resolucion_minima(f"dataset_joined/{classes[i]}"))

	# for clase in classes:
	#   procesar_directorio(f"dataset_joined/{clase}", f"dataset_normalized/{clase}")


	for element in classes:
		procesar_directorio(f"step2_dataset_balanced/{element}", f"step3_dataset_normalized/{element}", element)
