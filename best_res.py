import os
from collections import Counter
from PIL import Image

def encontrar_mejor_resolucion_cuadrada_con_relleno(directorio):
  """
  Encuentra la mejor resolución cuadrada para redimensionar un conjunto de imágenes 
  en un directorio dado, considerando la posibilidad de añadir relleno.

  Args:
    directorio: La ruta al directorio que contiene las imágenes.

  Returns:
    tuple: La resolución cuadrada más común (ancho, alto).
  """

  dimensiones = []
  for carpeta, _, archivos in os.walk(directorio):
    for archivo in archivos:
      if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
        ruta_imagen = os.path.join(carpeta, archivo)
        try:
          with Image.open(ruta_imagen) as img:
            ancho, alto = img.size
            dimensiones.extend([ancho, alto]) # Añadimos ancho y alto por separado
        except IOError:
          print(f"No se pudo abrir la imagen: {ruta_imagen}")

  if not dimensiones:
    return None, None  # No se encontraron imágenes

  # Encontrar la dimensión más común (considerando ancho y alto)
  dimension_mas_comun = Counter(dimensiones).most_common(1)[0][0]
  return dimension_mas_comun, dimension_mas_comun  # Devolver como resolución cuadrada

# Ejemplo de uso
directorio_imagenes = "step2_dataset_balanced" 
mejor_resolucion = encontrar_mejor_resolucion_cuadrada_con_relleno(directorio_imagenes)

if mejor_resolucion:
  ancho, alto = mejor_resolucion
  print(f"La mejor resolución cuadrada para redimensionar con relleno es: {ancho}x{alto}")
else:
  print("No se encontraron imágenes en el directorio.")