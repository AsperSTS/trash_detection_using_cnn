import os
import requests
from bs4 import BeautifulSoup #type: ignore
from urllib.parse import urljoin
from PIL import Image
from io import BytesIO

def descargar_imagenes(query, num_imagenes=5, dir_destino='test_images'):
  """
  Descarga imágenes de Google Images.

  Args:
    query: Término de búsqueda.
    num_imagenes: Número de imágenes a descargar.
    dir_destino: Directorio donde se guardarán las imágenes.
  """

  # Crear el directorio de destino si no existe
  os.makedirs(dir_destino, exist_ok=True)

  # URL de búsqueda de Google Images
  url = f'https://www.google.com/search?q={query}&tbm=isch'

  # Obtener el contenido de la página
  response = requests.get(url)
  soup = BeautifulSoup(response.content, 'html.parser')

  # Encontrar las etiquetas <img>
  imagenes = soup.find_all('img')

  # Descargar las imágenes
  for i, imagen in enumerate(imagenes[:num_imagenes]):
    try:
      # Obtener la URL de la imagen
      url_imagen = imagen['src']

      # Descargar la imagen
      response = requests.get(url_imagen)
      imagen_pil = Image.open(BytesIO(response.content))

      # Guardar la imagen
      nombre_archivo = os.path.join(dir_destino, f'{query}_{i+1}.jpg')
      imagen_pil.save(nombre_archivo)
      print(f'Imagen guardada: {nombre_archivo}')

    except Exception as e:
      print(f'Error al descargar la imagen {i+1}: {e}')

# Lista de términos de búsqueda
queries = ['papel', 'carton', 'metal', 'sopa', 'calzado', 
           'desechos', 'biologico', 'baterias', 'vidrio']

# Descargar 5 imágenes para cada término de búsqueda
for query in queries:
  descargar_imagenes(query)