# Clasificación de Residuos con Deep Learning

Este repositorio contiene el código para un sistema de clasificación de residuos que utiliza Deep Learning y técnicas de visión artificial. El sistema está diseñado para identificar y clasificar diferentes tipos de residuos, como orgánico, plástico, vidrio, papel, etc.

## Pipeline de Preprocesamiento de Imágenes (`image_pipeline.py`)

Este script implementa un pipeline para preprocesar y balancear un conjunto de datos de imágenes, preparándolo para su uso en el entrenamiento del modelo de clasificación.

### Funcionalidades

1. **Balanceo de clases:**
   - Aumenta las clases con pocas imágenes mediante técnicas de aumento de datos (data augmentation) para equilibrar el conjunto de datos.
   - Utiliza la biblioteca `ImageDataGenerator` de Keras para generar nuevas imágenes a partir de las existentes, aplicando transformaciones aleatorias como rotaciones, desplazamientos, zoom y volteos.

2. **Preprocesamiento de imágenes:**
   - Aplica correcciones de color y contraste utilizando CLAHE (Contrast Limited Adaptive Histogram Equalization) y corrección gamma.
   - Reduce el ruido y suaviza las imágenes con un filtro bilateral.
   - Redimensiona las imágenes al tamaño especificado en la configuración.

### Configuración

La configuración del pipeline se define en la clase `ProcessingConfig`.  Puedes modificar los siguientes parámetros:

* `target_size`:  Tamaño objetivo de las imágenes después del redimensionamiento.
* `target_class_count`: Número objetivo de imágenes por clase después del balanceo.
* `quality`: Calidad de las imágenes guardadas (0-100).
* `supported_formats`: Formatos de imagen soportados.

### Uso

1. **Organiza tu conjunto de datos:**
   - Coloca las imágenes en carpetas separadas para cada clase.
   - La estructura del directorio debe ser la siguiente:

     ```
     dataset/
       clase1/
         imagen1.jpg
         imagen2.jpg
         ...
       clase2/
         imagen1.jpg
         imagen2.jpg
         ...
       ...
     ```

2. **Ajusta la configuración:**
   - Modifica los parámetros en la clase `ProcessingConfig` según tus necesidades.

3. **Ejecuta el script:**
   - `python image_pipeline.py`

El script procesará las imágenes y guardará el conjunto de datos balanceado y preprocesado en un nuevo directorio.


## Entrenamiento del Modelo (`train_model.py`)

Este script entrena un modelo de clasificación de imágenes para identificar diferentes tipos de residuos.

### Funcionalidades

* **Construcción del modelo:**
    * Define una arquitectura de red neuronal convolucional (CNN) con capas convolucionales, pooling, dropout y normalización por lotes.
    * Utiliza el optimizador Adam con AMSGrad para un entrenamiento más eficiente.
    * Compila el modelo con la función de pérdida 'categorical_crossentropy' y las métricas 'accuracy', 'AUC', 'Precision' y 'Recall'.

* **Entrenamiento del modelo:**
    * Carga y preprocesa un conjunto de datos de imágenes de residuos.
    * Divide los datos en conjuntos de entrenamiento y validación.
    * Entrena el modelo utilizando la configuración especificada, incluyendo el número de épocas, la tasa de aprendizaje y el tamaño del lote.
    * Utiliza callbacks para monitorear el entrenamiento, ajustar la tasa de aprendizaje y guardar el mejor modelo.

* **Visualización y guardado de resultados:**
    * Genera gráficos de las métricas de entrenamiento (accuracy, loss, AUC, precision, recall).
    * Guarda el modelo entrenado en un archivo `.keras`.
    * Guarda un resumen de la arquitectura del modelo en un archivo de texto.

### Configuración

La configuración del modelo y el entrenamiento se define en la clase `ModelConfig`. Puedes modificar los siguientes parámetros:

* `train_dir`: Directorio que contiene el conjunto de datos de imágenes.
* `classes`:  Lista de las clases de residuos.
* `batch_size`: Tamaño del lote para el entrenamiento.
* `image_size`: Tamaño de las imágenes de entrada.
* `validation_split`: Porcentaje de datos utilizado para validación.
* `epochs`: Número de épocas de entrenamiento.
* `learning_rate`: Tasa de aprendizaje inicial.
* `preprocess_config`:  Número de configuración de preprocesamiento (opcional).
* `metrics_dir`: Directorio para guardar las métricas y los resultados.
* `seed`: Semilla aleatoria para la reproducibilidad.
* `experiment_name`: Nombre del experimento (opcional).

### Uso

1. **Prepara el conjunto de datos:**
   * Organiza las imágenes de residuos en carpetas separadas para cada clase.
   * Asegúrate de que el conjunto de datos esté preprocesado y balanceado utilizando `image_pipeline.py`.

2. **Ajusta la configuración:**
   * Modifica los parámetros en la clase `ModelConfig` según tus necesidades.

3. **Ejecuta el script:**
   * `python train_model.py`

El script entrenará el modelo y guardará los resultados en el directorio especificado.

## Dependencias

* Python 3.7+
* OpenCV (`cv2`)
* Pillow (`PIL`)
* TensorFlow (`tensorflow`)
* Matplotlib (`matplotlib`)
* NumPy (`numpy`)
* gspread (`gspread`)
* google-auth-httplib2 (`google-auth-httplib2`)
* google-auth-oauthlib (`google-auth-oauthlib`)

Puedes instalar las dependencias con `pip`:

```bash
pip install opencv-python Pillow tensorflow matplotlib numpy gspread google-auth-httplib2 google-auth-oauthlib