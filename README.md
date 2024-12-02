# Clasificación de Residuos 

Este repositorio contiene el código para un sistema de clasificación de residuos que utiliza Deep Learning y técnicas de visión artificial. El sistema está diseñado para identificar y clasificar diferentes tipos de residuos, como orgánico, plástico, vidrio, papel, etc.

## Instalacion de requerimientos
1. Clona este repositorio.
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
## image_pipeline.py

## Descripción
Proporciona un flujo de trabajo para preprocesar imágenes, incluyendo balanceo de clases mediante aumentos de datos y redimensionamiento de imágenes para datasets desbalanceados.

## Requisitos
- Python 3.8+
- OpenCV
- TensorFlow/Keras
- NumPy

## Ejecucion
```bash
   python image_pipeline.py
```

## train.py

Utiliza una red neuronal convolucional (CNN) para clasificar residuos en diferentes categorías. El objetivo principal es entrenar un modelo eficiente que pueda distinguir entre seis tipos de residuos a partir de imágenes.


## Requisitos

Este proyecto requiere las siguientes bibliotecas:

- Python 3.8 o superior
- TensorFlow 2.17.0
- Keras 3.4.1
- NumPy
- Matplotlib
- Scikit-learn
- Google Sheets API (gspread y google-auth)
- Otros: `dataclasses`, `os`, `pathlib`, `logging`, `gc`

Asegúrate de tener un archivo JSON de credenciales de Google Service Account para guardar resultados en Google Sheets.

## Ejecucion
```bash
   python train.py
```

## image_comparisson.py

Permite comparar imágenes originales y preprocesadas desde dos directorios distintos. Proporciona herramientas visuales y métricas para evaluar diferencias entre las imágenes, como histogramas RGB y diferencias estructurales.


## Requisitos

Antes de ejecutar el código, asegúrate de tener instaladas las siguientes dependencias:

- Python 3.7+
- Streamlit
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- Pillow

### Estructura de directorios
├── step3_dataset_normalized_24k_config12/
│   ├── biologico/
│   ├── desechos/
│   ├── metal/
│   ├── papel/
│   ├── plasticoYtextil/
│   └── vidrio/
├── step3_dataset_normalized_24k_NoPreprocessing/
│   ├── biologico/
│   ├── desechos/
│   ├── metal/
│   ├── papel/
│   ├── plasticoYtextil/
│   └── vidrio/

## Ejecucion 
```bash
   streamlit run image_comparisson.py
```

## useModel.py

Pe clasificar imágenes de residuos en seis categorías predefinidas los modelos obtenidos con el codigo train.py y una interfaz gráfica construida con `Tkinter`.


## Requisitos

- Python 3.8 o superior.
- Bibliotecas necesarias:
  - `tensorflow`
  - `numpy`
  - `Pillow`
  - `tkinter` (incluida por defecto con Python en la mayoría de las plataformas)

## Ejecucion 
```bash
   streamlit run useModel.py
```
