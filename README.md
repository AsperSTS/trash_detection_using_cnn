# Sistema de Clasificación de Residuos

Sistema de clasificación automática de residuos basado en Deep Learning y visión artificial. Identifica y clasifica diferentes tipos de residuos (orgánico, plástico, vidrio, papel, etc.) mediante imágenes.

## Instalación

```bash
git clone [repositorio]
pip install -r requirements.txt
```

## Componentes Principales

### 1. Pipeline de Imágenes (image_pipeline.py)
- Preprocesamiento de imágenes
- Balanceo de clases
- Redimensionamiento
- **Requisitos**: Python 3.8+, OpenCV, TensorFlow/Keras, NumPy

### 2. Entrenamiento (train.py)
- Implementa CNN para clasificación
- Guarda resultados en Google Sheets
- **Requisitos**: TensorFlow 2.17.0, Keras 3.4.1, Google Sheets API
- **Ejecución**: `python train.py`

### 3. Comparador de Imágenes (image_comparison.py)
- Interfaz web para comparar imágenes originales y procesadas
- Visualización de histogramas RGB y métricas
- **Requisitos**: Streamlit, OpenCV, NumPy, Matplotlib
- **Ejecución**: `streamlit run image_comparison.py`

### 4. Interfaz de Usuario (useModel.py)
- Interfaz gráfica para clasificación de imágenes
- Utiliza modelos entrenados
- **Requisitos**: TensorFlow, NumPy, Pillow, Tkinter
- **Ejecución**: `python useModel.py`

## Estructura de Directorios
```
├── step3_dataset_normalized_24k_config12/
│   ├── biologico/
│   ├── desechos/
│   ├── metal/
│   ├── papel/
│   ├── plasticoYtextil/
│   └── vidrio/
├── step3_dataset_normalized_24k_NoPreprocessing/
    └── [mismas subcarpetas]
```

## Requisitos Generales
- Python 3.8+
- Credenciales de Google Service Account (para train.py)
- Dependencias listadas en requirements.txt
