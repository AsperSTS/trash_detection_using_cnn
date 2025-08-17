# Waste Classification System

An automated waste classification system based on Deep Learning and computer vision. Identifies and classifies different types of waste (organic, plastic, glass, paper, etc.) through images.

## Test Results

Metrics and graphs can be found in the following subfolders:
- `graficas_etapa1_rtx2050`
- `graficas_etapa2_rtx2050`
- `graficas_etapa2_rtx4060`
- `graficas_etapa3_rtx2050`

## Installation

```bash
git clone https://github.com/AsperSTS/trash_detection_using_cnn.git
pip install -r requirements.txt
```

## Parameter and Path Configuration

- Dataset paths must be modified according to your local setup

## Main Components

### 1. Image Pipeline (image_pipeline.py)
- Image preprocessing
- Class balancing
- Image resizing
- **Requirements**: Python 3.8+, OpenCV, TensorFlow/Keras, NumPy

### 2. Training Module (train.py)
- Implements CNN for classification
- Saves results to Google Sheets
- **Requirements**: TensorFlow 2.17.0, Keras 3.4.1, Google Sheets API
- **Execution**: `python train.py`

### 3. Image Comparison Tool (image_comparison.py)
- Web interface for comparing original and processed images
- RGB histogram visualization and metrics
- **Requirements**: Streamlit, OpenCV, NumPy, Matplotlib
- **Execution**: `streamlit run image_comparison.py`

### 4. User Interface (useModel.py)
- Graphical interface for image classification
- Uses trained models for prediction
- **Requirements**: TensorFlow, NumPy, Pillow, Tkinter
- **Execution**: `python useModel.py`

## Directory Structure for image_comparison.py

```
├── step3_dataset_normalized_con_preprocesamiento/
│   ├── biologico/
│   ├── desechos/
│   ├── metal/
│   ├── papel/
│   ├── plasticoYtextil/
│   └── vidrio/
├── step3_dataset_normalized_sin_preprocesamiento/
│   ├── biologico/
│   ├── desechos/
│   ├── metal/
│   ├── papel/
│   ├── plasticoYtextil/
│   └── vidrio/
```

## Waste Categories

- **biologico**: Organic/biological waste
- **desechos**: General waste
- **metal**: Metal materials
- **papel**: Paper and cardboard
- **plasticoYtextil**: Plastic and textile materials
- **vidrio**: Glass materials

## General Requirements

- Python 3.8+
- Google Service Account credentials (for train.py)
- Dependencies listed in requirements.txt

## Usage Workflow

1. **Setup**: Configure dataset paths and install dependencies
2. **Training**: Run `python train.py` to train the CNN model
3. **Comparison**: Use `streamlit run image_comparison.py` to analyze preprocessing effects
4. **Classification**: Execute `python useModel.py` for real-time waste classification

## Features

- Deep learning-based waste classification
- Multiple preprocessing pipelines
- Web-based comparison interface
- Google Sheets integration for results tracking
- User-friendly GUI for model inference

---

*Computer Vision Project for Environmental Sustainability*