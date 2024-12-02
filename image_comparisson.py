import streamlit as st
import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def load_image(image_path):
    """Carga y retorna una imagen usando OpenCV"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb

def plot_histograms(img1, img2):
    """Genera histogramas RGB comparativos"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        hist1 = cv2.calcHist([img1], [i], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [i], None, [256], [0, 256])
        
        ax1.plot(hist1, color=color, alpha=0.7)
        ax2.plot(hist2, color=color, alpha=0.7)
    
    ax1.set_title('Histograma Original')
    ax2.set_title('Histograma Preprocesada')
    ax1.grid(True)
    ax2.grid(True)
    
    return fig

def calculate_image_differences(img1, img2):
    """Calcula métricas de diferencia entre imágenes"""
    # MSE
    mse = np.mean((img1 - img2) ** 2)
    
    # PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Diferencia estructural
    diff = cv2.absdiff(img1, img2)
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    return mse, psnr, diff_norm

def main():
    st.set_page_config(layout="wide")
    st.title("Comparador de Imágenes Originales vs Preprocesadas")
    
    # Directorios base
    dir_prep = "step3_dataset_normalized_24k_config12"
    dir_noprep = "step3_dataset_normalized_24k_NoPreprocessing"
    folders = ["biologico", "desechos", "metal", "papel", "plasticoYtextil", "vidrio"]
    
    # Sidebar para controles
    with st.sidebar:
        st.header("Controles")
        selected_folder = st.selectbox("Seleccionar categoría:", folders)
        
        # Obtener lista de imágenes
        image_files = os.listdir(os.path.join(dir_noprep, selected_folder))
        
        # Modo de selección
        selection_mode = st.radio("Modo de selección:", 
                                ["Aleatorio", "Manual"])
        
        if selection_mode == "Aleatorio":
            if st.button("Cargar imagen aleatoria"):
                st.session_state.current_image = random.choice(image_files)
        else:
            st.session_state.current_image = st.selectbox("Seleccionar imagen:", 
                                                         image_files)
    
    # Main content
    if 'current_image' in st.session_state:
        # Cargar imágenes
        img1_path = os.path.join(dir_noprep, selected_folder, st.session_state.current_image)
        img2_path = os.path.join(dir_prep, selected_folder, st.session_state.current_image)
        
        img1_cv, img1_rgb = load_image(img1_path)
        img2_cv, img2_rgb = load_image(img2_path)
        
        # Mostrar imágenes
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Imagen Original")
            st.image(img1_rgb, use_column_width=True)
        
        with col2:
            st.subheader("Imagen Preprocesada")
            st.image(img2_rgb, use_column_width=True)
        
        # Mostrar histogramas
        st.subheader("Comparación de Histogramas RGB")
        hist_fig = plot_histograms(img1_cv, img2_cv)
        st.pyplot(hist_fig)
        
        # Cálculo y visualización de diferencias
        mse, psnr, diff_norm = calculate_image_differences(img1_cv, img2_cv)
        
        # Métricas
        st.subheader("Métricas de Diferencia")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MSE", f"{mse:.2f}")
        with col2:
            st.metric("PSNR", f"{psnr:.2f} dB")
        with col3:
            st.image(diff_norm, caption="Diferencia Estructural", use_column_width=True)
        
        # Información adicional
        st.info(f"Imagen actual: {st.session_state.current_image}")

if __name__ == "__main__":
    main()