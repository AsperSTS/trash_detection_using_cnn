import tensorflow as tf
from keras import layers, models, callbacks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import cv2
import os
from tqdm import tqdm
import pandas as pd

# Configuración inicial
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
NUM_CLASSES = 12  # papel, metal, plástico, vidrio, orgánico, otros

class WasteClassifier:
    def __init__(self):
        self.class_names = ['battery', 'white-glass', 'clothes', 'trash', 'green-glass', 'plastic', 'metal', 'brown-glass', 'cardboard', 'biological', 'paper', 'shoes']

        self.img_size = IMG_SIZE
        self.model = None
        
    def load_dataset(self, base_path):
        """Carga y prepara el dataset de imágenes de residuos."""
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(tqdm(self.class_names)):
            class_path = os.path.join(base_path, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)
                        labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    
        return np.array(images), np.array(labels)
    
    def preprocess_image(self, image, apply_preprocessing=True):
        """Aplica técnicas de preprocesamiento a las imágenes."""
        if apply_preprocessing:
            # Redimensionar
            image = cv2.resize(image, (self.img_size, self.img_size))
            
            # Normalización
            image = image / 255.0
            
            # Aumentación de datos
            if np.random.random() > 0.5:
                image = tf.image.random_flip_left_right(image)
            
            # Ajuste de brillo y contraste
            image = tf.image.random_brightness(image, 0.2)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            
            # Rotación aleatoria
            image = tf.image.rot90(image, k=np.random.randint(4))
            
            return image
        else:
            # Solo redimensionar para la versión sin preprocesamiento
            return cv2.resize(image, (self.img_size, self.img_size))
    
    def create_model(self):
        """Crea el modelo CNN con técnicas anti-sobreajuste."""
        model = models.Sequential([
            # Primera capa convolucional
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Segunda capa convolucional
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Tercera capa convolucional
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Cuarta capa convolucional
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Capas densas
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ])
        
        return model
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test, apply_preprocessing=True):
        """Entrena y evalúa el modelo."""
        # Preprocesar datos
        print("Preprocesando imágenes...")
        X_train_proc = np.array([self.preprocess_image(img, apply_preprocessing) 
                                for img in tqdm(X_train)])
        X_test_proc = np.array([self.preprocess_image(img, apply_preprocessing) 
                               for img in tqdm(X_test)])
        
        # Crear y compilar modelo
        self.model = self.create_model()
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks para mejor entrenamiento
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        # Entrenar modelo
        history = self.model.fit(
            X_train_proc, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Evaluar modelo
        test_loss, test_acc = self.model.evaluate(X_test_proc, y_test)
        y_pred = self.model.predict(X_test_proc)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        return history, test_acc, y_pred_classes
    
    def analyze_results(self, history, y_test, y_pred, title):
        """Analiza y visualiza los resultados del modelo."""
        # Métricas de entrenamiento
        plt.figure(figsize=(15, 5))
        
        # Gráfico de precisión
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Entrenamiento')
        plt.plot(history.history['val_accuracy'], label='Validación')
        plt.title(f'Precisión - {title}')
        plt.xlabel('Época')
        plt.ylabel('Precisión')
        plt.legend()
        
        # Gráfico de pérdida
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Entrenamiento')
        plt.plot(history.history['val_loss'], label='Validación')
        plt.title(f'Pérdida - {title}')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        
        # Matriz de confusión
        plt.subplot(1, 3, 3)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(f'Matriz de Confusión - {title}')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Reporte de clasificación
        print(f"\nReporte de Clasificación - {title}")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Análisis de errores comunes
        errors = y_test != y_pred
        error_distribution = pd.DataFrame({
            'Clase Real': [self.class_names[i] for i in y_test[errors]],
            'Predicción': [self.class_names[i] for i in y_pred[errors]]
        }).value_counts()
        
        print("\nErrores más comunes:")
        print(error_distribution.head())

def main():
    # Crear instancia del clasificador
    classifier = WasteClassifier()
    
    # Cargar dataset
    print("Cargando dataset...")
    X, y = classifier.load_dataset('garbage_classification')
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Entrenar y evaluar modelo con preprocesamiento
    print("\nEntrenando modelo con preprocesamiento...")
    history_prep, acc_prep, y_pred_prep = classifier.train_and_evaluate(
        X_train, y_train, X_test, y_test, apply_preprocessing=True
    )
    
    # Entrenar y evaluar modelo sin preprocesamiento
    print("\nEntrenando modelo sin preprocesamiento...")
    history_no_prep, acc_no_prep, y_pred_no_prep = classifier.train_and_evaluate(
        X_train, y_train, X_test, y_test, apply_preprocessing=False
    )
    
    # Analizar resultados
    classifier.analyze_results(
        history_prep, y_test, y_pred_prep, "Con Preprocesamiento"
    )
    classifier.analyze_results(
        history_no_prep, y_test, y_pred_no_prep, "Sin Preprocesamiento"
    )
    
    # Comparar resultados finales
    print("\nComparación de Precisión Final:")
    print(f"Con preprocesamiento: {acc_prep:.4f}")
    print(f"Sin preprocesamiento: {acc_no_prep:.4f}")

if __name__ == "__main__":
    main()