import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization #type: ignore
from keras.regularizers import l2 #type: ignore
import matplotlib.pyplot as plt
from keras.optimizers import Adam #type: ignore
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from contextlib import redirect_stdout
import os
import gc
import csv
# 148 seed: run 9

# Configuración GPU (sin cambios)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    except RuntimeError as e:
        print(e)

# Directorio y clases
# train_dir = "step3_dataset_normalized/"

train_dir = "step3_dataset_normalized_noprep"
classes = np.array(["biologico", "desechos", "metal", "papel", "plasticoYtextil", "vidrio"])

# Parametros informe csv
preprocess_config = 2

# Parámetros optimizados
batch_size = 24
image_size = (160, 160)
validation_split = 0.25
epochs = 20  # Aumentado para ver mejor la evolución
learning_rate = 0.0001  # Reducido para un aprendizaje más estable
seed = tf.random.uniform(shape=[], minval=0, maxval=1000, dtype=tf.int64).numpy() # 407 before



# seed = 
print(f"Semilla utilizada: {seed}") 


# Función para preprocesamiento de imágenes
def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalización
    return image, label

# Función para crear datasets
def create_dataset(directory, subset):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=validation_split,
        subset=subset,
        seed=seed, # Before: 123
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True
    )
    # Aplicar preprocesamiento, cache, repeat y prefetch
    return dataset.map(preprocess_image).cache().repeat().prefetch(buffer_size=tf.data.AUTOTUNE)

# Crear datasets
print("Cargando datos...")
train_dataset = create_dataset(train_dir, "training")
validation_dataset = create_dataset(train_dir, "validation")

# Calcular steps_per_epoch
total_images = sum(len(os.listdir(os.path.join(train_dir, clase))) for clase in classes)
train_images = int(total_images * (1 - validation_split))
val_images = int(total_images * validation_split)

steps_per_epoch = train_images // batch_size
validation_steps = val_images // batch_size

print(f"Total images: {total_images}")
print(f"Training images: {train_images}")
print(f"Validation images: {val_images}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

# Definir el modelo con arquitectura mejorada
model = Sequential([
    # Primera capa convolucional
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(image_size[0], image_size[1], 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Segunda capa convolucional
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Tercera capa convolucional
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Capas densas
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),   
    BatchNormalization(),

    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])
# model = Sequential([
#     # Primera capa convolucional
#     Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001), input_shape=(image_size[0], image_size[1], 3)),
#     BatchNormalization(),
#     Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
#     BatchNormalization(),
#     MaxPooling2D(2, 2),
#     Dropout(0.3),
    
#     # Segunda capa convolucional
#     Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
#     BatchNormalization(),
#     Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
#     BatchNormalization(),
#     MaxPooling2D(2, 2),
#     Dropout(0.3),
    
#     # Capas densas
#     Flatten(),
#     Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
#     BatchNormalization(),
#     Dropout(0.6),
#     Dense(64, activation='relu', kernel_regularizer=l2(0.001)),   
#     BatchNormalization(),
#     Dropout(0.6),
#     Dense(len(classes), activation='softmax')
# ])

# Compilar el modelo
optimizer = Adam(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', 'AUC']
)

# Crear directorio para métricas
metrics_dir = "mk1_metrics"
os.makedirs(metrics_dir, exist_ok=True)


# Identificador de ejecución basado en la cantidad de carpetas dentro de metrics_dir
ejecucion = len([d for d in os.listdir(metrics_dir) if os.path.isdir(os.path.join(metrics_dir, d))]) + 1
run_dir = os.path.join(metrics_dir, f"run_{ejecucion}")
os.makedirs(run_dir, exist_ok=True)  # Crear directorio para esta ejecución


with open(os.path.join(run_dir,f'resumen_modelo_{ejecucion}.txt'), 'w') as f:
  with redirect_stdout(f):
    model.summary()
    
# Callback para reducir el learning rate
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001
)

# Early stopping para evitar overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# # Entrenar el modelo
# history = model.fit(
#     train_dataset,
#     epochs=epochs,
#     validation_data=validation_dataset,
#     steps_per_epoch=steps_per_epoch,
#     validation_steps=validation_steps,
#     callbacks=[reduce_lr, early_stopping],
#     verbose=1
# )

try:
    # Entrenar el modelo
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[reduce_lr, early_stopping],
        verbose=1
    )
    
    # Evaluación en el dataset de validación
    print("\nEvaluando modelo...")
    val_loss, val_accuracy, val_auc = model.evaluate(validation_dataset, steps=validation_steps)
    print(f'Precisión en validación: {val_accuracy:.4f}')
    print(f'Pérdida en validación: {val_loss:.4f}')
    print(f'AUC en validación: {val_auc:.4f}')
    
    

    # Guardar el modelo en el directorio específico
    model_path = os.path.join(run_dir, f"modelo_clasificacion_basura_{ejecucion}.keras")
    model.save(model_path)

    # Guardar métricas gráficas
    print("Generando gráficas...")

    # Gráfica de precisión
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Precisión (entrenamiento)')
    plt.plot(history.history['val_accuracy'], label='Precisión (validación)')
    plt.title('Evolución de la Precisión')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()

    # Gráfica de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Pérdida (entrenamiento)')
    plt.plot(history.history['val_loss'], label='Pérdida (validación)')
    plt.title('Evolución de la Pérdida')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()

    # Guardar gráfica
    metricas_path = os.path.join(run_dir, f'metricas_entrenamiento_{ejecucion}.png')
    plt.tight_layout()
    plt.savefig(metricas_path)
    plt.close()

    # Generar y guardar matriz de confusión
    # print("Generando matriz de confusión...")
    # Y_pred = []
    # Y_true = []

    # for batch in validation_dataset:
    #     x, y = batch
    #     predictions = model.predict(x, verbose=0)
    #     Y_pred.extend(np.argmax(predictions, axis=1))
    #     Y_true.extend(np.argmax(y, axis=1))
        # gc.collect()  # Limpiar memoria después de cada lote



    # Resultados CSV
    results_csv = os.path.join(metrics_dir, 'resultados.csv')
    results_fieldnames = ['ejecucion', 'val_accuracy', 'val_loss', 'val_auc']
    file_exists = os.path.isfile(results_csv)

    with open(results_csv, mode='a', newline='') as f_results:
        writer = csv.DictWriter(f_results, fieldnames=results_fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'ejecucion': ejecucion,
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
            'val_auc': val_auc
        })

    # Configuración CSV
    config_csv = os.path.join(metrics_dir, 'configuracion.csv')
    config_fieldnames = [
        'ejecucion', 'batch_size', 'image_size', 'epochs', 
        'learning_rate', 'optimizer', 'seed', 'validation_split','preprocess_configuration'
    ]
    file_exists = os.path.isfile(config_csv)

    with open(config_csv, mode='a', newline='') as f_config:
        writer = csv.DictWriter(f_config, fieldnames=config_fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'ejecucion': ejecucion,
            'batch_size': batch_size,
            'image_size': image_size,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'optimizer': 'Adam (amsgrad=True)',
            'seed': seed,
            'validation_split': validation_split,
            'preprocess_configuration': preprocess_config
        })
    
     # Obtener predicciones para todas las imágenes de validación en una sola llamada
    print("Generando predicciones para la matriz de confusión...")
    validation_dataset = validation_dataset.prefetch(buffer_size=32)  # Asegurar prefetch

    # # Predicciones para todo el conjunto de validación
    # Y_pred_probs = model.predict(validation_dataset, verbose=1)
    # Y_pred = np.argmax(Y_pred_probs, axis=1)

    # # Etiquetas verdaderas (se deben extraer del dataset)
    # Y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in validation_dataset])


    # plt.figure(figsize=(10, 8))
    # cm = confusion_matrix(Y_true, Y_pred)
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    #             xticklabels=classes,
    #             yticklabels=classes)
    # plt.title('Matriz de Confusión')
    # plt.xlabel('Predicción')
    # plt.ylabel('Valor Real')
    # plt.xticks(rotation=45)
    # plt.yticks(rotation=45)
    # plt.tight_layout()
    # confusion_matrix_path = os.path.join(run_dir, 'matriz_confusion.png')
    # plt.savefig(confusion_matrix_path)
    # plt.close()

    # # Guardar reporte de clasificación
    # report = classification_report(Y_true, Y_pred, target_names=classes, digits=4)
    # report_path = os.path.join(run_dir, 'reporte_clasificacion.txt')
    # with open(report_path, 'w') as f:
    #     f.write("Métricas de Evaluación del Modelo\n")
    #     f.write("================================\n\n")
    #     f.write(f"Precisión en validación: {val_accuracy:.4f}\n")
    #     f.write(f"Pérdida en validación: {val_loss:.4f}\n\n")
    #     f.write("Reporte de Clasificación:\n")
    #     f.write("------------------------\n")
    #     f.write(report)

except Exception as e:
    print(f"Error durante el entrenamiento: {str(e)}")
finally:
    # Limpiar memoria
    gc.collect()
    tf.keras.backend.clear_session()
    