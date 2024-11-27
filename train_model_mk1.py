import csv
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from keras.optimizers import Adam # type: ignore
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import gc  # Para limpieza de memoria
import os  # Para operaciones de sistema

# Configurar el uso de memoria de GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Opcionalmente, limitar la memoria GPU
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])  # Limitar a 4GB
    except RuntimeError as e:
        print(e)

# Configuración de memoria
tf.config.set_soft_device_placement(True)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# Directorio para guardar las métricas
metrics_dir = "mk1_metrics"
os.makedirs(metrics_dir, exist_ok=True)  # Crear la carpeta si no existe

# Directorio y clases
train_dir = "step3_dataset_normalized/"
classes = np.array(["biologico", "desechos", "metal", "papel", "plasticoYtextil", "vidrio"])

# Parámetros optimizados
batch_size = 10  # Reducido para menor consumo de memoria
image_size = (160, 160)  # Reducido el tamaño de imagen
validation_split = 0.2
epochs = 10
_learning_rate = 0.001

# Función para crear datasets
def create_dataset(directory, subset):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=validation_split,
        subset=subset,
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

# Crear datasets
print("Cargando datos...")
train_dataset = create_dataset(train_dir, "training")
validation_dataset = create_dataset(train_dir, "validation")

# Configurar el pipeline de datos
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Modelo CNN más ligero
# model = Sequential([
#     tf.keras.layers.Rescaling(1./255),
#     Conv2D(16, (5, 5), activation='relu', input_shape=(*image_size, 3)),
#     MaxPooling2D((2, 2)),
#     Conv2D(32, (5, 5), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (5, 5), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(64, activation='relu'),
#     Dropout(0.2),
#     Dense(len(classes), activation='softmax')
# ])

# model = Sequential([
#     tf.keras.layers.Rescaling(1./255),
#     Conv2D(16, (3, 3), activation='relu', input_shape=(*image_size, 3)),  # Reducir tamaño del kernel
#     MaxPooling2D((2, 2)),
    
#     Conv2D(32, (3, 3), activation='relu'),  # Reducir tamaño del kernel
#     MaxPooling2D((2, 2)),
    
#     Flatten(),
#     Dense(32, activation='relu'),  # Reducir unidades
#     Dropout(0.2),
#     Dense(len(classes), activation='softmax')
# ])

model = Sequential([
    tf.keras.layers.Rescaling(1./255),
    Conv2D(16, (5, 5), activation='relu', input_shape=(*image_size, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),  # Añadir dropout después de las capas convolucionales
    
    Conv2D(32, (5, 5), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),  # Añadir más dropout
    
    Conv2D(64, (5, 5), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),  # Añadir más dropout
    
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),   # Aumentar el dropout aquí
    Dense(len(classes), activation='softmax')
])
# Compilar
model.compile(
    optimizer=Adam(
        learning_rate=_learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True    # Activar amsgrad puede ayudar con el sobreajuste
    ),
    # optimizer=Adam(learning_rate=_learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'AUC']
)

# Callback para early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Callback para reducir el learning rate
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    # monitor='val_loss',
    # factor=0.2,
    # patience=2,
    # min_lr=0.00001
    monitor='val_loss',
    factor=0.5,           # Cambio de 0.2 a 0.5 para una reducción más gradual
    patience=3,           # Aumentado de 2 a 3
    min_lr=1e-6,         # Reducido para permitir más ajuste fino
    cooldown=1,          # Añadido período de enfriamiento
    verbose=1 
)

print("Iniciando entrenamiento...")

try:
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Guardar el modelo
    print("Guardando modelo...")
    model.save(os.path.join(metrics_dir, "modelo_clasificacion_basura.keras"))
    
    # Limpiar memoria
    gc.collect()
    
    # Guardar métricas de entrenamiento y validación
    print("Generando gráficas...")
    plt.figure(figsize=(12, 4))
    
    # Gráfica de precisión
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
    
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'metricas_entrenamiento.png'))
    plt.close()
    
    # Evaluación en el dataset de validación
    print("\nEvaluando modelo...")
    val_loss, val_accuracy, val_auc = model.evaluate(validation_dataset)
    print(f'Precisión en validación: {val_accuracy:.4f}')
    print(f'Pérdida en validación: {val_loss:.4f}')
    print(f'AUC en validación: {val_auc:.4f}')

    # Directorios de los archivos
    results_csv = os.path.join(metrics_dir, 'resultados.csv')
    config_csv = os.path.join(metrics_dir, 'configuracion.csv')

    # Campos para resultados
    results_fieldnames = ['ejecucion', 'val_accuracy', 'val_loss', 'val_auc']

    # Campos para configuración
    config_fieldnames = [
        'ejecucion', 'batch_size', 'image_size', 'epochs', 
        'learning_rate', 'optimizer', 'dropout_rates', 'architecture'
    ]

    # Contador de ejecuciones
    ejecucion = len(os.listdir(metrics_dir))  # Basado en la cantidad de archivos generados

    # Guardar resultados
    with open(results_csv, mode='a', newline='') as f_results:
        writer = csv.DictWriter(f_results, fieldnames=results_fieldnames)
        if not os.path.isfile(results_csv):
            writer.writeheader()
        writer.writerow({
            'ejecucion': ejecucion,
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
            'val_auc': val_auc
        })

    # Guardar configuración
    with open(config_csv, mode='a', newline='') as f_config:
        writer = csv.DictWriter(f_config, fieldnames=config_fieldnames)
        if not os.path.isfile(config_csv):
            writer.writeheader()
        writer.writerow({
            'ejecucion': ejecucion,
            'batch_size': batch_size,
            'image_size': image_size,
            'epochs': epochs,
            'learning_rate': _learning_rate,
            'optimizer': 'Adam (amsgrad=True)',
            'dropout_rates': '0.25, 0.5',
            'architecture': '3 Conv2D + MaxPooling + Dropout + Dense'
        })
    
    # Generar y guardar la matriz de confusión
    print("Generando matriz de confusión...")
    Y_pred = []
    Y_true = []
    
    for batch in validation_dataset:
        x, y = batch
        predictions = model.predict(x, verbose=0)
        Y_pred.extend(np.argmax(predictions, axis=1))
        Y_true.extend(np.argmax(y, axis=1))
        gc.collect()  # Limpiar memoria después de cada lote

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(Y_true, Y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'matriz_confusion.png'))
    plt.close()
    
    # Guardar el reporte de clasificación
    report = classification_report(Y_true, Y_pred, target_names=classes, digits=4)
    print("\nReporte de Clasificación:")
    print(report)
    
    with open(os.path.join(metrics_dir, 'reporte_clasificacion.txt'), 'w') as f:
        f.write("Métricas de Evaluación del Modelo\n")
        f.write("================================\n\n")
        f.write(f"Precisión en validación: {val_accuracy:.4f}\n")
        f.write(f"Pérdida en validación: {val_loss:.4f}\n\n")
        f.write("Reporte de Clasificación:\n")
        f.write("------------------------\n")
        f.write(report)

except Exception as e:
    print(f"Error durante el entrenamiento: {str(e)}")
finally:
    # Limpiar memoria
    gc.collect()
    tf.keras.backend.clear_session()
