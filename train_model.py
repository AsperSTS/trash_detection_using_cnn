import numpy as np
import tensorflow as tf
import datetime
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization # type: ignore
from keras.optimizers import Adam   # type: ignore
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
# import seaborn as sns
from contextlib import redirect_stdout
import os
import gc
import csv
import gspread
from google.oauth2.service_account import Credentials



SPREADSHEET_NAME = "EjecucionesProcDigitalDeImagenes"
SHEET_NAME_RESULTADOS = "resultados"
SHEET_NAME_CONFIGURACIONES = "configuraciones"

# Configuration class to hold all parameters
class Config:
    def __init__(self):
        self.train_dir = "step3_dataset_normalized_24k_config12"
        # self.train_dir = "step3_dataset_normalized_noprep"
        # self.train_dir = "step3_dataset_normalized_canny70"
        # self.train_dir = "step3_dataset_normalized_config6_9000"
        self.classes = np.array(["biologico", "desechos", "metal", "papel", "plasticoYtextil", "vidrio"])
        self.batch_size = 16
        self.image_size = (128, 128)
        self.validation_split = 0.30
        self.epochs = 15
        self.learning_rate = 0.0001
        self.preprocess_config = 12
        self.metrics_dir = "mk1_metrics"
        self.seed =  331#tf.random.uniform(shape=[], minval=0, maxval=1000, dtype=tf.int64).numpy()
# GPU Configuration
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=3800)])
        except RuntimeError as e:
            print(e)

# Data preprocessing functions
def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def create_dataset(config, subset):
    dataset = tf.keras.utils.image_dataset_from_directory(
        config.train_dir,
        validation_split=config.validation_split,
        subset=subset,
        seed=config.seed,
        image_size=config.image_size,
        batch_size=config.batch_size,
        label_mode='categorical',
        shuffle=True
    )
    return dataset.map(preprocess_image).cache().repeat().prefetch(buffer_size=tf.data.AUTOTUNE)

# Model architecture
def create_model(config):
    
    model = Sequential([
        # First convolutional block
        Conv2D(16, (3, 3), activation='relu', padding='same', 
               input_shape=(config.image_size[0], config.image_size[1], 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.1),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.5),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Dense layers
        Flatten(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(len(config.classes), activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    return model

# Training callbacks
def get_callbacks():
    return [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001
        )
        # ,
        # tf.keras.callbacks.EarlyStopping(
            # monitor='val_loss',
            # patience=5,
            # restore_best_weights=True
        # )
    ]

# Visualization functions
def plot_training_history(history, run_dir, execution_num):
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f'training_metrics_{execution_num}.png'))
    plt.close()
def get_next_execution_number(metrics_dir):
    # Revisa cuántos directorios hay en `metrics_dir`
    if os.path.exists(metrics_dir):
        existing_runs = [d for d in os.listdir(metrics_dir) if os.path.isdir(os.path.join(metrics_dir, d))]
        if existing_runs:
            # Encuentra el máximo número basado en el formato de los nombres de directorio
            max_execution_num = max(
                int(d.split('_')[1]) for d in existing_runs if d.startswith("run_")
            )
            return max_execution_num + 1
    return 1
def save_to_google_sheets(spreadsheet_name, sheet_name, data):
    """Guarda los datos en una hoja de Google Sheets, creando la hoja si no existe.

    Args:
        spreadsheet_name: Nombre de la hoja de cálculo de Google Sheets.
        sheet_name: Nombre de la hoja dentro de la hoja de cálculo.
        data: Diccionario con los datos a guardar.
    """

    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    creds = Credentials.from_service_account_file('trashdetectionusingcnn199502-0e33c8aba5ab.json', scopes=scopes)

    client = gspread.authorize(creds)

    # Abre la hoja de cálculo por nombre (no la crea si no existe)
    spreadsheet = client.open(spreadsheet_name)

    # Intenta abrir la hoja, si no existe, créala con los nombres de los atributos como encabezados
    try:
        worksheet = spreadsheet.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=sheet_name, rows="100", cols="20")
        # Añadir encabezados de atributos
        header = list(data.keys())
        worksheet.insert_row(header, 1)

    # Añadir datos como nueva fila
    new_row = list(data.values())
    worksheet.append_row(new_row)
def save_results(config, metrics_dir, execution_num, val_metrics):
    """Guarda los resultados y la configuración en Google Sheets."""

    save_to_google_sheets(
        SPREADSHEET_NAME, 
        SHEET_NAME_RESULTADOS, 
        {
            'execution': execution_num,
            'val_accuracy': val_metrics[1],
            'val_loss': val_metrics[0],
            'val_auc': val_metrics[2]
        }
    )
    
    save_to_google_sheets(
        SPREADSHEET_NAME,
        SHEET_NAME_CONFIGURACIONES, 
        {
            'execution': execution_num,
            'batch_size': config.batch_size,
            'image_size': f"({config.image_size[0]})x({config.image_size[1]})",
            'epochs': config.epochs,
            'learning_rate': config.learning_rate,
            'optimizer': 'Adam (amsgrad=True)',
            'seed': config.seed,
            'validation_split': config.validation_split,
            'preprocess_configuration': config.preprocess_config,
            'dataset': config.train_dir
        }
    )
# Results saving functions
def save_results_csv(config, metrics_dir, execution_num, val_metrics):
    save_to_csv(
        os.path.join(metrics_dir, 'resultados.csv'),
        ['execution', 'val_accuracy', 'val_loss', 'val_auc'],
        {
            'execution': execution_num,
            'val_accuracy': val_metrics[1],
            'val_loss': val_metrics[0],
            'val_auc': val_metrics[2]
        }
    )
    
    save_to_csv(
        os.path.join(metrics_dir, 'configuracion.csv'),
        ['execution', 'batch_size', 'image_size', 'epochs', 'learning_rate', 
         'optimizer', 'seed', 'validation_split', 'preprocess_configuration', "dataset"],
        {
            'execution': execution_num,
            'batch_size': config.batch_size,
            'image_size': config.image_size,
            'epochs': config.epochs,
            'learning_rate': config.learning_rate,
            'optimizer': 'Adam (amsgrad=True)',
            'seed': config.seed,
            'validation_split': config.validation_split,
            'preprocess_configuration': config.preprocess_config,
            'dataset': config.train_dir
        }
    )

def save_to_csv(filepath, fieldnames, row_data):
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

# Main training function
def train_model():
    try:
        # Initialize configuration
        config = Config()
        setup_gpu()
        
        # Create directories
        os.makedirs(config.metrics_dir, exist_ok=True)
        # execution_num = len([d for d in os.listdir(config.metrics_dir) 
        #                    if os.path.isdir(os.path.join(config.metrics_dir, d))]) + 1
        execution_num = get_next_execution_number(config.metrics_dir)
        run_dir = os.path.join(config.metrics_dir, f"run_{execution_num}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Print seed
        print(f"Seed used: {config.seed}")
        
        # Prepare datasets
        print("Loading data...")
        train_dataset = create_dataset(config, "training")
        validation_dataset = create_dataset(config, "validation")
        
        # Calculate steps
        total_images = sum(len(os.listdir(os.path.join(config.train_dir, clase))) 
                          for clase in config.classes)
        train_images = int(total_images * (1 - config.validation_split))
        val_images = int(total_images * config.validation_split)
        steps_per_epoch = train_images // config.batch_size
        validation_steps = val_images // config.batch_size
        
        # Create and train model
        model = create_model(config)
        with open(os.path.join(run_dir, f'model_summary_{execution_num}.txt'), 'w') as f:
            with redirect_stdout(f):
                model.summary()
        
        log_dir = os.path.join(run_dir, "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        history = model.fit(
            train_dataset,
            epochs=config.epochs,
            validation_data=validation_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            # callbacks=get_callbacks(),
            callbacks=[tensorboard_callback],
            verbose=1
        )
        
        # Evaluate model
        print("\nEvaluating model...")
        val_metrics = model.evaluate(validation_dataset, steps=validation_steps)
        print(f'Validation Accuracy: {val_metrics[1]:.4f}')
        print(f'Validation Loss: {val_metrics[0]:.4f}')
        print(f'Validation AUC: {val_metrics[2]:.4f}')
        
        # Save model
        model.save(os.path.join(run_dir, f"waste_classification_model_{execution_num}.keras"))
        
        # Generate and save visualizations
        print("Generating plots...")
        plot_training_history(history, run_dir, execution_num)
        
        # Save results
        save_results(config, config.metrics_dir, execution_num, val_metrics)
        
    except Exception as e:
        print(f"Training error: {str(e)}")
    finally:
        gc.collect()
        tf.keras.backend.clear_session()

# if __name__ == "__main__":
#     for i in range(0,5):
#         train_model()
if __name__ == "__main__":
    # batch_sizes = [16, 32, 64]
    # learning_rates = [0.0001, 0.0005, 0.001]
    # epochs_list = [20, 25, 30]

    # for i in range(5):  # Ejecutar 5 entrenamientos
    #     config = Config()
    #     config.batch_size = np.random.choice(batch_sizes)
    #     config.learning_rate = np.random.choice(learning_rates)
    #     config.epochs = np.random.choice(epochs_list)
    #     print(f"Executing run {i + 1} with batch_size={config.batch_size}, "
    #           f"learning_rate={config.learning_rate}, epochs={config.epochs}")
    #     train_model(config)


    train_model()