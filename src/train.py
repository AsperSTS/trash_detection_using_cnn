import numpy as np
import tensorflow as tf
import datetime
from dataclasses import dataclass, field
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Tuple, List, Optional, Dict
import logging
from contextlib import redirect_stdout
import os
import gc
import csv
import gspread
from google.oauth2.service_account import Credentials
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input #type: ignore
from keras.optimizers import Adam #type: ignore
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping #type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration parameters for the model and training"""
    # train_dir: str = "step3_dataset_normalized_24k_config12"
    train_dir: str = "step3_dataset_normalized"
    classes: np.ndarray = field(default_factory=lambda: np.array(["biologico", "desechos", "metal", "papel", "plasticoYtextil", "vidrio"]))
    batch_size: int = 32
    image_size: Tuple[int, int] = (128, 128)
    validation_split: float = 0.3
    epochs: int = 30
    learning_rate: float = 0.0001 #0.0001 
    preprocess_config: int = 12
    metrics_dir: str = "graficas_modelos_rtx2050_final"
    seed: int = 120 #tf.random.uniform(shape=[], minval=0, maxval=1000, dtype=tf.int64).numpy() #120 
    experiment_name: str = "waste_classification"
    spreadsheet_name = "EjecucionesProcDigitalDeImagenes"
    sheet_name_results = "resultados_rtx2050_final"
    sheet_name_config = "configuraciones_rtx2050_final"

class GPUManager:
    @staticmethod
    def setup_gpu(memory_limit: int = 4096) -> None:
        """Configure GPU settings"""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
                logger.info(f"GPU configured successfully with memory limit: {memory_limit}")
            except RuntimeError as e:
                logger.error(f"GPU configuration failed: {e}")

class DataManager:
    @staticmethod
    def preprocess_image(image, label):
        """Preprocess a single image"""
        return tf.cast(image, tf.float32) / 255.0, label

    @classmethod
    def create_dataset(cls, config: ModelConfig, subset: str) -> tf.data.Dataset:
        """Create and configure a dataset"""
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
        return dataset.map(cls.preprocess_image).cache().repeat().prefetch(buffer_size=tf.data.AUTOTUNE)

    @staticmethod
    def calculate_steps(config: ModelConfig) -> Tuple[int, int]:
        """Calculate training and validation steps"""
        total_images = sum(len(os.listdir(Path(config.train_dir) / clase)) 
                          for clase in config.classes)
        train_images = int(total_images * (1 - config.validation_split))
        val_images = int(total_images * config.validation_split)
        return (
            train_images // config.batch_size,
            val_images // config.batch_size
        )

class ModelBuilder:
    @staticmethod
    def create_model(config: ModelConfig) -> Sequential:
        """Create and compile the model"""
        model = Sequential()
        # Define la capa de entrada
        input_layer = Input(shape=(*config.image_size, 3))

        # Enhanced model architecture with residual connections
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(2, 2)(x)
        x = Dropout(0.2)(x)

        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(2, 2)(x)
        x = Dropout(0.3)(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(2, 2)(x)
        x = Dropout(0.4)(x)

        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(len(config.classes), activation='softmax')(x)


        
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        optimizer = Adam(
            learning_rate=config.learning_rate
            # ,
            # amsgrad=True
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )


    
        return model


class CallbackBuilder:
    @staticmethod
    def get_callbacks(run_dir: Path, execution_num: int) -> List[tf.keras.callbacks.Callback]:
        """Create training callbacks"""
        log_dir = run_dir / f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        return [
            TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=1, min_lr=0.00001)

        ]

class ResultManager:
    def __init__(self, config: ModelConfig):
        
        self.config = config
        self.metrics_dir = Path(config.metrics_dir)
        self.run_dir = None
        self.execution_num = None


    def initialize_run(self) -> None:
        """Initialize directories for a new training run"""
        self.metrics_dir.mkdir(exist_ok=True)
        self.execution_num = self._get_next_execution_number()
        self.run_dir = self.metrics_dir / f"run_{self.execution_num}"
        self.run_dir.mkdir(exist_ok=True)
        logger.info(f"Initialized run directory: {self.run_dir}")

    def _get_next_execution_number(self) -> int:
        """Get the next execution number"""
        if self.metrics_dir.exists():
            existing_runs = [d for d in self.metrics_dir.iterdir() if d.is_dir()]
            if existing_runs:
                return max(int(d.name.split('_')[1]) for d in existing_runs if d.name.startswith("run_")) + 1
        return 1
    def save_to_google_sheets(self, spreadsheet_name: str, sheet_name: str, data: Dict) -> None:
        """Guarda los datos en una hoja de Google Sheets, creando la hoja si no existe.

        Args:
            spreadsheet_name: Nombre de la hoja de cálculo de Google Sheets.
            sheet_name: Nombre de la hoja dentro de la hoja de cálculo.
            data: Diccionario con los datos a guardar.
        """

        try:
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            creds = Credentials.from_service_account_file('your_service_account.json', scopes=scopes)

            client = gspread.authorize(creds)

            # Abre la hoja de cálculo por nombre (no la crea si no existe)
            spreadsheet = client.open(spreadsheet_name)

            # Intenta abrir la hoja, si no existe, créala con los nombres de los atributos como encabezados
            try:
                worksheet = spreadsheet.worksheet(sheet_name)
            except gspread.WorksheetNotFound:
                worksheet = spreadsheet.add_worksheet(title=sheet_name, rows="100", cols="20")
                # Añadir encabezados de atributos
            
            for key, value in data.items():
                if isinstance(value, np.int64):
                    data[key] = int(value)
                elif isinstance(value, np.float64):  # Convertir float64 a float
                    data[key] = float(value)    

            # Añadir datos como nueva fila
            new_row = list(data.values())
            worksheet.append_row(new_row)

        except Exception as e:
            logger.error(f"Error al guardar en Google Sheets: {e}")


    def save_individual_metrics(self, val_metrics: List):
        """Guarda los resultados y la configuración en Google Sheets."""
        
        try:
            self.save_to_google_sheets(
                self.config.spreadsheet_name, 
                self.config.sheet_name_results, 
                {
                    'execution': self.execution_num,
                    'val_accuracy': val_metrics[1],
                    'val_loss': val_metrics[0],
                    'val_auc': val_metrics[2]
                }
            )
            
            self.save_to_google_sheets(
                self.config.spreadsheet_name,
                self.config.sheet_name_config, 
                {
                    'execution': self.execution_num,
                    'batch_size': self.config.batch_size,
                    'image_size': f"({self.config.image_size[0]})x({self.config.image_size[1]})",
                    'epochs': self.config.epochs,
                    'learning_rate': self.config.learning_rate,
                    'optimizer': 'Adam (amsgrad=True)',
                    'seed': self.config.seed,
                    'validation_split': self.config.validation_split,
                    'preprocess_configuration': self.config.preprocess_config,
                    'dataset': self.config.train_dir
                }
            )
        except Exception as e:
            logger.error(f"Error al guardar las métricas individuales: {e}")
            
    def save_model_summary(self, model: Sequential) -> None:
        """Save model architecture summary"""
        summary_path = self.run_dir / f'model_summary_{self.execution_num}.txt'
        with open(summary_path, 'w') as f:
            with redirect_stdout(f):
                model.summary()

    def plot_training_history(self, history) -> None:
        """Generate and save training plots"""
        metrics = ['accuracy', 'loss', 'AUC', 'Precision', 'Recall']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            if metric in history.history:
                axes[idx].plot(history.history[metric], label=f'Training {metric}')
                axes[idx].plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
                axes[idx].set_title(f'{metric.capitalize()} Evolution')
                axes[idx].set_xlabel('Epoch')
                axes[idx].set_ylabel(metric.capitalize())
                axes[idx].legend() 
        plt.tight_layout()
        plt.savefig(self.run_dir / f'training_metrics_{self.execution_num}.png',dpi=300)
        plt.close()
    def plot_confusion_matrix(self, model: Sequential, dataset: tf.data.Dataset, class_names: List[str], steps: int) -> None:
        """Genera y guarda una matriz de confusión."""
        logger.info("Generating confusion matrix...")
        y_true = []
        y_pred = []

        # Recopila etiquetas verdaderas y predicciones
        for batch, labels in dataset.take(steps):
            y_true.extend(tf.argmax(labels, axis=1).numpy())
            predictions = model.predict(batch)
            y_pred.extend(tf.argmax(predictions, axis=1).numpy())

        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)

        # Guardar la matriz de confusión
        plt.title('Confusion Matrix')
        plt.savefig(self.run_dir / f'confusion_matrix_{self.execution_num}.png', dpi=300)
        plt.close()
        logger.info("Confusion matrix saved.")
class Trainer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.result_manager = ResultManager(config)

    def train(self) -> None:
        """Execute the training pipeline"""
        try:
            logger.info("Starting training pipeline...")
            GPUManager.setup_gpu()
            self.result_manager.initialize_run()
            
            logger.info(f"Using seed: {self.config.seed}")
            tf.random.set_seed(self.config.seed)
            
            # Prepare datasets
            logger.info("Loading and preparing datasets...")
            train_dataset = DataManager.create_dataset(self.config, "training")
            validation_dataset = DataManager.create_dataset(self.config, "validation")
            steps_per_epoch, validation_steps = DataManager.calculate_steps(self.config)
            
            # Create and train model
            logger.info("Building model...")
            model = ModelBuilder.create_model(self.config)
            self.result_manager.save_model_summary(model)
            
            callbacks = CallbackBuilder.get_callbacks(self.result_manager.run_dir, self.result_manager.execution_num)
            
            logger.info("Starting training...")
            history = model.fit(
                train_dataset,
                epochs=self.config.epochs,
                validation_data=validation_dataset,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save individual metrics to store them in Google Sheets
            val_metrics = model.evaluate(validation_dataset, steps=validation_steps)
            print(f'Validation Accuracy: {val_metrics[1]:.4f}')
            print(f'Validation Loss: {val_metrics[0]:.4f}')
            print(f'Validation AUC: {val_metrics[2]:.4f}')
            # Save results and visualizations
            
            model.save(os.path.join(self.result_manager.run_dir, f"waste_classification_model_{self.result_manager.execution_num}.keras"))
            
            
            logger.info("Saving results and generating visualizations...")
            self.result_manager.plot_training_history(history)
            self.result_manager.save_individual_metrics(val_metrics)
            
            steps_for_confusion = validation_steps  # Puedes ajustar según el tamaño del dataset de validación
            self.result_manager.plot_confusion_matrix(
                model,
                validation_dataset,
                class_names=self.config.classes.tolist(),
                steps=steps_for_confusion
            )
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
        finally:
            logger.info("Cleaning up...")
            gc.collect()
            tf.keras.backend.clear_session()

def main():
    config = ModelConfig()
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()