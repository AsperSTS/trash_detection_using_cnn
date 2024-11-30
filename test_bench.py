import numpy as np
import tensorflow as tf
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional
import logging
from contextlib import redirect_stdout
import os
import gc
import csv
import gspread
from google.oauth2.service_account import Credentials
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization #type: ignore
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
    train_dir: str = "step3_dataset_normalized_24k_config12"
    classes: np.ndarray = np.array(["biologico", "desechos", "metal", "papel", "plasticoYtextil", "vidrio"])
    batch_size: int = 16
    image_size: Tuple[int, int] = (128, 128)
    validation_split: float = 0.30
    epochs: int = 35
    learning_rate: float = 0.0001
    preprocess_config: int = 12
    metrics_dir: str = "mk1_metrics"
    seed: int = 331
    experiment_name: str = "waste_classification"

class GPUManager:
    @staticmethod
    def setup_gpu(memory_limit: int = 3800) -> None:
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
        model = Sequential([
            # Enhanced model architecture with residual connections
            Conv2D(32, (3, 3), activation='relu', padding='same', 
                   input_shape=(*config.image_size, 3)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.2),
            
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.3),
            
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.4),
            
            Flatten(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(len(config.classes), activation='softmax')
        ])
        
        optimizer = Adam(
            learning_rate=config.learning_rate,
            amsgrad=True
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
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001),
            ModelCheckpoint(
                run_dir / f"best_model_{execution_num}.keras",
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
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

    def save_model_summary(self, model: Sequential) -> None:
        """Save model architecture summary"""
        summary_path = self.run_dir / f'model_summary_{self.execution_num}.txt'
        with open(summary_path, 'w') as f:
            with redirect_stdout(f):
                model.summary()

    def plot_training_history(self, history) -> None:
        """Generate and save training plots"""
        metrics = ['accuracy', 'loss', 'auc', 'precision', 'recall']
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
        plt.savefig(self.run_dir / f'training_metrics_{self.execution_num}.png')
        plt.close()

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
            
            # Save results and visualizations
            logger.info("Saving results and generating visualizations...")
            self.result_manager.plot_training_history(history)
            
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