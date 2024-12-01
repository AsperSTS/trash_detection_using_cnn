import os
from pathlib import Path
import logging
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import time
from functools import wraps
import shutil
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_pipeline.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class ProcessingConfig:
    """Configuration for image processing pipeline"""
    target_size: Tuple[int, int] = (128, 128)
    target_class_count: int = 4140
    quality: int = 95
    supported_formats: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    enable_processing: bool = True  # New flag to control image processing
    
def timer_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper
class ImageAugmentor:
    """Handles image augmentation for balancing dataset"""
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            shear_range=0.2,
            fill_mode='reflect',
            dtype=np.uint8
        )

    @staticmethod
    def load_image_high_quality(img_path: str) -> np.ndarray:
        """Load image maintaining original quality"""
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def save_image_high_quality(img_path: str, img: np.ndarray, quality: int = 100) -> None:
        """Save image with high quality"""
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])

    @timer_decorator
    def balance_class(self, class_name: str, image_paths: List[str], output_dir: str) -> None:
        """Balance a single class by generating augmented images"""
        class_output_dir = Path(output_dir) / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy original images
        logging.info(f"Copying {len(image_paths)} original images for class {class_name}")
        for i, img_path in enumerate(image_paths):
            dest_path = class_output_dir / f"{class_name}_original_{i}.jpg"
            shutil.copy2(img_path, dest_path)
        
        current_count = len(image_paths)
        if current_count < self.config.target_class_count:
            needed_images = self.config.target_class_count - current_count
            logging.info(f"Generating {needed_images} augmented images for class {class_name}")
            
            for i in range(needed_images):
                try:
                    img_path = np.random.choice(image_paths)
                    img = self.load_image_high_quality(img_path)
                    img_array = np.expand_dims(img, 0)
                    
                    for batch in self.datagen.flow(img_array, batch_size=1):
                        new_img = batch[0].astype('uint8')
                        new_img_path = class_output_dir / f"{class_name}_aug_{i}.jpg"
                        self.save_image_high_quality(str(new_img_path), new_img)
                        break
                except Exception as e:
                    logging.error(f"Error generating augmented image {i} for class {class_name}: {str(e)}")
class ImageProcessor:
    """Handles image preprocessing and normalization"""
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
    @staticmethod
    def _gamma(v: np.ndarray, gamma_val: float) -> np.ndarray:
        """Apply gamma correction"""
        normalized_v = v / 255.0
        corrected_v = np.power(normalized_v, gamma_val)
        return np.uint8(corrected_v * 255)
    
    @staticmethod
    def apply_clahe(v_channel: np.ndarray, clip_limit: float = 1.5, 
                    tile_grid_size: Tuple[int, int] = (4, 4)) -> np.ndarray:
        """Apply CLAHE"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(v_channel)

    # @timer_decorator
    # def preprocess_image(self, img: np.ndarray) -> np.ndarray:
    #     """Preprocess single image"""
    #     if not self.config.enable_processing:
    #         return img
            
    #     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     img_hsv[..., 2] = self.apply_clahe(img_hsv[..., 2], clip_limit=0.6, tile_grid_size=(6, 6))
    #     img_hsv[..., 2] = self._gamma(img_hsv[..., 2], 1.1)
    #     img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    #     return cv2.bilateralFilter(img_bgr, 5, 30, 30)
    @timer_decorator
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess single image with more subtle enhancements that preserve original colors
        
        Args:
            img: Input image in BGR format
        Returns:
            Preprocessed image in BGR format
        """
        if not self.config.enable_processing:
            return img
        
        
        # 2. Procesamiento HSV más conservador
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # CLAHE más sutil en el canal V
        img_hsv[..., 2] = self.apply_clahe(
            img_hsv[..., 2],
            clip_limit=0.8,  # Reducido para evitar sobre-exposición
            tile_grid_size=(4, 4)  # Reducido para evitar artefactos
        )
        
        # Gamma correction más sutil
        img_hsv[..., 2] = self._gamma(img_hsv[..., 2], 1.05)  # Reducido de 1.2 a 1.05
        
        # Volver a BGR
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        
        # 3. Mejora de detalles conservadora
        # Filtro bilateral con parámetros más suaves
        img = cv2.bilateralFilter(
            img,
            d=5,
            sigmaColor=25,  # Reducido de 75 a 25
            sigmaSpace=25   # Reducido de 75 a 25
        )
        
        # 4. Corrección de color y brillo final
        # Convertir a LAB para ajuste final de luminancia
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE final muy sutil solo en luminancia
        clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(4,4))
        l = clahe.apply(l)
        
        # Merge canales
        lab = cv2.merge((l,a,b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return img
    def resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image maintaining aspect ratio"""
        return cv2.resize(img, self.config.target_size, interpolation=cv2.INTER_CUBIC)

    @timer_decorator
    def process_directory(self, input_dir: str, output_dir: str, class_name: str) -> None:
        """Process all images in a directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir) / class_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        processed = 0
        errors = 0
        
        for idx, file_path in enumerate(input_path.glob('*'), 1):
            if file_path.suffix.lower() in self.config.supported_formats:
                try:
                    img = cv2.imread(str(file_path))
                    if img is None:
                        raise ValueError(f"Failed to read image: {file_path}")
                    
                    processed_img = self.preprocess_image(img)
                    resized_img = self.resize_image(processed_img)
                    
                    output_file = output_path / f"{class_name}{idx}.jpg"
                    cv2.imwrite(str(output_file), resized_img, 
                              [cv2.IMWRITE_JPEG_QUALITY, self.config.quality])
                    
                    processed += 1
                    
                except Exception as e:
                    errors += 1
                    logging.error(f"Error processing {file_path}: {str(e)}")
        
        logging.info(f"Directory {input_dir} processing completed: "
                    f"{processed} processed, {errors} errors")

class DatasetPipeline:
    """Main pipeline orchestrator with flexible step execution"""
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.augmentor = ImageAugmentor(config)
        self.processor = ImageProcessor(config)

    def run_augmentation_step(self, input_path: str, intermediate_path: str, 
                            classes: List[str]) -> None:
        """Step 1: Run only the augmentation step"""
        logging.info("Starting dataset balancing...")
        for class_name in classes:
            class_dir = Path(input_path) / class_name
            if not class_dir.is_dir():
                continue
            
            image_paths = list(glob(str(class_dir / "*.jpg")))
            self.augmentor.balance_class(class_name, image_paths, intermediate_path)
        
        logging.info("Augmentation step completed.")

    def run_processing_step(self, intermediate_path: str, output_path: str, 
                          classes: List[str]) -> None:
        """Step 2: Run only the processing and resizing step"""
        logging.info("Starting image preprocessing and resizing...")
        for class_name in classes:
            input_dir = Path(intermediate_path) / class_name
            if not input_dir.is_dir():
                continue
            
            self.processor.process_directory(str(input_dir), output_path, class_name)
        
        logging.info("Processing step completed.")

    def process_dataset(self, input_path: str, intermediate_path: str, 
                       output_path: str, classes: List[str], 
                       run_augmentation: bool = True, 
                       run_processing: bool = True) -> None:
        """Run complete dataset processing pipeline with flexible step execution"""
        if run_augmentation:
            self.run_augmentation_step(input_path, intermediate_path, classes)
        
        if run_processing:
            self.run_processing_step(intermediate_path, output_path, classes)
        
        logging.info("Dataset processing pipeline completed successfully.")

def main():
    # Configuration
    config = ProcessingConfig(
        target_size=(128, 128),
        enable_processing=False  # Set to False to disable processing but keep resizing
    )
    
    # Paths
    input_path = "step1_dataset_joined"
    intermediate_path = "step2_dataset_balanced"
    # output_path = "step3_dataset_normalized_24k_config12"
    output_path = "step3_dataset_normalized_24k_NoPreprocessing2"
    
    # Classes
    classes = [
        "biologico", "desechos", "metal", 
        "papel", "plasticoYtextil", "vidrio"
    ]
    
    # Create pipeline
    pipeline = DatasetPipeline(config)
    
    # Example 1: Run only augmentation step
    # pipeline.run_augmentation_step(input_path, intermediate_path, classes)
    
    # Example 2: Run only processing step
    # pipeline.run_processing_step(intermediate_path, output_path, classes)
    
    # Example 3: Run both steps but disable image processing (only resize)
    # config.enable_processing = False
    # pipeline.process_dataset(input_path, intermediate_path, output_path, classes)
    
    # Example 4: Run complete pipeline with all processing enabled
    pipeline.process_dataset(
        input_path, 
        intermediate_path, 
        output_path, 
        classes,
        run_augmentation=False,
        run_processing=True
    )

if __name__ == "__main__":
    main()