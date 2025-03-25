import os
import pandas as pd
from PIL import Image

class CoralDataset:
    """
    A class to handle the coral reef dataset.
    """

    def __init__(self, dataset_path: str = "data/dataset/"):
        """
        Initialize the dataset.

        Args:
            dataset_path (str): Path to the dataset folder.
        """
        self.DATASET_DIR_PATH = dataset_path
        self.ANNOTATIONS_PATH = os.path.join(dataset_path, "combined_annotations_remapped.csv")
        self.IMAGES_PATH = os.path.join(dataset_path, "images")

    def load_annotations(self) -> pd.DataFrame:
        """Load and return annotations DataFrame """
        return pd.read_csv(self.ANNOTATIONS_PATH)

    def get_image(self, image_name: str) -> Image.Image:
        """Loads image from dataset (image name can be added with or without extension)"""
        base_path = os.path.join(self.IMAGES_PATH, image_name.split('.')[0])
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
            img_path = f"{base_path}{ext}"
            if os.path.exists(img_path):
                return Image.open(img_path)
        raise FileNotFoundError(f"No image found for {image_name} in {self.IMAGES_PATH}")
