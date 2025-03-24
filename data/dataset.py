import os
import pandas as pd

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
    
    