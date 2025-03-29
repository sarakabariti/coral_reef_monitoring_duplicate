import os
import pandas as pd
from PIL import Image

class CoralConditionDataset:
    """A class to handle the coral condiition dataset."""
    def __init__(self, dataset_path: str ="data/coral_condition/"):
        self.IMAGES_PATH = os.path.join(dataset_path, "images")
        self.ANNOTATIONS_PATH = pd.read_csv(os.path.join(dataset_path, "annotations.csv"))
        self.LABELSET_PATH = pd.read_csv(os.path.join(dataset_path, "labelset.csv"))
        
    def get_image(self, patch_id):
        img_path = os.path.join(self.IMAGES_PATH, f"{patch_id}.jpg")  
        return Image.open(img_path)
    
    def get_label(self, patch_id):
        return self.ANNOTATIONS_PATH[self.ANNOTATIONS_PATH["patchid"] == patch_id]["label"].values[0]