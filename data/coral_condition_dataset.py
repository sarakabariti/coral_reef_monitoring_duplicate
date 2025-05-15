import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image

class CoralConditionDataset:
    """A class to handle the coral condiition dataset."""
    def __init__(self, dataset_path: str ="data/coral_condition/"):
        self.IMAGES_PATH = os.path.join(dataset_path, "images")
        self.ANNOTATIONS_PATH = pd.read_csv(os.path.join(dataset_path, "annotations.csv"))
        self.LABELSET_PATH = pd.read_csv(os.path.join(dataset_path, "labelset.csv"))
        self.METADATA_PATH = pd.read_csv(os.path.join(dataset_path, "surveys_metadata.csv"))
        
    def get_image(self, patch_id):
        img_path = os.path.join(self.IMAGES_PATH, f"{patch_id}.jpg")  
        return Image.open(img_path)
    
    def get_label(self, patch_id):
        return self.ANNOTATIONS_PATH[self.ANNOTATIONS_PATH["patchid"] == patch_id]["label"].values[0]
    
    def get_preprocessed_annotations(self):
        """Converts labels to lists and cache binary matrix for NPMI calculations"""
        # Create a clean copy (don't modify original)
        annotations = self.ANNOTATIONS_PATH.copy()
        # Add label_list column if not present
        if 'label_list' not in annotations.columns:
            annotations['label_list'] = annotations['label'].apply(
                lambda x: [int(x)] if isinstance(x, int) else [int(l) for l in str(x).split(',')]
            )
        self.mlb = MultiLabelBinarizer()
        self.binary_matrix = pd.DataFrame(
            self.mlb.fit_transform(annotations['label_list']),
            columns=self.mlb.classes_,
            index=annotations.index
        )
        return annotations
    
    def multi_label_npmi(self, context_labels: list, target_label: str):
        """Calculate Normalized Pointwise Mutual Information (NPMI) between a set of context labels and a target label."""
        # Convert label names to IDs
        context_ids = [self.LABELSET_PATH[self.LABELSET_PATH['label_name'] == l]['label'].values[0] 
                      for l in context_labels]
        target_id = self.LABELSET_PATH[self.LABELSET_PATH['label_name'] == target_label]['label'].values[0]
        
        # P(context)
        context_mask = np.all(self.binary_matrix[context_ids] == 1, axis=1)
        p_context = context_mask.mean()
        
        # P(target)
        p_target = (self.binary_matrix[target_id] == 1).mean()
        
        # P(context AND target)
        p_joint = (context_mask & (self.binary_matrix[target_id] == 1)).mean()
        
        # NPMI calculation
        if p_joint > 0:
            npmi = (np.log(p_joint) - np.log(p_context * p_target)) / -np.log(p_joint)
        else:
            npmi = 0
            
        return npmi