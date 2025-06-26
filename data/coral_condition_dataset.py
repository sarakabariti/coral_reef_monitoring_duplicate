import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from PIL import Image
import random

class CoralConditionDataset:
    """A class to handle the coral condiition dataset."""
    def __init__(self, dataset_path: str ="data/coral_condition/", annotations_file: str = None):
        self.IMAGES_PATH = os.path.join(dataset_path, "images")
        self.ANNOTATIONS_PATH = pd.read_csv(os.path.join(dataset_path, annotations_file)) if annotations_file else pd.read_csv(os.path.join(dataset_path, "annotations.csv"))
        self.LABELSET_PATH = pd.read_csv(os.path.join(dataset_path, "labelset.csv"))
        self.METADATA_PATH = pd.read_csv(os.path.join(dataset_path, "surveys_metadata.csv"))

        # Initialize binary matrix components
        self._binary_matrix = None
        self._mlb = None

    def get_image(self, patch_id):
        """Get image by patch ID."""
        img_path = os.path.join(self.IMAGES_PATH, f"{patch_id}.jpg")  
        return Image.open(img_path)
    
    def get_label(self, patch_id):
        "Get labels by patch ID"
        return self.ANNOTATIONS_PATH[self.ANNOTATIONS_PATH["patchid"] == patch_id]["label"].values[0]
    
    def get_preprocessed_annotations(self):
        """Converts labels to lists of integers & returns annotations with label_list column"""
        # Create clean copy 
        annotations = self.ANNOTATIONS_PATH.copy()
        annotations['label_list'] = annotations['label'].apply(
            lambda x: [int(x)] if isinstance(x, int) else [int(l) for l in str(x).split(',')]
        )
        
        # Initialize binary matrix if not already done
        if self._binary_matrix is None:
            self._initialize_binary_matrix(annotations)
            
        return annotations
    
    def _initialize_binary_matrix(self, annotations):
        """Initialize the binary label matrix"""
        self._mlb = MultiLabelBinarizer()
        self._binary_matrix = pd.DataFrame(
            self._mlb.fit_transform(annotations['label_list']),
            columns=self._mlb.classes_,
            index=annotations.index
        )
    
    @property
    def binary_matrix(self):
        """Accessor for the binary matrix"""
        if self._binary_matrix is None:
            self.get_preprocessed_annotations()
        return self._binary_matrix
    
    @property 
    def mlb(self):
        """Accessor for the MultiLabelBinarizer"""
        if self._mlb is None:
            self.get_preprocessed_annotations()
        return self._mlb
    
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
    

    def split_dataset(self, test_size=0.2, val_size=0.1, random_state=42):
        """Split dataset into train, validation and test sets"""

        # Ensure labels are preprocessed
        annotations = self.get_preprocessed_annotations()

        # For each sample, use its most frequent class as stratification group
        dominant_classes = np.argmax(self.binary_matrix.values, axis=1)
        
        # First split into train+val and test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            annotations['patchid'],
            self.binary_matrix.values,
            test_size=test_size,
            random_state=random_state,
            stratify=dominant_classes  
        )
        
        # Second split into train and val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, 
            y_trainval,
            test_size=val_ratio,
            random_state=random_state,
            stratify=dominant_classes[X_trainval.index]  # Use same stratification
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def calculate_class_weights(self, y_train):
        """Calculate weights for imbalanced classes"""
        class_counts = y_train.sum(axis=0)
        total_samples = len(y_train)
        num_classes = y_train.shape[1]
        
        # Calculate weights that balance classes
        weights = total_samples / (num_classes * class_counts)
        # Convert to dictionary {class index: weight}
        class_weights = {i: weight for i, weight in enumerate(weights)}
        return class_weights

class CoralImageProcessor:
    def __init__(self, images_dir):
        self.images_dir = images_dir
        self.target_size = (512, 512)
        # Normalization parameters (ImageNet mean and std)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def load_image(self, image_id):
        img_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        return Image.open(img_path)
    
    def normalize(self, img):
        """Normalize image to 0-1 range and apply ImageNet stats"""
        img_array = np.array(img) / 255.0
        img_array = (img_array - self.mean) / self.std
        return img_array
    
    def random_augment(self, img):
        """Apply random augmentations"""
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() > 0.5:
            img = img.rotate(random.choice([90, 180, 270]))
        return img
    
    def preprocess(self, image_id, augment=False):
        """Full preprocessing pipeline"""
        img = self.load_image(image_id)
        if augment:
            img = self.random_augment(img)
        return self.normalize(img)