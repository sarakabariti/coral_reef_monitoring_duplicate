import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        self.class_colormap = self._generate_class_colormap()

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
    
        
    def _generate_class_colormap(self) -> dict:
        """Generates a consistent color mapping for all classes"""
        # Fixed colors for some major classes
        colormap = {
            'crustose_coralline_algae': [255, 50, 50], # Red
            'turf': [0, 255, 0], # Green
            'sand': [255, 255, 0], # Yellow
            'porites': [0, 0, 255], # Blue
            'macroalgae': [128, 0, 128], # Purple
            'pocillopora': [0, 255, 255], # Cyan
            'broken_coral_rubble': [255, 92, 0] # Orange
        }

        # Get all unique classes from annotations
        annotations = self.load_annotations()
        unique_classes = sorted(annotations['Label'].unique())

        # For remaining classes, generate colors
        used_colors = set(tuple(v) for v in colormap.values())
        for c in unique_classes:
            if c not in colormap:
                # Generate a unique color
                color = tuple(np.random.randint(0, 256, 3))
                while color in used_colors:
                    color = tuple(np.random.randint(0, 256, 3))
                colormap[c] = color
                used_colors.add(color)
        
        return colormap
    
    def visualize_annotations(self, image_name: str):
        """ Visualizes original image with overlaid annotation points and a legend for class colors"""
        try:
            # Load image & convert to array
            img = np.array(self.get_image(image_name))
            h, w = img.shape[:2]
            
            # Load annotations and filter for image
            annotations_df = self.load_annotations()
            img_ann = annotations_df[annotations_df['Name'] == image_name]
            
            if img_ann.empty:
                print(f"No annotations found for image: {image_name}")
                return
            
            # Get color mapping
            class_colormap = self.class_colormap

            # Create figure 
            fig = plt.figure(figsize=(18, 8))
            gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.3])

            # Original image
            ax1 = fig.add_subplot(gs[0])
            ax1.imshow(img)
            ax1.set_title(f'Original Image\n{image_name}\n{w}x{h}px', pad=20)
            ax1.axis('off')
            
            # Annotation overlay
            ax2 = fig.add_subplot(gs[1])
            ax2.imshow(img) 
            
            # Group annotations by class 
            class_groups = img_ann.groupby('Label')
            
            # Plot each class with consistent color 
            for cls, group in class_groups:
                color = np.array(class_colormap[cls])/255  # Convert to 0-1 range
                ax2.scatter(
                    x=group['Column'],
                    y=group['Row'],
                    color=[color],
                    s=100,
                    label=cls,
                    edgecolors='black',
                    linewidths=0.8, 
                    alpha=0.9
                )
            
            ax2.set_title(f'Annotation Points\n{len(img_ann)} total', pad=20)
            ax2.axis('off')
            
            # Legend
            legend_elements = [
                plt.Line2D(
                    [0], [0],
                    marker='o',
                    color='w',
                    markerfacecolor=np.array(class_colormap[cls])/255,
                    markersize=12,
                    markeredgecolor='black',
                    label=f"{cls} ({len(group)})"
                )
                for cls, group in class_groups
            ]

            # Legend subplot
            ax3 = fig.add_subplot(gs[2])
            ax3.axis('off')
            ax3.legend(
                handles=legend_elements,
                loc='center',
                title="Classes Present",
                frameon=True,
                framealpha=0.9,
                edgecolor='black'
            )
            
            plt.tight_layout()
            plt.show()

            # Print annotation statistics
            print(f"\nAnnotation Statistics for {image_name}:")
            print(f"Total points: {len(img_ann)}")
            print(f"Classes present: {len(class_groups)}")
            print("Points per class (sorted by count):")
            print(class_groups.size().sort_values(ascending=False))
            
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")