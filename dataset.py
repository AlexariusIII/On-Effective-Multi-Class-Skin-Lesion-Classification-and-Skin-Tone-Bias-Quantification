"""
Dataset definition for skin lesion classification and skin tone bias quantification.
Provides the SkinLesionDataset class for loading and preprocessing image data.
"""
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class SkinLesionDataset(Dataset):
    """
    PyTorch Dataset for skin lesion images and metadata.
    Handles multiple tasks (binary, multiclass, skin_tone) and supports modality/skin tone awareness.
    """
    def __init__(self, csv_path=None, transform=None, task='binary', subset=None, evaluation=False, df=None, modality_aware=False, skin_tone_aware=False):
        """
        Args:
            csv_path (str): Path to the CSV file containing image metadata.
            transform: Optional transform to be applied on images.
            task (str): Task type ('binary', 'multiclass', 'skin_tone').
            subset: Optional, filter by skin tone subset.
            evaluation (bool): If True, returns image IDs for evaluation.
            df (pd.DataFrame): Optional, use provided DataFrame instead of reading from csv_path.
            modality_aware (bool): If True, include modality index.
            skin_tone_aware (bool): If True, include skin tone index.
        """
        if df is not None:
            self.df = df
        else:
            self.df = pd.read_csv(csv_path)
        data_dir = '../data'
        self.image_root_dirs = {
            'ddi': os.path.join(data_dir, 'ddi/images'),
            'fp': os.path.join(data_dir, 'fritzpatrick/images/huggingface_fritzpatrick'),
            'pad': os.path.join(data_dir, 'pad-ufes-20/images'),
            'scin': os.path.join(data_dir, 'scin/images'),
            'mskcc': os.path.join(data_dir, 'MSKCC/images'),
            'argentina': os.path.join(data_dir, 'argentina/images'),
            'isic_24': os.path.join(data_dir, 'isic_24/images'),
            'ham': os.path.join(data_dir, 'ham10000/images'),
            'bcn': os.path.join(data_dir, 'bcn20000/images'),
            'derm12345': os.path.join(data_dir, 'derm12345/images'),
            'derm7pt': os.path.join(data_dir, 'derm7pt/images'),
            'med_node': os.path.join(data_dir, 'med_node/images'),
            'sd_198': os.path.join(data_dir, 'sd-198/images'),
            'synthetic': os.path.join(data_dir, 'synthetic_images'),
        }
        self.transform = transform
        self.subset = subset
        self.evaluation = evaluation
        self.modality_aware = modality_aware
        self.skin_tone_aware = skin_tone_aware
        self.task = task
        # Create label mapping depending on task
        if self.task == 'binary':
            self.label_map = {'benignant': 0, 'malignant': 1}
        elif self.task == 'multiclass':
            self.label_map = {'BCC': 0, 'SCC': 1, 'ACK': 2, 'NEV': 3, 'MEL': 4, 'SEK': 5}
        elif self.task == 'skin_tone':
            self.label_map = {'A': 0, 'B': 1, 'C': 2}
        self.df = self.df[self.df[task].notna()]
        if self.subset:
            self.df = self.df[self.df['skin_tone'] == self.subset]
        self.num_classes = len(self.label_map)
        self.image_type_map = {
            'clinical': 0,
            'dermatoscopic': 1,
            'synthetic': 2
        }
        self.skin_tone_map = {
            'A': 0,
            'B': 1,
            'C': 2
        }

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        Returns image, label, and optionally modality/skin tone indices and image_id depending on flags.
        """
        row = self.df.iloc[idx]
        image_id = row['image_id']
        dataset_name = row['dataset']
        modality_index = self.image_type_map[row['image_type']]
        if self.skin_tone_aware:
            skin_tone_index = self.skin_tone_map[row['combined_skin_tone']]
        label = self.label_map[row[self.task]]
        image_root = self.image_root_dirs[dataset_name]
        image_path = os.path.join(image_root, image_id)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.evaluation:
            if self.modality_aware:
                if self.skin_tone_aware:
                    return image, label, modality_index, skin_tone_index, image_id
                else:
                    return image, label, modality_index, image_id
            elif self.skin_tone_aware:
                return image, label, skin_tone_index, image_id
            else:
                return image, label, image_id
        if self.modality_aware:
            if self.skin_tone_aware:
                return image, label, modality_index, skin_tone_index
            else:
                return image, label, modality_index
        elif self.skin_tone_aware:
            return image, label, skin_tone_index
        else:
            return image, label 
 