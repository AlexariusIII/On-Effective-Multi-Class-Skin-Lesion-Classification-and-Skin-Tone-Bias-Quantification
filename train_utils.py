"""
Utility functions for training and evaluation of skin lesion classification models.
Includes data path resolution and model/loss initialization helpers.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import RandAugment, TrivialAugmentWide
from dataset import SkinLesionDataset
from model import Timm_Classification_Model, Single_Gated_Model, Double_Gated_Model
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import wandb
import yaml
from pathlib import Path
import os
from timm.data.mixup import Mixup
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import random

def get_data_paths(task, img_type, oversample, subset=False):
    """
    Determines and returns training and validation CSV paths based on arguments.
    Args:
        task (str): Task name.
        img_type (str): Image type.
        oversample (str): Oversampling method.
        subset (bool): Whether to use a subset of the data.
    Returns:
        train_path (str), val_path (str): Paths to training and validation CSVs.
    """
    base_data_dir = "data"
    if img_type == "all":
        if subset:
            train_path = f"{base_data_dir}/{task}/subset/train.csv"
            val_path = f"{base_data_dir}/{task}/subset/test.csv"
        else:
            train_path = f"{base_data_dir}/{task}/train.csv"
            val_path = f"{base_data_dir}/{task}/test.csv"
    else:
        if oversample == "simple":
            print('Using simple oversampling')
            train_path = f"{base_data_dir}/{task}/oversampled/train_{img_type}.csv"
        elif oversample == "synthetic":
            print('Using synthetic oversampling')
            train_path = f"{base_data_dir}/{task}/synthetic_oversampled/train_{img_type}.csv"
        else:
            train_path = f"{base_data_dir}/{task}/subset/train{img_type}.csv"
            train_path = f"{base_data_dir}/{task}/train_{img_type}.csv"
        if subset:
            val_path = f"{base_data_dir}/{task}/subset/test_{img_type}.csv"
        else:
            val_path = f"{base_data_dir}/{task}/test_{img_type}.csv"
    for path in [train_path, val_path]:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Data file not found: {path}. Please check your data directory and arguments.")
    return train_path, val_path

def initialize_model_and_loss(modality_aware, skin_tone_aware, model_arch, task, class_weight_flag, train_dataset, device, embed_dim=64):
    """
    Initializes the model and criterion based on arguments.
    Args:
        modality_aware (bool): If True, use modality-aware model.
        skin_tone_aware (bool): If True, use skin tone-aware model.
        model_arch (str): Model architecture.
        task (str): Task name.
        class_weight_flag (bool): If True, use class weights in loss.
        train_dataset: Dataset object for training.
        device: torch.device.
        embed_dim (int): Embedding dimension for gated models.
    Returns:
        model (nn.Module), criterion (nn.Module)
    """
    if modality_aware:
        if skin_tone_aware:
            model = Double_Gated_Model(num_classes=train_dataset.num_classes, model_arch=model_arch, embed_dim=embed_dim).to(device)
        else:
            model = Single_Gated_Model(num_classes=train_dataset.num_classes, model_arch=model_arch, modality_aware=modality_aware, embed_dim=embed_dim).to(device)
    elif skin_tone_aware:
        model = Single_Gated_Model(num_classes=train_dataset.num_classes, model_arch=model_arch, skin_tone_aware=skin_tone_aware, embed_dim=embed_dim).to(device)
    else:
        model = Timm_Classification_Model(num_classes=train_dataset.num_classes, model_arch=model_arch).to(device)
    criterion = nn.CrossEntropyLoss()
    if class_weight_flag:
        y = train_dataset.df[task].to_list()
        y = [train_dataset.label_map[x] for x in y]
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y),
            y=y
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        print(f"Using class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    return model, criterion