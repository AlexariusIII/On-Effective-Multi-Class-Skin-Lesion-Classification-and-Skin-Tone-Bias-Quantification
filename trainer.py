"""
Trainer class for skin lesion classification and skin tone bias quantification.
Handles training, validation, and metric logging for various model architectures.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import RandAugment, TrivialAugmentWide
from dataset import SkinLesionDataset
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
from timm.loss import SoftTargetCrossEntropy


class Trainer:
    """
    Trainer class for model training and validation.
    Handles training loop, validation, and metric logging.
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, config, classes, modality_aware, skin_tone_aware, wandb=None, mixup_fn=None, eval_mode=False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.classes = classes
        self.mixup_fn = mixup_fn
        self.train_criterion = SoftTargetCrossEntropy() if self.mixup_fn else criterion
        self.wandb = wandb
        self.modality_aware = modality_aware
        self.skin_tone_aware = skin_tone_aware
        self.eval_mode = eval_mode
        self.patience = 10
        self.min_delta = 0.001
        self.counter = 0
        self.best_val_metric = -np.inf

    def plot_confusion_matrix(self, y_true, y_pred, classes):
        """
        Plot and log confusion matrix to wandb.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        if self.wandb:
            self.wandb.log({"confusion_matrix": wandb.Image(plt)})
        plt.close()

    def _train_epoch(self):
        """
        Run one training epoch.
        Returns average loss, accuracy, and F1 score.
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_data in pbar:
            self.optimizer.zero_grad()
            images, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
            if self.modality_aware:
                if self.skin_tone_aware:
                    modality_index = batch_data[2].to(self.device)
                    skin_tone_index = batch_data[3].to(self.device)
                else:
                    modality_index = batch_data[2].to(self.device)
                    skin_tone_index = None
            elif self.skin_tone_aware:
                skin_tone_index = batch_data[2].to(self.device)
                modality_index = None
            else:
                modality_index = None
                skin_tone_index = None
            if self.mixup_fn is not None:
                images, labels = self.mixup_fn(images, labels)
                if modality_index is not None:
                    if skin_tone_index is not None:
                        outputs = self.model(images, modality_index, skin_tone_index)
                    else:
                        outputs = self.model(images, modality_index)
                elif skin_tone_index is not None:
                    outputs = self.model(images, skin_tone_index)
                else:
                    outputs = self.model(images)
                predicted = torch.argmax(outputs.data, dim=1)
                true_labels = torch.argmax(labels.data, dim=1)
            else:
                if modality_index is not None:
                    if skin_tone_index is not None:
                        outputs = self.model(images, modality_index, skin_tone_index)
                    else:
                        outputs = self.model(images, modality_index)
                elif skin_tone_index is not None:
                    outputs = self.model(images, skin_tone_index)
                else:
                    outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                true_labels = labels
            loss = self.train_criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)
        acc = 100 * np.mean(all_preds_np == all_labels_np)
        f1 = f1_score(all_labels_np, all_preds_np, average='weighted')
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, acc, f1

    def _validate_epoch(self):
        """
        Run one validation epoch.
        Returns average loss, accuracy, F1 score, and optionally raw outputs and IDs.
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_img_ids = []
        all_raw_outputs = []
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch_data in pbar:
                images, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
                if self.modality_aware:
                    if self.skin_tone_aware:
                        modality_index = batch_data[2].to(self.device)
                        skin_tone_index = batch_data[3].to(self.device)
                    else:
                        modality_index = batch_data[2].to(self.device)
                        skin_tone_index = None
                elif self.skin_tone_aware:
                    skin_tone_index = batch_data[2].to(self.device)
                    modality_index = None
                else:
                    modality_index = None
                    skin_tone_index = None
                if modality_index is not None:
                    if skin_tone_index is not None:
                        outputs = self.model(images, modality_index, skin_tone_index)
                    else:
                        outputs = self.model(images, modality_index)
                elif skin_tone_index is not None:
                    outputs = self.model(images, skin_tone_index)
                else:
                    outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                all_raw_outputs.extend(outputs.cpu().numpy())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if self.eval_mode:
                    all_img_ids.extend(batch_data[-1])
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return (total_loss / len(self.val_loader),
                100 * correct / total,
                f1,
                all_img_ids,
                all_preds,
                all_labels,
                all_raw_outputs
                )

    def train(self, ckpt_dir, run_name):
        """
        Run the full training loop for the specified number of epochs.
        Saves the best model based on validation F1 score.
        """
        best_val_f1 = 0
        for epoch in range(self.config['num_epochs']):
            print(f'\nEpoch {epoch+1}/{self.config["num_epochs"]}')
            train_loss, train_acc, train_f1 = self._train_epoch()
            val_loss, val_acc, val_f1, all_ids, all_preds, all_labels, all_raw_outputs = self._validate_epoch()
            self.scheduler.step(val_loss)

            # Early stopping logic (monitor val_f1 for improvement)
            if val_f1 > self.best_val_metric + self.min_delta:
                print(f"Validation F1 improved from {self.best_val_metric:.4f} to {val_f1:.4f}. Saving model...")
                self.best_val_metric = val_f1
                self.counter = 0  # Reset counter since improvement was observed
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(self.model.state_dict(), f"{ckpt_dir}/{run_name}.pth")
                print(f"New best model saved to {ckpt_dir}/{run_name}.pth with F1: {self.best_val_metric:.4f}")
            else:
                self.counter += 1
                print(f"Validation F1 did not improve. Patience counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    print(f"Early stopping triggered after {self.patience} epochs without improvement.")
                    break # Stop training

            if self.wandb:
                self.wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_f1": train_f1,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })

            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}')

            # if epoch == self.config['num_epochs'] - 1:
            #     self.plot_confusion_matrix(all_labels, all_preds, self.classes)