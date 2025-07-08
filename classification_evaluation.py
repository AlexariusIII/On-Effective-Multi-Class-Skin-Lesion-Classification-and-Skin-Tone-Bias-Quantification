"""
Evaluation utilities for skin lesion classification and skin tone bias quantification.
Provides the Evaluator class for model evaluation, metrics computation, and result aggregation.
"""
import pandas as pd
from torchvision import transforms
from dataset import SkinLesionDataset
from torch.utils.data import DataLoader
import torch
from train_utils import get_data_paths, initialize_model_and_loss
import torch.optim as optim
from trainer import Trainer
from scipy.special import softmax
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score

class Evaluator:
    """
    Evaluator class for running model evaluation and computing metrics on skin lesion datasets.
    """
    def __init__(self, device, config, task, img_type, oversample, modality_aware, skin_tone_aware, class_weight_flag, model_arch, ckpt_path=None, val_transform=None):
        """
        Initialize the Evaluator.
        Args:
            device: torch.device
            config: dict, configuration parameters
            task: str, classification task
            img_type: str, image type
            oversample: str, oversampling method
            modality_aware: bool, if model is modality aware
            skin_tone_aware: bool, if model is skin tone aware
            class_weight_flag: bool, if class weighting is used
            model_arch: str, model architecture
            ckpt_path: str or None, checkpoint path
            val_transform: torchvision transform or None
        """
        self.device = device
        self.config = config
        self.task = task
        self.img_type = img_type
        self.oversample = oversample
        self.modality_aware = modality_aware
        self.skin_tone_aware = skin_tone_aware
        self.class_weight_flag = class_weight_flag
        self.model_arch = model_arch
        if ckpt_path is None:
            self.ckpt_path = f"checkpoints/{img_type}/{task}/{model_arch}/arch-{model_arch}_cw-{class_weight_flag}_ma-{modality_aware}_sa-{skin_tone_aware}_ov-{oversample}.pth"
        else:
            self.ckpt_path = ckpt_path
        if val_transform is None:
            self.val_transform = transforms.Compose([
                transforms.Resize((config['image_size'], config['image_size'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.val_transform = val_transform

    def get_dataset_and_loader(self, test_path=None, test_df=None, batch_size=None):
        """
        Prepare test dataset and dataloader.
        Args:
            test_path: str or None, path to test CSV
            test_df: pd.DataFrame or None
            batch_size: int or None
        Returns:
            test_df, test_dataset, test_loader
        """
        if batch_size is None:
            batch_size = self.config['batch_size']
        if test_df is not None:
            pass
        elif test_path is not None:
            test_df = pd.read_csv(test_path)
        else:
            _, test_path = get_data_paths(self.task, self.img_type, self.oversample)
            test_df = pd.read_csv(test_path)
        test_dataset = SkinLesionDataset(csv_path=test_path, df=test_df, transform=self.val_transform, task=self.task, modality_aware=self.modality_aware, skin_tone_aware=self.skin_tone_aware, evaluation=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=self.config['num_workers'])
        return test_df, test_dataset, test_loader

    def load_model(self, test_dataset, embed_dim=64):
        """
        Load model and criterion from checkpoint.
        Args:
            test_dataset: SkinLesionDataset
            embed_dim: int
        Returns:
            model, criterion
        """
        model, criterion = initialize_model_and_loss(self.modality_aware, self.skin_tone_aware, self.model_arch, self.task, self.class_weight_flag, test_dataset, self.device, embed_dim=embed_dim)
        model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
        model.eval()
        return model, criterion

    def get_predictions(self, test_df, test_dataset, test_loader, model=None, criterion=None, embed_dim=128):
        """
        Run model on test set and return predictions and results.
        Args:
            test_df: pd.DataFrame
            test_dataset: SkinLesionDataset
            test_loader: DataLoader
            model: torch.nn.Module or None
            criterion: loss function or None
            embed_dim: int
        Returns:
            test_res_df: pd.DataFrame, test results merged with test_df
            test_res: tuple, raw results from validation
        """
        if model is None and criterion is None:
            model, criterion = initialize_model_and_loss(self.modality_aware, self.skin_tone_aware, self.model_arch, self.task, self.class_weight_flag, test_dataset, self.device, embed_dim=embed_dim)
            model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
        model.eval()
        optimizer = optim.AdamW(model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.config['scheduler_patience'], factor=self.config['scheduler_factor'])
        trainer = Trainer(
            model=model,
            train_loader=None,
            val_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            config=self.config,
            classes=list(test_dataset.label_map.keys()),
            mixup_fn=False,
            modality_aware=self.modality_aware,
            skin_tone_aware=self.skin_tone_aware,
            wandb=None,
            eval_mode=True
        )
        test_res = trainer._validate_epoch()
        val_loss, val_acc, val_f1, all_ids, all_preds, all_labels, all_raw_outputs = test_res
        pred_df = pd.DataFrame({'raw_output': all_raw_outputs, 'pred': all_preds, 'gt': all_labels, 'image_id': all_ids})
        test_res_df = pd.merge(pred_df, test_df, on='image_id')
        return test_res_df, test_res

    def get_merices(self, test_res_df):
        """
        Compute metrics for each skin tone subset and overall.
        Args:
            test_res_df: pd.DataFrame, test results
        Returns:
            metric_df: pd.DataFrame, metrics for each skin tone subset
        """
        skin_tones = ['All', 'A', 'B', 'C']
        data_dicts = {}
        for skin_tone in skin_tones:
            result_dict = {}
            if skin_tone == 'All':
                dataset = test_res_df
            else:
                dataset = test_res_df[test_res_df['skin_tone'] == skin_tone]
            if len(dataset) == 0:
                result_dict['acc'] = float('nan')
                result_dict['f1'] = float('nan')
                result_dict['auroc'] = float('nan')
                result_dict['ap'] = float('nan')
                data_dicts[skin_tone] = result_dict
                continue
            preds = dataset['pred'].to_list()
            labels = dataset['gt'].to_list()
            raws = np.stack(dataset['raw_output'].to_numpy())
            result_dict['acc'] = accuracy_score(labels, preds)
            result_dict['f1'] = f1_score(labels, preds, average='macro')
            softmax_probs = softmax(raws, axis=1)
            result_dict['auroc'] = roc_auc_score(labels, softmax_probs, multi_class='ovr')
            result_dict['ap'] = average_precision_score(labels, softmax_probs, average='macro')
            data_dicts[skin_tone] = result_dict
        metric_df = pd.DataFrame.from_dict(data_dicts, orient='index')
        metric_df = metric_df.reset_index().rename(columns={'index': 'skintone_subset'})
        return metric_df

    def get_merices_skintone_classification(self, test_res_df):
        """
        Compute metrics for skin tone classification task.
        Args:
            test_res_df: pd.DataFrame, test results
        Returns:
            metric_df: pd.DataFrame, metrics for the model architecture
        """
        data_dicts = {}
        result_dict = {}
        dataset = test_res_df
        preds = dataset['pred'].to_list()
        labels = dataset['gt'].to_list()
        raws = dataset['raw_output'].to_list()
        result_dict['acc'] = accuracy_score(labels, preds)
        result_dict['f1'] = f1_score(labels, preds, average='macro')
        result_dict['auroc'] = roc_auc_score(labels, softmax(raws, axis=1), multi_class='ovr')
        result_dict['ap'] = average_precision_score(labels, softmax(raws, axis=1), average='macro')
        data_dicts[self.model_arch] = result_dict
        metric_df = pd.DataFrame.from_dict(data_dicts, orient='index')
        metric_df = metric_df.reset_index().rename(columns={'index': 'Arch'})
        return metric_df