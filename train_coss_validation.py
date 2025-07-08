"""
Cross-validation training script for skin lesion classification and skin tone bias quantification.
Handles argument parsing, data loading, model training, and evaluation with k-fold cross-validation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SkinLesionDataset
import numpy as np
import wandb
import yaml
from pathlib import Path
import os
from timm.data.mixup import Mixup
import argparse
from trainer import Trainer
from classification_evaluation import Evaluator
from train_utils import get_data_paths, initialize_model_and_loss
from sklearn.model_selection import StratifiedKFold
import pandas as pd

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """
    Main cross-validation training and evaluation routine.
    Parses arguments, loads data, trains model, and evaluates on test set for each fold.
    """
    set_seed(42)
    parser = argparse.ArgumentParser(description="Skin Lesion Classification")
    parser.add_argument('--task', type=str, default="skin_tone", help="Specify the classification task ('binary','multiclass,'skin_tone')")
    parser.add_argument('--img_type', type=str, default="derma", help="Specify the image type to train on")
    parser.add_argument('--arch', type=str, default=None, help="Specify the model architecture to use (see model class)")
    parser.add_argument('--class_weight', type=str, default="False", help="Specify to add class weight or not. Use str(True)")
    parser.add_argument('--modality_aware', type=str, default="False", help="")
    parser.add_argument('--skin_tone_aware', type=str, default="False", help="")
    parser.add_argument('--oversample', type=str, default="False", help="Specify to use oversampling or not")
    parser.add_argument('--subset', type=str, default="False", help="Specify to use subset of train data")
    args = parser.parse_args()
    task = args.task
    img_type = args.img_type
    oversample = args.oversample
    modality_aware = args.modality_aware == "True"
    skin_tone_aware = args.skin_tone_aware == "True"
    model_arch = args.arch
    class_weight_flag = args.class_weight == "True"
    subset = args.subset == "True"
    config_path = Path(__file__).parent / 'config.yml'
    config = load_config(config_path)
    if model_arch == 'convnext_base':
        batch_size = 64
    else:
        batch_size = config['batch_size']
    print(f"Using batch size: {batch_size}")
    run_name = f"arch-{args.arch}_cw-{args.class_weight}_ma-{args.modality_aware}_sa-{args.skin_tone_aware}_ov-{args.oversample}_subset-{args.subset}"
    print(f"Starting Training for Classification task: Img_Type:{args.img_type}-Task:{args.task}-MA:{args.modality_aware}-SA:{args.skin_tone_aware}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_size'], scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_path, test_path = get_data_paths(task, img_type, oversample, subset)
    train_data_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X = train_data_df.drop('stratify_col', axis=1)
    y = train_data_df['stratify_col']
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_data_df['fold'] = -1
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        train_data_df.iloc[test_index, train_data_df.columns.get_loc('fold')] = fold
    for i in range(5):
        print(f"Starting Cross Validation Run: {i}")
        train_df = train_data_df[train_data_df['fold'] != i]
        val_df = train_data_df[train_data_df['fold'] == i]
        train_dataset = SkinLesionDataset(df=train_df, transform=train_transform, task=args.task, modality_aware=modality_aware, skin_tone_aware=skin_tone_aware)
        val_dataset = SkinLesionDataset(df=val_df, transform=val_transform, task=args.task, modality_aware=modality_aware, skin_tone_aware=skin_tone_aware)
        test_dataset = SkinLesionDataset(df=test_df, transform=val_transform, task=args.task, modality_aware=modality_aware, skin_tone_aware=skin_tone_aware)
        print(f"Train Size: {len(train_dataset)}")
        print(f"Val Size: {len(val_dataset)}")
        print(f"Test Size: {len(test_dataset)}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config['num_workers'], drop_last=config['mix_up'])
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config['num_workers'])
        model, criterion = initialize_model_and_loss(modality_aware, skin_tone_aware, model_arch, task, class_weight_flag, train_dataset, device, embed_dim=config['embed_dim'])
        wandb.init(
            project=f"isic25-{args.img_type}-{args.task}",
            group=run_name,
            name=f"Fold-{i}",
            config=config
        )
        wandb.watch(model, log="all")
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=config['scheduler_patience'], factor=config['scheduler_factor'])
        mixup_fn = None
        if config['mix_up']:
            mixup_fn = Mixup(
                mixup_alpha=0.8,
                cutmix_alpha=1.0,
                label_smoothing=0.1,
                num_classes=train_dataset.num_classes
            )
        ckpt_dir = f"../../synthetic-data-volume/checkpoints_cross-validated/{args.img_type}/{args.task}/{args.arch}/{run_name}"
        os.makedirs(ckpt_dir, exist_ok=True)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            classes=list(train_dataset.label_map.keys()),
            mixup_fn=mixup_fn,
            modality_aware=modality_aware,
            skin_tone_aware=skin_tone_aware,
            wandb=wandb
        )
        trainer.train(ckpt_dir, f"fold-{i}")
        ckpt_path = f"{ckpt_dir}/fold-{i}.pth"
        evaluator = Evaluator(
            device=device,
            config=config,
            task=task,
            img_type=img_type,
            oversample=oversample,
            modality_aware=modality_aware,
            skin_tone_aware=skin_tone_aware,
            class_weight_flag=class_weight_flag,
            model_arch=model_arch,
            ckpt_path=ckpt_path,
        )
        print("Loading best checkpoint and evaluating model on testset")
        test_df_eval, test_dataset_eval, test_loader_eval = evaluator.get_dataset_and_loader()
        test_res_df, test_res = evaluator.get_predictions(test_df_eval, test_dataset_eval, test_loader_eval, embed_dim=config['embed_dim'])
        metrics_df = evaluator.get_merices(test_res_df)
        result_path = f"results/{img_type}/{task}/{model_arch}/fold_{i}"
        os.makedirs(result_path, exist_ok=True)
        metrics_df.to_csv(os.path.join(result_path, f"metrics_fold_{i}.csv"), index=False)

if __name__ == '__main__':
    main()