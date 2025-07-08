"""
Training script for skin lesion classification and skin tone bias quantification.
Handles argument parsing, data loading, model training, and evaluation.
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
    Main training and evaluation routine.
    Parses arguments, loads data, trains model, and evaluates on test set.
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
    if model_arch != 'convnext_base':
        batch_size = 64
    else:
        batch_size = config['batch_size']
    print(f"Using batch size: {batch_size}")
    run_name = f"arch-{args.arch}_cw-{args.class_weight}_ma-{args.modality_aware}_sa-{args.skin_tone_aware}_ov-{args.oversample}_subset-{args.subset}"
    ckpt_dir = f"checkpoints/{args.img_type}/{args.task}/{args.arch}"
    wandb.init(
        project=f"skin-cancer-{args.img_type}-{args.task}",
        name=run_name,
        config=config
    )
    print(f"Starting Training for Classification task: Img_Type:{args.img_type}-Task:{args.task}-MA:{args.modality_aware}-SA:{args.skin_tone_aware}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    wandb.config.update({"device": str(device), "model": str(args.arch)})
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_size'], scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.01, 0.05), ratio=(0.3, 3.3), value='random'),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_path, val_path = get_data_paths(task, img_type, oversample, subset)
    train_dataset = SkinLesionDataset(csv_path=train_path, transform=train_transform, task=args.task, modality_aware=modality_aware, skin_tone_aware=skin_tone_aware)
    val_dataset = SkinLesionDataset(csv_path=val_path, transform=val_transform, task=args.task, modality_aware=modality_aware, skin_tone_aware=skin_tone_aware)
    print(f"Train Size: {len(train_dataset)}")
    print(f"Test Size: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config['num_workers'], drop_last=config['mix_up'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config['num_workers'])
    model, criterion = initialize_model_and_loss(modality_aware, skin_tone_aware, model_arch, task, class_weight_flag, train_dataset, device, embed_dim=config['embed_dim'])
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
    trainer.train(ckpt_dir, run_name)
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}.pth")
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
    test_df, test_dataset, test_loader = evaluator.get_dataset_and_loader()
    test_res_df, test_res = evaluator.get_predictions(test_df, test_dataset, test_loader, embed_dim=config['embed_dim'])
    metrics_df = evaluator.get_merices(test_res_df)
    result_path = f"results/{img_type}/{task}/{model_arch}"
    file_name = f"arch-{model_arch}_cw-{class_weight_flag}_ma-{modality_aware}_sa-{skin_tone_aware}_ov-{oversample}.csv"
    os.makedirs(result_path, exist_ok=True)
    metrics_df.to_csv(os.path.join(result_path, file_name), index=False)
    test_acc = metrics_df['acc'].to_list()[0]

if __name__ == '__main__':
    main()