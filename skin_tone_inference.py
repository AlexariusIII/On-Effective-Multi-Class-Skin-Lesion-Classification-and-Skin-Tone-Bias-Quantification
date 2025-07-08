"""
Script for inferring skin tone using a trained classification model and saving results.
Loads a dataset, runs inference, and outputs predictions and combined results.
"""
import pandas as pd
from torchvision import transforms
from dataset import SkinLesionDataset
from torch.utils.data import DataLoader
import torch
from model import Timm_Classification_Model
from sklearn.metrics import accuracy_score, f1_score

# Load dataset and apply transforms
# (mock skin_tone label, save original label)
data_path = './data/processed_data/combined_data.csv'
val_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
df = pd.read_csv(data_path)
df['original_skin_tone'] = df['skin_tone']
df['skin_tone'] = 'A'
dataset = SkinLesionDataset(data_path, transform=val_transform, task="skin_tone", df=df, evaluation=True)
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

# Model loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = dataset.num_classes

def load_model(ckpt_path, num_classes, arch='convnext_base'):
    """
    Load a classification model from checkpoint.
    Args:
        ckpt_path (str): Path to checkpoint.
        num_classes (int): Number of output classes.
        arch (str): Model architecture.
    Returns:
        model (nn.Module): Loaded model.
    """
    model = Timm_Classification_Model(num_classes=num_classes, model_arch=arch).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model

arch = "vit_small"
ckpt_path = f"checkpoints/all/skin_tone/vit_small/arch-vit_small_cw-False_ma-False_ov-none.pth"
model = load_model(ckpt_path, num_classes=num_classes, arch=arch)

# Inference
all_ids = []
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels, ids in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_ids.extend(ids)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_res = pd.DataFrame(data={"image_id": all_ids, "pred": all_preds, "gt": all_labels})
test_res = pd.merge(test_res, df, on='image_id')

# Map predictions to skin tone labels
skin_tone_mapping = {0: 'A', 1: 'B', 2: 'C'}
test_res['predicted_skin_tone'] = test_res['pred'].map(skin_tone_mapping)
test_res['skin_tone'] = test_res['original_skin_tone']

# Save results
test_res[['image_id', 'binary', 'multiclass', 'dataset', 'image_type', 'skin_tone', 'predicted_skin_tone']]
results_df = test_res[['image_id', 'binary', 'multiclass', 'dataset', 'image_type', 'skin_tone', 'predicted_skin_tone']]
results_df['combined_skin_tone'] = results_df['skin_tone'].combine_first(results_df['predicted_skin_tone'])
results_df.to_csv('./data/processed_data/combined_data_with_pseudo.csv', index=False)

# Optionally, analyze misclassifications
diff_df = results_df.dropna(subset=['skin_tone'])
falsed_preds_df = diff_df[diff_df['skin_tone'] != diff_df['predicted_skin_tone']]



