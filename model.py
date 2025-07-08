"""
Model definitions for skin lesion classification and skin tone bias quantification.
Includes Timm_Classification_Model, Single_Gated_Model, and Double_Gated_Model.
"""
import torch
import torch.nn as nn
import timm

class Timm_Classification_Model(nn.Module):
    """
    Wrapper for timm image classification models with a custom classifier head.
    """
    def __init__(self, num_classes=3, model_arch='convnext', dropout_rate=0.3):
        super(Timm_Classification_Model, self).__init__()
        self.model_dict = {
            'convnext_base': 'convnext_base.fb_in22k_ft_in1k_384',
            'convnext_small': 'convnext_small.fb_in22k_ft_in1k_384',
            'convnext_tiny': 'convnext_tiny.fb_in22k_ft_in1k_384',
            'mobilenet': 'mobilenetv3_small_100',
            'tinyvit': 'tiny_vit_21m_384.dist_in22k_ft_in1k',
            'vit_small': 'vit_small_patch16_384.augreg_in21k_ft_in1k',
            'vit_base': 'vit_base_patch16_384.augreg_in21k_ft_in1k',
            'efficientnetv2_s': 'tf_efficientnetv2_s.in21k_ft_in1k',
        }
        print(f'Using model: {model_arch}')
        self.backbone = timm.create_model(
            self.model_dict[model_arch],
            pretrained=True,
            num_classes=0  # remove head, get features
        )
        backbone_feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(backbone_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )
        print(f"Native Model - Classifier input dim: {backbone_feat_dim}")

    def forward(self, x):
        """Forward pass."""
        feat = self.backbone(x)
        return self.classifier(feat)

class Single_Gated_Model(nn.Module):
    """
    Single-gated neural network model that modulates backbone features
    based on modality or skin tone information.
    """
    def __init__(self, num_classes=3, model_arch='convnext_base', dropout_rate=0.3, embed_dim=128, modality_aware=False, skin_tone_aware=False):
        super(Single_Gated_Model, self).__init__()
        self.model_dict = {
            'convnext_base': 'convnext_base.fb_in22k_ft_in1k_384',
            'convnext_small': 'convnext_small.fb_in22k_ft_in1k_384',
            'convnext_tiny': 'convnext_tiny.fb_in22k_ft_in1k_384',
            'mobilenet': 'mobilenetv3_small_100',
            'tinyvit': 'tiny_vit_21m_384.dist_in22k_ft_in1k',
            'vit_small': 'vit_small_patch16_384.augreg_in21k_ft_in1k',
            'vit_base': 'vit_base_patch16_384.augreg_in21k_ft_in1k',
            'efficientnetv2_s': 'tf_efficientnetv2_s.in21k_ft_in1k',
        }
        print(f'Using model: {model_arch} for Single Gated Model')
        self.backbone = timm.create_model(
            self.model_dict[model_arch],
            pretrained=True,
            num_classes=0
        )
        backbone_feat_dim = self.backbone.num_features
        if modality_aware:
            self.gated_embedding = nn.Embedding(2, embed_dim)
            self.embedding_type = 'modality'
        elif skin_tone_aware:
            self.gated_embedding = nn.Embedding(3, embed_dim)
            self.embedding_type = 'skin_tone'
        else:
            raise ValueError("Either 'modality_aware' or 'skin_tone_aware' must be True for Single_Gated_Model.")
        self.gating_network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 2, backbone_feat_dim),
            nn.Sigmoid()
        )
        classifier_input_dim = backbone_feat_dim + embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )
        print(f"Single Gated Model - backbone feature dim: {backbone_feat_dim}")
        print(f"Single Gated Model - embed dim: {embed_dim}")
        print(f"Single Gated Model - Classifier input dim: {classifier_input_dim}")

    def forward(self, x, gate_idx):
        """
        Forward pass for the Single_Gated_Model.
        Args:
            x (torch.Tensor): Input image tensor.
            gate_idx (torch.Tensor): Index for the gating mechanism (modality_idx or skin_tone_idx).
        Returns:
            torch.Tensor: Model output (logits for classification).
        """
        feat = self.backbone(x)
        gate_emb = self.gated_embedding(gate_idx)
        gate_values = self.gating_network(gate_emb)
        modulated_feat = feat * gate_values
        combined = torch.cat([modulated_feat, gate_emb], dim=1)
        out = self.classifier(combined)
        return out

class Double_Gated_Model(nn.Module):
    """
    Double-gated neural network model that modulates backbone features
    based on both modality and skin tone information.
    """
    def __init__(self, num_classes=3, model_arch='convnext_base', dropout_rate=0.3, embed_dim=128):
        super(Double_Gated_Model, self).__init__()
        self.model_dict = {
            'convnext_base': 'convnext_base.fb_in22k_ft_in1k_384',
            'convnext_small': 'convnext_small.fb_in22k_ft_in1k_384',
            'convnext_tiny': 'convnext_tiny.fb_in22k_ft_in1k_384',
            'mobilenet': 'mobilenetv3_small_100',
            'tinyvit': 'tiny_vit_21m_384.dist_in22k_ft_in1k',
            'vit_small': 'vit_small_patch16_384.augreg_in21k_ft_in1k',
            'vit_base': 'vit_base_patch16_384.augreg_in21k_ft_in1k',
            'efficientnetv2_s': 'tf_efficientnetv2_s.in21k_ft_in1k',
            'vit_b_dinov2': 'vit_base_patch14_reg4_dinov2.lvd142m',
        }
        print(f'Using model: {model_arch} for Double Gated Model')
        self.backbone = timm.create_model(
            self.model_dict[model_arch],
            pretrained=True,
            num_classes=0
        )
        backbone_feat_dim = self.backbone.num_features
        self.modality_embedding = nn.Embedding(2, embed_dim)
        self.skin_tone_embedding = nn.Embedding(3, embed_dim)
        self.modality_gating_network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 2, backbone_feat_dim),
            nn.Sigmoid()
        )
        self.skin_tone_gating_network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 2, backbone_feat_dim),
            nn.Sigmoid()
        )
        classifier_input_dim = backbone_feat_dim + 2 * embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )
        print(f"Double Gated Model - backbone feature dim: {backbone_feat_dim}")
        print(f"Double Gated Model - embed dim: {embed_dim}")
        print(f"Double Gated Model - Classifier input dim: {classifier_input_dim}")

    def forward(self, x, modality_idx, skintone_idx):
        """
        Forward pass for the Double_Gated_Model.
        Args:
            x (torch.Tensor): Input image tensor.
            modality_idx (torch.Tensor): Index for modality embedding.
            skintone_idx (torch.Tensor): Index for skin tone embedding.
        Returns:
            torch.Tensor: Model output (logits for classification).
        """
        feat = self.backbone(x)
        modality_emb = self.modality_embedding(modality_idx)
        skin_tone_emb = self.skin_tone_embedding(skintone_idx)
        modality_gate = self.modality_gating_network(modality_emb)
        skin_tone_gate = self.skin_tone_gating_network(skin_tone_emb)
        modulated_feat = feat * modality_gate * skin_tone_gate
        combined = torch.cat([modulated_feat, modality_emb, skin_tone_emb], dim=1)
        out = self.classifier(combined)
        return out