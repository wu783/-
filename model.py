"""
水果分类CNN模型 - 基于ResNet-50迁移学习
"""

import torch
import torch.nn as nn
from torchvision import models
from fruits import NUM_CLASSES


class FruitClassifier(nn.Module):
    """
    基于ResNet-50迁移学习的水果分类器

    使用ImageNet预训练的ResNet-50作为骨干网络，
    替换最后的全连接层适配水果分类任务。
    参数量约2500万，适合中等规模多分类任务。
    """

    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=0.3, freeze_backbone=True):
        """
        Args:
            num_classes: 水果类别数 (默认从fruits.py自动获取)
            dropout_rate: Dropout比率
            freeze_backbone: 是否冻结骨干网络
        """
        super(FruitClassifier, self).__init__()

        # 加载预训练ResNet-50
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )

        # 替换分类头
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """冻结骨干网络"""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        """解冻全部参数"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_last_layers(self, num_layers=2):
        """解冻最后几层"""
        layers = [self.backbone.layer1, self.backbone.layer2,
                  self.backbone.layer3, self.backbone.layer4]
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())


def get_model(num_classes=NUM_CLASSES, **kwargs):
    """获取模型实例"""
    return FruitClassifier(num_classes=num_classes, **kwargs)
