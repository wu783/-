"""
评估模块 - 水果分类模型评估与可视化
支持Matplotlib和MATLAB两种绘图方式
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import io as sio
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import torch
import torch.nn as nn
from fruits import CLASS_NAMES

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_training_history(history, save_dir='results/plots'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    # ==== 保存为 .mat 文件 (供MATLAB绘图) ====
    mat_data = {
        'epoch': np.array(list(epochs), dtype=np.float64),
        'train_loss': np.array(history['train_loss'], dtype=np.float64),
        'val_loss': np.array(history['val_loss'], dtype=np.float64),
        'train_acc': np.array(history['train_acc'], dtype=np.float64),
        'val_acc': np.array(history['val_acc'], dtype=np.float64),
    }
    mat_path = os.path.join(save_dir, 'training_history.mat')
    sio.savemat(mat_path, mat_data)
    print(f"训练数据已导出为MATLAB格式: {mat_path}")

    # ==== Matplotlib绘图 (备用) ====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history['train_loss'], 'b-', lw=2, label='Training Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', lw=2, label='Validation Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history['train_acc'], 'b-', lw=2, label='Training Acc')
    axes[1].plot(epochs, history['val_acc'], 'r-', lw=2, label='Validation Acc')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curves', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线(PNG)已保存: {save_dir}/training_history.png")


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir='results/plots', top_k=30):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names[:top_k], yticklabels=class_names[:top_k],
                annot_kws={'size': 7}, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Confusion Matrix ({top_k} Classes)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: {save_dir}/confusion_matrix.png")


def evaluate_model(model, val_loader, device, class_names, save_dir='results/plots'):
    from train import validate
    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)

    val_loss, val_acc, preds, labels = validate(
        model, val_loader, nn.CrossEntropyLoss(), device
    )

    acc = accuracy_score(labels, preds)
    print(f"\n整体准确率: {acc:.4f} ({acc*100:.2f}%)")
    print(f"验证损失: {val_loss:.4f}")
    print("\n分类报告 (前10类):")
    print(classification_report(labels, preds, target_names=class_names,
                                labels=range(min(10, len(class_names))), digits=4))

    plot_confusion_matrix(labels, preds, class_names, save_dir)
    return {'accuracy': acc, 'val_loss': val_loss, 'val_acc': val_acc}
