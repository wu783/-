"""
训练模块 - 23种水果分类
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from collections import Counter


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_state = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        improved = (score < self.best_score - self.min_delta) if self.mode == 'min' else (score > self.best_score + self.min_delta)
        if improved:
            self.best_score = score
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{total_epochs}', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(model(inputs), 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100.*correct/total:.1f}%'})
    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for inputs, labels in tqdm(loader, desc='验证', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return running_loss / total, 100.0 * correct / total, np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, val_loader, device,
                epochs=30, lr=0.001, weight_decay=1e-4, save_dir='results/models'):
    os.makedirs(save_dir, exist_ok=True)

    # 类别加权
    train_targets = []
    for _, y in train_loader.dataset:
        try:
            train_targets.append(y.item() if torch.is_tensor(y) else y)
        except:
            train_targets.append(y)
    class_counts = Counter(train_targets)
    max_count = max(class_counts.values())
    weights = torch.tensor(
        [max_count / max(class_counts.get(i, 1), 1) for i in range(len(class_counts))],
        dtype=torch.float
    ).to(device)
    print(f"类别分布范围: {min(class_counts.values())}~{max_count}, 权重范围: {weights.min():.2f}~{weights.max():.2f}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    early_stop = EarlyStopping(patience=12, mode='min')

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    t_start = time.time()

    print(f"\n开始训练: epochs={epochs}, lr={lr}, batch={train_loader.batch_size}")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch:3d}/{epochs}] Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc, 'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  => 保存最佳模型 ({val_acc:.2f}%)")

        if early_stop(val_loss, model):
            print(f"\n早停触发! 最佳准确率: {best_acc:.2f}%")
            break

    print(f"\n训练完成! 用时: {time.time()-t_start:.0f}秒, 最佳准确率: {best_acc:.2f}%")
    ckpt = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(ckpt['model_state_dict'])
    return history
