"""
主程序入口 - 23种水果图像分类
用法:
    python main.py                        # 完整流程: 下载+训练+评估
    python main.py --download_only        # 仅下载数据集
    python main.py --train_only           # 仅训练 (需先下载数据)
    python main.py --eval_only            # 仅评估已有模型
"""

import os, sys, argparse, torch
import torch.nn as nn
from datetime import datetime

def parse_args():
    p = argparse.ArgumentParser(description='30种水果分类 - ResNet-50迁移学习')
    p.add_argument('--download_only', action='store_true', help='仅下载数据')
    p.add_argument('--train_only', action='store_true', help='仅训练')
    p.add_argument('--eval_only', action='store_true', help='仅评估')
    p.add_argument('--data_dir', type=str, default=None, help='已有数据集路径')
    p.add_argument('--train_per_class', type=int, default=100, help='每类训练图数量')
    p.add_argument('--test_per_class', type=int, default=20, help='每类测试图数量')
    p.add_argument('--img_size', type=int, default=224, help='图像尺寸')
    p.add_argument('--epochs', type=int, default=30, help='训练轮数')
    p.add_argument('--batch_size', type=int, default=32, help='批次大小')
    p.add_argument('--lr', type=float, default=0.001, help='学习率')
    p.add_argument('--cpu', action='store_true', help='强制CPU')
    p.add_argument('--num_workers', type=int, default=2, help='数据线程')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cpu') if args.cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("水果图像分类 - ResNet-50 迁移学习")
    print(f"设备: {device} | 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    from model import get_model
    from dataset import prepare_dataset, create_dataloaders
    from train import train_model
    from evaluate import evaluate_model, plot_training_history
    from fruits import CLASS_NAMES

    # ==== 数据准备 ====
    data_dir = prepare_dataset(
        data_dir=args.data_dir,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class
    )

    if args.download_only:
        print("\n数据下载完成!")
        return

    # ==== 数据加载 ====
    print(f"\n创建数据加载器 (img_size={args.img_size}, batch={args.batch_size})...")
    train_loader, val_loader, num_classes = create_dataloaders(
        data_dir, img_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"类别数: {num_classes}")

    if args.train_only or args.eval_only:
        pass  # 继续

    # ==== 模型 ====
    print(f"\n创建模型 ResNet-50 (num_classes={num_classes})...")
    model = get_model(num_classes=num_classes, freeze_backbone=True)
    model = model.to(device)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_p:,} | 可训练: {train_p:,} ({100*train_p/total_p:.1f}%)")

    # ==== 训练 ====
    if not args.eval_only:
        history = train_model(model, train_loader, val_loader, device,
                             epochs=args.epochs, lr=args.lr, save_dir='results/models')
        plot_training_history(history, save_dir='results/plots')

        # 第二阶段: 解冻最后2层微调
        print("\n第二阶段: 解冻最后2层微调...")
        model.unfreeze_last_layers(2)
        optim2 = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
        # 快速微调几轮
        from train import train_one_epoch, validate
        for ep in range(1, 6):
            tl, ta = train_one_epoch(model, train_loader, nn.CrossEntropyLoss(), optim2, device, ep, 5)
            vl, va, _, _ = validate(model, val_loader, nn.CrossEntropyLoss(), device)
            print(f"  微调 Epoch {ep}/5: Train Acc={ta:.2f}%, Val Acc={va:.2f}%")
            if va > history['val_acc'][-1] + 1:
                torch.save({'model_state_dict': model.state_dict(), 'val_acc': va},
                          'results/models/best_model.pth')
        print("微调完成")

    # ==== 评估 ====
    model_path = 'results/models/best_model.pth'
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"\n已加载最佳模型 (Val Acc: {ckpt['val_acc']:.2f}%)")

    metrics = evaluate_model(model, val_loader, device, CLASS_NAMES, save_dir='results/plots')
    print(f"\n项目完成! 最终准确率: {metrics['accuracy']*100:.2f}%")
    return metrics


if __name__ == '__main__':
    main()
