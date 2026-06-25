import os
import time
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
from icrawler.builtin import BaiduImageCrawler
from fruits import FRUITS, NUM_CLASSES


# ============================================================
# 数据集爬取
# ============================================================

def download_fruit_images(save_dir='data/train', images_per_class=100):
    print("\n" + "=" * 60)
    print(f"自构{NUM_CLASSES}种水果数据集 - 百度图片爬取 (目标: 每类{images_per_class}张)")
    print("=" * 60)

    for folder, cn_name, keywords in FRUITS:
        cls_dir = os.path.join(save_dir, folder)
        os.makedirs(cls_dir, exist_ok=True)

        existing = len([f for f in os.listdir(cls_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if existing >= images_per_class:
            print(f"  [{cn_name}] 已有 {existing} 张，跳过")
            continue

        needed = images_per_class - existing
        per_q = max(40, needed // len(keywords) + 30)

        for kw in keywords:
            current = len([f for f in os.listdir(cls_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if current >= images_per_class:
                break
            try:
                crawler = BaiduImageCrawler(
                    downloader_threads=8,
                    storage={'root_dir': cls_dir},
                    log_level='WARNING'
                )
                crawler.crawl(keyword=kw, max_num=per_q,
                             min_size=(100, 100), file_idx_offset='auto')
                time.sleep(1.0)
            except Exception as e:
                print(f"  [{cn_name}] '{kw}' 出错: {e}")

        # 清理非图片文件
        for f in os.listdir(cls_dir):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                os.remove(os.path.join(cls_dir, f))

        count = len([f for f in os.listdir(cls_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  [{cn_name}] {count} 张")

    print(f"\n训练集保存路径: {save_dir}")
    show_stats(save_dir)


def download_test_images(save_dir='data/test', images_per_class=20):
    print(f"\n下载测试集 (每类{images_per_class}张)...")
    for folder, cn_name, keywords in FRUITS:
        cls_dir = os.path.join(save_dir, folder)
        os.makedirs(cls_dir, exist_ok=True)
        existing = len([f for f in os.listdir(cls_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if existing >= images_per_class:
            continue
        try:
            crawler = BaiduImageCrawler(
                downloader_threads=6,
                storage={'root_dir': cls_dir},
                log_level='WARNING'
            )
            crawler.crawl(keyword=keywords[0], max_num=images_per_class,
                         min_size=(100, 100), file_idx_offset='auto')
            time.sleep(0.5)
        except:
            pass
    show_stats(save_dir)


def validate_images(data_dir):
    import warnings
    warnings.filterwarnings('ignore')
    for folder, cn_name, _ in FRUITS:
        cls_dir = os.path.join(data_dir, folder)
        if not os.path.exists(cls_dir):
            continue
        removed = 0
        for f in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, f)
            try:
                img = Image.open(fpath)
                img.verify()
                img = Image.open(fpath)
                if img.width < 50 or img.height < 50:
                    os.remove(fpath); removed += 1
            except:
                os.remove(fpath); removed += 1
        count = len([f for f in os.listdir(cls_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if removed > 0:
            print(f"  [{cn_name}] 清理 {removed} 张损坏 → 剩余 {count} 张")


def show_stats(data_dir):
    total = 0
    for folder, cn_name, _ in FRUITS:
        cls_dir = os.path.join(data_dir, folder)
        if os.path.exists(cls_dir):
            n = len([f for f in os.listdir(cls_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            total += n
    print(f"总计: {total} 张图片 ({NUM_CLASSES}类)")


def prepare_dataset(data_dir=None, train_per_class=100, test_per_class=20):
    if data_dir and os.path.exists(data_dir):
        print(f"使用已有数据集: {data_dir}")
        return data_dir

    download_fruit_images('data/train', train_per_class)
    validate_images('data/train')

    download_test_images('data/test', test_per_class)
    validate_images('data/test')

    return 'data/train'


# ============================================================
# 数据预处理与加载
# ============================================================

def get_transforms(img_size=224, augment=True):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        orig = self.subset.dataset.transform
        self.subset.dataset.transform = self.transform
        x, y = self.subset[idx]
        self.subset.dataset.transform = orig
        return x, y

    def __len__(self):
        return len(self.subset)


def create_dataloaders(data_dir, img_size=224, batch_size=32,
                       val_split=0.2, num_workers=2):
    train_transform, val_transform = get_transforms(img_size, augment=True)
    full_dataset = datasets.ImageFolder(root=data_dir)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset.dataset.transform = train_transform
    val_dataset = TransformedSubset(val_dataset, val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    num_classes = len(full_dataset.classes)
    print(f"类别数: {num_classes}")
    print(f"训练集: {train_size} 张, 验证集: {val_size} 张")
    print(f"每epoch批次: 训练={len(train_loader)}, 验证={len(val_loader)}")

    return train_loader, val_loader, num_classes
