# 🍎 23种水果智能识别 — 基于ResNet-50迁移学习

## 项目简介

基于ResNet-50深度卷积神经网络，实现23种常见水果的自动识别分类。使用ImageNet预训练权重进行迁移学习，通过百度图片搜索引擎自构训练数据集，并基于Gradio构建交互式Web识别界面。

## 支持的23种水果

🍎苹果 🥑牛油果 🍌香蕉 🫐蓝莓 🍒樱桃 🥥椰子 🐉火龙果 🍈榴莲
🍇葡萄 🥝猕猴桃 🍋柠檬 🍒荔枝 🥭芒果 🍊橙子 🍈木瓜 🍑桃子
🍐梨 🍍菠萝 🍑石榴 ⭐杨桃 🍓草莓 🍅番茄 🍉西瓜

## 项目结构

```
人工智能课程设计2/
├── main.py              # 主程序入口（下载→训练→评估全流程）
├── model.py             # ResNet-50迁移学习模型（FruitClassifier）
├── fruits.py            # 23种水果类别定义（含中英文名、搜索关键词）
├── dataset.py           # 数据爬取（百度图片）与预处理
├── train.py             # 训练模块（类别加权、早停、学习率调度）
├── evaluate.py          # 评估与可视化（PNG + MATLAB .mat导出）
├── gui.py               # Gradio Web交互界面（Top-5预测）
├── requirements.txt     # Python依赖
├── data/
│   ├── train/           # 训练集（23类，百度图片爬取，每类约100–160张）
│   └── test/            # 测试集（23类，百度图片爬取，每类约15–40张）
└── results/
    ├── models/          # 模型权重（best_model.pth）
    └── plots/           # 可视化图表（训练曲线、混淆矩阵、.mat文件）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据 & 训练

```bash
# 一键完成：下载数据 + 训练 + 评估
python main.py

# 或分步执行
python main.py --download_only       # 仅下载数据集
python main.py --train_only          # 仅训练（需已有数据）
python main.py --eval_only           # 仅评估已有模型

# 使用已有数据集（跳过下载）
python main.py --data_dir ./data/train
```

### 3. 启动GUI

```bash
python gui.py
```

浏览器打开 `http://localhost:7860`，上传水果图片即可获得Top-5识别结果。

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--download_only` | 仅下载数据集（百度图片爬取） | False |
| `--train_only` | 仅训练模型 | False |
| `--eval_only` | 仅评估已有模型 | False |
| `--data_dir` | 已有数据集目录（指定后跳过下载） | None |
| `--train_per_class` | 每类训练图目标数量 | 100 |
| `--test_per_class` | 每类测试图目标数量 | 20 |
| `--img_size` | 输入图像尺寸 | 224 |
| `--epochs` | 第一阶段训练轮数 | 30 |
| `--batch_size` | 批次大小 | 32 |
| `--lr` | 初始学习率 | 0.001 |
| `--num_workers` | 数据加载线程数 | 2 |
| `--cpu` | 强制使用CPU | False |

### GUI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 模型权重路径 | results/models/best_model.pth |
| `--port` | 服务端口 | 7860 |
| `--share` | 生成公网分享链接 | False |

## 数据集

### 数据来源

通过 `icrawler` 库调用**百度图片搜索引擎**爬取，每类使用多个中英文关键词以提高样本多样性。例如：

- 苹果: `苹果 水果`, `红苹果 水果`, `apple fruit`
- 草莓: `草莓 水果`, `strawberry fruit`

### 数据预处理与增强

- 自动校验并清理损坏/过小（<50px）的图片
- 训练集增强：随机水平翻转（p=0.5）、随机旋转（±20°）、颜色抖动、随机平移
- 归一化：ImageNet 均值/标准差 `([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`
- 训练/验证按 8:2 随机划分（固定随机种子 42）

## 模型架构

| 组件 | 配置 |
|------|------|
| 骨干网络 | ResNet-50（ImageNet-1K V2 预训练权重） |
| 分类头 | 2048 → Dropout(0.3) → 512 → ReLU → Dropout(0.15) → 256 → ReLU → 23 |
| 输入尺寸 | 224×224 RGB |
| 总参数量 | 约 2500 万 |

## 训练策略

采用**两阶段训练**方案，训练过程中自动应用类别加权损失以缓解数据不均衡。

### 第一阶段：冻结骨干，训练分类头

- 冻结 ResNet-50 全部卷积层（`fc` 层除外）
- 优化器：Adam（lr=0.001, weight_decay=1e-4）
- 学习率调度：ReduceLROnPlateau（factor=0.5, patience=5）
- 早停：patience=12（监控验证损失）
- 损失函数：类别加权 CrossEntropyLoss
- 最多训练 30 个 epoch

### 第二阶段：解冻微调

- 解冻 ResNet-50 最后 2 层（layer3, layer4）
- 学习率降至第一阶段 1/10（lr=0.0001）
- 固定训练 5 个 epoch
- 仅当验证准确率提升超过 1% 时保存新模型

## 评估与可视化

运行 `python main.py --eval_only` 或训练完成后自动执行：

- 整体准确率与验证损失
- 分类报告（sklearn classification_report）
- 混淆矩阵（`results/plots/confusion_matrix.png`）
- 训练曲线（`results/plots/training_history.png`）
- MATLAB 格式训练数据导出（`results/plots/training_history.mat`）

## GUI 功能

基于 **Gradio** 构建的 Web 交互界面，支持：

- 📤 上传水果图片（支持 jpg/png/webp 格式）
- 🔍 实时显示 Top-5 预测结果（含概率百分比）
- 📊 前三名详情（概率对比）
- 📖 折叠面板展示模型信息与使用建议
- 🌐 支持 `--share` 参数生成公网链接

## 依赖

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
pillow>=10.0.0
tqdm>=4.65.0
icrawler>=0.6.0
gradio>=5.0.0
```

## 参考文献

1. He K, Zhang X, Ren S, et al. Deep Residual Learning for Image Recognition. CVPR 2016.
2. PyTorch Documentation. https://pytorch.org/docs/stable/
3. icrawler Documentation. https://github.com/hellock/icrawler
4. Gradio Documentation. https://www.gradio.app/docs/
