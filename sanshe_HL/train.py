# ============================================================
# train.py
#
# 功能：
#   使用散射退化图像(train)和对应真实图像(labels)
#   训练UNet网络，实现散射图像恢复。
#
# 数据结构：
#
# DATA_ROOT
# │
# ├── train
# │    ├── image001
# │    │    ├── noise1.bmp
# │    │    ├── noise2.bmp
# │    │    └── ...
# │    │
# │    ├── image002
# │    │    ├── noise1.bmp
# │    │    └── ...
# │
# └── labels
#      ├── image001.bmp
#      ├── image002.bmp
#      └── ...
#
# 训练目标：
#
# 输入：
#   散射图像（烟雾、浑浊介质、散斑等）
#
# 输出：
#   对应真实图像
#
# 作者建议：
#   RTX5060 / RTX4090 / A6000
#   使用 AMP 混合精度训练
#
# ============================================================

import os

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset import ScatterDataset
from network import UNet


# ============================================================
# 用户参数区
# ============================================================

# 数据集根目录
DATA_ROOT = r"F:\code\wangluo\wt_futong\data"

# 模型保存目录
SAVE_DIR = r"F:\code\wangluo\wt_futong\modles"

# 图像统一尺寸
#
# 所有图像都会被Resize到该尺寸
#
# 推荐：
# 256 -> 快速训练
# 512 -> 高质量训练
#
IMAGE_SIZE = 256

# Batch大小
#
# RTX5060 8GB:
#     建议 4~8
#
# RTX4090:
#     建议 16~32
#
# RTX A6000:
#     建议 16~32
#
BATCH_SIZE = 8

# 总训练轮数
EPOCHS = 30

# 初始学习率
LR = 1e-4

# DataLoader线程数
#
# Windows建议：
# 4~8
#
NUM_WORKERS = 0

# ============================================================ 
# 创建模型保存目录
# ============================================================

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# 自动选择设备
#
# RTX5060
# RTX4090
# A6000
# CPU
#
# 全部兼容
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("Device :", device)

if torch.cuda.is_available():

    print("GPU Name :",
          torch.cuda.get_device_name(0))

    print("CUDA Version :",
          torch.version.cuda)

print("=" * 60)

# ============================================================
# 构建数据集
# ============================================================

dataset = ScatterDataset(
    root_dir=DATA_ROOT,
    image_size=IMAGE_SIZE
)

print("Dataset Size =", len(dataset))

# ============================================================
# DataLoader
#
# shuffle=True
#     每个epoch随机打乱
#
# pin_memory=True
#     提高CPU->GPU传输速度
#
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# ============================================================
# 创建UNet模型
# ============================================================

model = UNet()

# ============================================================
# 多GPU训练
#
# 单GPU：
#     RTX5060
#
# 多GPU：
#     2×A6000
#     4×4090
#
# 自动开启
# ============================================================

if torch.cuda.device_count() > 1:

    print(
        f"Use {torch.cuda.device_count()} GPUs"
    )

    model = nn.DataParallel(model)

# 移动到GPU
model = model.to(device)

# ============================================================
# 损失函数
#
# MSE:
#
#     (prediction-target)^2
#
# 图像恢复任务最常用
#
# 后续可替换：
#
# L1Loss
# SSIMLoss
# CharbonnierLoss
#
# ============================================================

criterion = nn.MSELoss()

# ============================================================
# AdamW优化器
#
# 比Adam更稳定
#
# weight decay防止过拟合
# ============================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

# ============================================================
# 学习率自动衰减
#
# loss连续5轮不下降
#
# LR *= 0.5
#
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5,
    min_lr=1e-7
)

# ============================================================
# AMP自动混合精度
#
# RTX5060
# RTX4090
# A6000
#
# 训练速度提高
# 显存占用降低
#
# ============================================================

scaler = torch.amp.GradScaler("cuda")

# ============================================================
# 保存最佳模型
# ============================================================

best_loss = 1e10

# ============================================================
# 开始训练
# ============================================================

for epoch in range(EPOCHS):

    # 切换训练模式
    model.train()

    epoch_loss = 0

    # ========================================================
    # 遍历一个Epoch所有Batch
    # ========================================================

    for batch_idx, (imgs, labels) in enumerate(loader):

        # ----------------------------------------------------
        # 数据搬运到GPU
        # ----------------------------------------------------

        imgs = imgs.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # 梯度清零
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # 自动混合精度前向传播
        # ----------------------------------------------------

        with torch.amp.autocast("cuda"):

            outputs = model(imgs)

            loss = criterion(
                outputs,
                labels
            )

        # ----------------------------------------------------
        # 反向传播
        # ----------------------------------------------------

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        epoch_loss += loss.item()

        # ----------------------------------------------------
        # 打印当前Batch
        # ----------------------------------------------------

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Batch [{batch_idx+1}/{len(loader)}] "
            f"Loss = {loss.item():.6f}"
        )

    # ========================================================
    # Epoch平均Loss
    # ========================================================

    avg_loss = epoch_loss / len(loader)

    # ========================================================
    # 更新学习率
    # ========================================================

    scheduler.step(avg_loss)

    print()

    print("=" * 60)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Average Loss = {avg_loss:.6f}"
    )

    print(
        "Current LR =",
        optimizer.param_groups[0]["lr"]
    )

    print("=" * 60)

    print()

    # ========================================================
    # 保存最佳模型
    # ========================================================

    if avg_loss < best_loss:

        best_loss = avg_loss

        save_file = os.path.join(
            SAVE_DIR,
            "best_model.pth"
        )

        torch.save(
            model.state_dict(),
            save_file
        )

        print(
            f"[SAVE] Best Model Saved\n"
            f"Loss = {best_loss:.6f}\n"
            f"Path = {save_file}"
        )

# ============================================================
# 训练结束
# ============================================================

print()
print("=" * 60)
print("Training Finished")
print("Best Loss =", best_loss)
print("=" * 60)