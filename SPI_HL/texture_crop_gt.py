# ============================================================
# Texture-aware GT Augmentation
#
# 功能：
# 1. 旋转
# 2. 翻转
# 3. 组合翻转
# 4. Sobel纹理分析
# 5. 高纹理区域优先裁剪
#
# Author: HanLei Version
# ============================================================

import os
import random

import numpy as np

from pathlib import Path

from PIL import Image

from scipy.ndimage import sobel

# ============================================================
# 参数区
# ============================================================

# GT路径
GT_DIR = r"D:\GT"

# 输出路径
OUT_DIR = r"D:\GT_PATCH"

# Patch尺寸
PATCH_SIZE = 256

# 每个增强图生成Patch数量
PATCH_PER_AUG = 30

# 梯度池大小
GRADIENT_POOL = 500

# 图像格式
IMG_EXT = ".png"

# ============================================================
# 创建目录
# ============================================================

Path(OUT_DIR).mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 数据增强
# ============================================================

def augment_image(img):

    augs = []

    augs.append(("ori", img))

    augs.append((
        "hflip",
        img.transpose(
            Image.FLIP_LEFT_RIGHT
        )
    ))

    augs.append((
        "vflip",
        img.transpose(
            Image.FLIP_TOP_BOTTOM
        )
    ))

    augs.append((
        "hvflip",
        img.transpose(
            Image.FLIP_LEFT_RIGHT
        ).transpose(
            Image.FLIP_TOP_BOTTOM
        )
    ))

    augs.append((
        "rot90",
        img.rotate(90)
    ))

    augs.append((
        "rot180",
        img.rotate(180)
    ))

    augs.append((
        "rot270",
        img.rotate(270)
    ))

    augs.append((
        "rot90_hflip",
        img.rotate(90).transpose(
            Image.FLIP_LEFT_RIGHT
        )
    ))

    return augs

# ============================================================
# Sobel纹理复杂度计算
# ============================================================

def texture_map(img):

    gray = np.array(
        img.convert("L"),
        dtype=np.float32
    )

    gx = sobel(gray, axis=0)

    gy = sobel(gray, axis=1)

    mag = np.sqrt(
        gx**2 + gy**2
    )

    return mag

# ============================================================
# 纹理引导裁剪
# ============================================================

def texture_crop(img,
                 patch_size,
                 pool_size=500):

    tex = texture_map(img)

    h, w = tex.shape

    if h < patch_size or w < patch_size:
        return None

    candidates = []

    scores = []

    for _ in range(pool_size):

        x = random.randint(
            0,
            w - patch_size
        )

        y = random.randint(
            0,
            h - patch_size
        )

        patch_tex = tex[
            y:y+patch_size,
            x:x+patch_size
        ]

        score = patch_tex.mean()

        candidates.append(
            (x, y)
        )

        scores.append(score)

    best_idx = np.argmax(scores)

    x, y = candidates[best_idx]

    patch = img.crop(
        (
            x,
            y,
            x + patch_size,
            y + patch_size
        )
    )

    return patch

# ============================================================
# 主程序
# ============================================================

files = sorted(
    [
        f for f in os.listdir(GT_DIR)
        if f.endswith(IMG_EXT)
    ]
)

total_patch = 0

for idx, name in enumerate(files):

    path = os.path.join(
        GT_DIR,
        name
    )

    img = Image.open(path)

    augs = augment_image(img)

    for aug_name, aug_img in augs:

        for k in range(
            PATCH_PER_AUG
        ):

            patch = texture_crop(
                aug_img,
                PATCH_SIZE,
                GRADIENT_POOL
            )

            if patch is None:
                continue

            save_name = (
                f"{Path(name).stem}"
                f"_{aug_name}"
                f"_patch{k:03d}.png"
            )

            patch.save(
                os.path.join(
                    OUT_DIR,
                    save_name
                )
            )

            total_patch += 1

    print(
        f"[{idx+1}/{len(files)}] "
        f"{name} 完成"
    )

print("\n================================")
print("增强完成")
print("总Patch:", total_patch)
print("保存路径:", OUT_DIR)
print("================================")