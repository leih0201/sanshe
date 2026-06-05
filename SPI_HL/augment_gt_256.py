# ============================================================
# GT增强程序（适用于256×256单像素成像）
#
# 功能：
# 1. 水平翻转
# 2. 垂直翻转
# 3. 组合翻转
# 4. 90°/180°/270°旋转
# 5. 随机小角度旋转
# 6. 随机平移
# 7. 随机缩放
#
# Author: HanLei Version
# ============================================================

import os
import random

from pathlib import Path

from PIL import Image
from PIL import ImageOps
from PIL import ImageChops

# ============================================================
# 参数区（重点修改）
# ============================================================

# GT输入路径
GT_DIR = r"F:\code\wangluo\wt_futong\T_images\GT"

# 输出路径
OUT_DIR = r"F:\code\wangluo\wt_futong\T_images\GT_kuochong"

# 支持格式
IMG_EXTS = (
    ".png",
    ".bmp",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff"
)

# ============================================================
# 仿射增强参数
# ============================================================

# 每张图生成多少个随机仿射增强样本
AFFINE_NUM = 20

# 随机旋转角度范围
ROTATE_MIN = -10
ROTATE_MAX = 10

# 平移范围（像素）
TRANSLATE_MAX = 10

# 缩放范围
SCALE_MIN = 0.90
SCALE_MAX = 1.10

# 输出尺寸
OUTPUT_SIZE = 256

# ============================================================
# 创建目录
# ============================================================

Path(OUT_DIR).mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 固定增强
# ============================================================

def basic_augment(img):

    augs = []

    augs.append(("ori", img))

    augs.append((
        "hflip",
        ImageOps.mirror(img)
    ))

    augs.append((
        "vflip",
        ImageOps.flip(img)
    ))

    augs.append((
        "hvflip",
        ImageOps.flip(
            ImageOps.mirror(img)
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
        ImageOps.mirror(
            img.rotate(90)
        )
    ))

    return augs

# ============================================================
# 随机仿射增强
# ============================================================

def random_affine(img):

    angle = random.uniform(
        ROTATE_MIN,
        ROTATE_MAX
    )

    scale = random.uniform(
        SCALE_MIN,
        SCALE_MAX
    )

    tx = random.randint(
        -TRANSLATE_MAX,
        TRANSLATE_MAX
    )

    ty = random.randint(
        -TRANSLATE_MAX,
        TRANSLATE_MAX
    )

    # ------------------------
    # 缩放
    # ------------------------

    w, h = img.size

    new_w = int(w * scale)
    new_h = int(h * scale)

    scaled = img.resize(
        (new_w, new_h),
        Image.BICUBIC
    )

    canvas = Image.new(
        img.mode,
        (w, h),
        color=0
    )

    paste_x = (w - new_w) // 2
    paste_y = (h - new_h) // 2

    canvas.paste(
        scaled,
        (paste_x, paste_y)
    )

    # ------------------------
    # 旋转
    # ------------------------

    rotated = canvas.rotate(
        angle,
        resample=Image.BICUBIC
    )

    # ------------------------
    # 平移
    # ------------------------

    translated = ImageChops.offset(
        rotated,
        tx,
        ty
    )

    return translated

# ============================================================
# 保存函数
# ============================================================

def save_image(img, path):

    img = img.resize(
        (
            OUTPUT_SIZE,
            OUTPUT_SIZE
        ),
        Image.BICUBIC
    )

    img.save(path)

# ============================================================
# 主程序
# ============================================================

files = sorted(
    [
        f for f in os.listdir(GT_DIR)
        if f.lower().endswith(
            IMG_EXTS
        )
    ]
)

print("================================")
print("发现图像数量:", len(files))
print("================================")

total_count = 0

for idx, name in enumerate(files):

    path = os.path.join(
        GT_DIR,
        name
    )

    img = Image.open(path)

    # 推荐灰度训练
    if img.mode != "L":
        img = img.convert("L")

    stem = Path(name).stem

    # ====================================================
    # 固定增强
    # ====================================================

    augs = basic_augment(img)

    for aug_name, aug_img in augs:

        save_name = (
            f"{stem}_{aug_name}.png"
        )

        save_image(
            aug_img,
            os.path.join(
                OUT_DIR,
                save_name
            )
        )

        total_count += 1

    # ====================================================
    # 随机仿射增强
    # ====================================================

    for k in range(AFFINE_NUM):

        aug_img = random_affine(img)

        save_name = (
            f"{stem}_affine_{k:03d}.png"
        )

        save_image(
            aug_img,
            os.path.join(
                OUT_DIR,
                save_name
            )
        )

        total_count += 1

    print(
        f"[{idx+1}/{len(files)}] "
        f"{name} 完成"
    )

print("\n================================")
print("GT增强完成")
print("输出数量:", total_count)
print("保存路径:", OUT_DIR)
print("================================")