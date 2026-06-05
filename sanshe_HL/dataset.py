# ==========================================
# dataset.py
# 散射图像恢复训练数据集
#
# 数据结构：
#
# data/
# ├── labels/
# │   ├── 001.bmp
# │   ├── 002.bmp
# │   └── ...
# │
# └── train/
#     ├── 001/
#     │   ├── img1.bmp
#     │   ├── img2.bmp
#     │   └── ...
#     │
#     ├── 002/
#     │   ├── img1.bmp
#     │   └── ...
#     │
#     └── ...
#
# 每个 train 子文件夹对应一个 label
# ==========================================

import os

from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ScatterDataset(Dataset):

    def __init__(
            self,
            root_dir,
            image_size=256,
            verbose=True
    ):
        """
        Parameters
        ----------
        root_dir : str
            数据集根目录

        image_size : int
            网络输入尺寸

        verbose : bool
            是否打印数据集信息
        """

        self.root_dir = root_dir

        self.train_dir = os.path.join(
            root_dir,
            "train"
        )

        self.label_dir = os.path.join(
            root_dir,
            "labels"
        )

        # ---------------------------
        # 检查目录
        # ---------------------------

        if not os.path.exists(self.train_dir):
            raise FileNotFoundError(
                f"Train folder not found:\n{self.train_dir}"
            )

        if not os.path.exists(self.label_dir):
            raise FileNotFoundError(
                f"Label folder not found:\n{self.label_dir}"
            )

        # ---------------------------
        # 图像预处理
        # ---------------------------

        self.transform = transforms.Compose([

            transforms.Resize(
                (image_size, image_size)
            ),

            transforms.ToTensor()

        ])

        self.samples = []

        supported_suffix = (
            ".bmp",
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff"
        )

        label_files = sorted(
            os.listdir(self.label_dir)
        )

        # ---------------------------
        # 构建样本列表
        # ---------------------------

        for label_file in label_files:

            label_name = os.path.splitext(
                label_file
            )[0]

            label_path = os.path.join(
                self.label_dir,
                label_file
            )

            train_subfolder = os.path.join(
                self.train_dir,
                label_name
            )

            if not os.path.isdir(train_subfolder):

                print(
                    f"[Warning] Missing folder:"
                    f" {train_subfolder}"
                )

                continue

            train_imgs = sorted(
                os.listdir(train_subfolder)
            )

            for img_name in train_imgs:

                if not img_name.lower().endswith(
                        supported_suffix):
                    continue

                noisy_path = os.path.join(
                    train_subfolder,
                    img_name
                )

                self.samples.append(
                    (
                        noisy_path,
                        label_path
                    )
                )

        if len(self.samples) == 0:
            raise RuntimeError(
                "No training samples found."
            )

        # ---------------------------
        # 输出统计信息
        # ---------------------------

        if verbose:

            print("\n========== Dataset Info ==========")

            print(
                f"Dataset Root : {root_dir}"
            )

            print(
                f"Input Size   : {image_size}"
            )

            print(
                f"Labels       : {len(label_files)}"
            )

            print(
                f"Samples      : {len(self.samples)}"
            )

            print("==================================\n")

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        noisy_path, label_path = self.samples[idx]

        try:

            noisy_img = Image.open(
                noisy_path
            ).convert("L")

            label_img = Image.open(
                label_path
            ).convert("L")

        except Exception as e:

            raise RuntimeError(
                f"\nImage Read Error:\n"
                f"{noisy_path}\n"
                f"{label_path}\n"
                f"{e}"
            )

        noisy_img = self.transform(
            noisy_img
        )

        label_img = self.transform(
            label_img
        )

        return noisy_img, label_img