# ============================================================
# infer.py
#
# 利用训练好的 UNet 模型进行散射图像恢复
#
# Compatible:
# RTX5060
# RTX5070
# RTX5080
# RTX5090
# RTX A6000
# RTX4090
#
# Author: HanLei Version
# ============================================================

import os
from pathlib import Path

import torch
import torchvision.transforms as transforms

from PIL import Image

from network import UNet


# ============================================================
# 参数区（只需要修改这里）
# ============================================================

# 模型路径
MODEL_PATH = r"F:\code\wangluo\wt_futong\modles\best_model.pth"

# 输入路径
# 可以是单张图片
# 也可以是整个文件夹
INPUT_PATH = r"F:\code\wangluo\wt_futong\ceui\输入"

# 输出路径
OUTPUT_DIR = r"F:\code\wangluo\wt_futong\ceui\输出"

# 网络输入尺寸
IMAGE_SIZE = 256

# ============================================================


def load_model(model_path, device):

    model = UNet()

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    # 自动兼容 DataParallel
    new_state_dict = {}

    for k, v in state_dict.items():

        if k.startswith("module."):

            k = k[7:]

        new_state_dict[k] = v

    model.load_state_dict(
        new_state_dict
    )

    model.to(device)

    model.eval()

    return model


def preprocess(image_path):

    img = Image.open(image_path).convert("L")

    original_size = img.size

    transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor()
    ])

    tensor = transform(img)

    tensor = tensor.unsqueeze(0)

    return tensor, original_size


def postprocess(output_tensor,
                original_size):

    output = output_tensor.squeeze()

    output = output.detach().cpu().numpy()

    output = output * 255.0

    output = output.clip(
        0,
        255
    ).astype("uint8")

    img = Image.fromarray(output)

    img = img.resize(
        original_size,
        Image.BILINEAR
    )

    return img


def process_one_image(
        model,
        image_path,
        save_path,
        device):

    # -----------------------------------------
    # 图像预处理
    # -----------------------------------------
    tensor, original_size = preprocess(
        image_path
    )

    tensor = tensor.to(device)

    # -----------------------------------------
    # 推理
    # -----------------------------------------
    with torch.no_grad():

       if device.type == "cuda":

           with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16):

               output = model(tensor)

       else:

            output = model(tensor)

    # -----------------------------------------
    # 后处理
    # -----------------------------------------
    result = postprocess(
        output,
        original_size
    )

    # -----------------------------------------
    # 保存结果
    # -----------------------------------------
    result.save(save_path)

    print(
        f"[OK] {image_path}"
    )


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # GPU检测
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)

    print("Device :", device)

    if torch.cuda.is_available():

        print(
            "GPU Name :",
            torch.cuda.get_device_name(0)
        )

    print("=" * 60)

    # --------------------------------------------------------
    # 加载模型
    # --------------------------------------------------------

    model = load_model(
        MODEL_PATH,
        device
    )

    # --------------------------------------------------------
    # 单张图像
    # --------------------------------------------------------

    if os.path.isfile(INPUT_PATH):

        filename = os.path.basename(
            INPUT_PATH
        )

        save_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        process_one_image(
            model,
            INPUT_PATH,
            save_path,
            device
        )

    # --------------------------------------------------------
    # 文件夹批量推理
    # --------------------------------------------------------

    else:

        exts = (
            ".bmp",
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff"
        )

        image_list = []

        for root, dirs, files in os.walk(
                INPUT_PATH):

            for file in files:

                if file.lower().endswith(exts):

                    image_list.append(
                        os.path.join(
                            root,
                            file
                        )
                    )

        print(
            f"Find {len(image_list)} images"
        )

        for image_path in image_list:

            filename = Path(
                image_path
            ).name

            save_path = os.path.join(
                OUTPUT_DIR,
                filename
            )

            process_one_image(
                model,
                image_path,
                save_path,
                device
            )

    print("\nInference Finished.")


if __name__ == "__main__":
    main()