# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

# 🚀 引入 CuPy 加速 (RTX 5060)
import cupy as cp

# ============================================================
# 参数类 (完全向后兼容，防止 TypeError)
# ============================================================
@dataclass
class ForwardParams:
    DMD_frame_rate: float = 65.0
    dead_time: float = 1e-7
    photon_count_rate: float = 3010720.0
    dark_count: float = 700.0
    detection_efficiency: float = 0.8
    environmental_detection_efficiency: float = 0.8
    environmental_noise: float = 0.02
    afterpulsing_prob: float = 0.001

@dataclass
class TVAL3Opts:
    TVL2: bool = True
    mu: float = 2 ** 8
    beta: float = 2 ** 5
    tol: float = 1e-4
    tol_inn: float = 1e-3
    maxcnt: int = 10
    maxit: int = 50
    init: int = 1
    disp: bool = False
    scale_A: bool = True
    scale_b: bool = True
    consist_mu: bool = False

@dataclass
class RunParams:
    input_path: str
    pattern_path: str
    output_root: str
    resolution: int = 256             # 恢复分辨率参数
    recursive: bool = True
    num_repeats: int = 100            # 默认生成 100 张
    random_seed: int = 1234
    save_counts_csv: bool = False
    save_counts_npy: bool = False
    save_original: bool = True
    save_metadata: bool = False
    save_scattered: bool = True
    save_scatter_debug: bool = False
    reconstruct_after_measurement: bool = True
    save_polar_products: bool = False
    final_folder_name: str = "train_dianchi"
    debug_folder_name: str = "labels_dianchi"

# ============================================================
# GPU 加速 A 算子
# ============================================================
class CuPyMatrixAOperator:
    def __init__(self, A: np.ndarray):
        self.d_A = cp.asarray(A, dtype=cp.float32)

    def __call__(self, u, mode):
        u_gpu = cp.asarray(u, dtype=cp.float32).reshape(-1)
        if mode == 1:
            res = self.d_A @ u_gpu
        else:
            res = self.d_A.T @ u_gpu
        return res

# ============================================================
# 核心处理函数 (精准处理文件夹与文件命名)
# ============================================================
def process_one_image(image_path, patterns, forward_params, recon_opts, run_params, rng, scattering_simulator, scatter_sim_kwargs):
    gt_full_name = os.path.basename(image_path)
    gt_base_name = os.path.splitext(gt_full_name)[0]

    # 1. 建立 Labels 备份目录
    label_dir = os.path.join(run_params.output_root, run_params.debug_folder_name)
    os.makedirs(label_dir, exist_ok=True)
    
    
   # 2. 建立 Train 数据集目录
   # 文件夹名称与GT完全一致
    folder_name = gt_base_name
    dataset_dir = os.path.join(run_params.output_root,run_params.final_folder_name,folder_name)
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"\n[RUNNING] 正在全力处理: "f"{gt_base_name} -> 存放于: "f"{folder_name}/")


    # 读取并处理原图
    gt = np.array(Image.open(image_path).convert("L").resize((run_params.resolution, run_params.resolution)), dtype=np.float32)
    Image.fromarray(gt.astype(np.uint8)).save(os.path.join(label_dir, f"{gt_base_name}.bmp"))

    # 生成 100 张变体
    for rep in range(run_params.num_repeats):
        degraded, meta, _ = scattering_simulator.simulate(gt_img=gt, rng=rng, **scatter_sim_kwargs)
        
        # 🚀 提取物理参数，精确拼接文件名 (T_001_beat_t0.150_A100.0_blur2.0.bmp)
        t_used = meta.get("t", 0.5)
        A_used = meta.get("A", 100.0)
        blur_used = meta.get("blur", 2.0)
        
        file_name = (f"{gt_base_name}_"f"t{t_used:.3f}_"f"A{A_used:.1f}_"f"blur{blur_used:.1f}.bmp")
        save_path = os.path.join(dataset_dir, file_name)
        
        # 将变体图存入硬盘
        Image.fromarray(np.clip(degraded, 0, 255).astype(np.uint8)).save(save_path)
        
        if (rep + 1) % 10 == 0:
            print(f"  -> {gt_base_name}: 已生成变体 {rep+1}/{run_params.num_repeats}")

def run_pipeline(run_params, forward_params, recon_opts, scattering_simulator, scatter_sim_kwargs):
    os.makedirs(run_params.output_root, exist_ok=True)
    
    print(f"[INFO] 正在向显存推送大矩阵基底 (这可能需要几秒钟)...")
    patterns = pd.read_csv(run_params.pattern_path, header=None).to_numpy(dtype=np.float32)
    print(f"[SYSTEM] 显存底本布设就绪。形状: {patterns.shape}")
    
    exts = ('.bmp', '.png', '.jpg')
    img_paths = [os.path.join(run_params.input_path, f) for f in os.listdir(run_params.input_path) if f.lower().endswith(exts)]
    
    rng = np.random.default_rng(run_params.random_seed)
    
    for path in img_paths:
        process_one_image(path, patterns, forward_params, recon_opts, run_params, rng, scattering_simulator, scatter_sim_kwargs)
    print("\n[SUCCESS] 所有图像变体已生成完毕！")