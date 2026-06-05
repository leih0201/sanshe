# -*- coding: utf-8 -*-
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# ============================================================
# 参数定义
# ============================================================
@dataclass
class ScatterParams:
    t: float
    A: float
    blur_sigma: float

@dataclass
class NoiseModelParams:
    c0: float
    c1: float
    c2: float
    c3: float
    gain_power: float = 0.5

# ============================================================
# 基础工具
# ============================================================
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def normalize_field_mean1(field: np.ndarray) -> np.ndarray:
    m = float(np.mean(field))
    return field / max(m, 1e-6)

# ============================================================
# 光照生成与渲染逻辑
# ============================================================
def synthetic_illumination_components(H, W, rng, shape_strength=0.25):
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = rng.integers(0, W), rng.integers(0, H)
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    dist /= (dist.max() + 1e-6)
    radial = np.exp(-2.5 * dist)
    shape = normalize_field_mean1(1.0 - shape_strength + shape_strength * radial)
    shape = gaussian_filter(shape, sigma=min(H, W)/12.0)
    return 1.0, shape.astype(np.float32), shape.astype(np.float32)

def render_clean_scattering(gt_img: np.ndarray, params: ScatterParams, illum_gain: float = 1.0, 
                            illum_shape: Optional[np.ndarray] = None, signal_gain: float = 1.0):
    gt = gt_img.astype(np.float32)
    gt_blur = gaussian_filter(gt, params.blur_sigma) if params.blur_sigma > 1e-8 else gt.copy()
    
    # 物理散射公式
    useful_signal = signal_gain * params.t * gt_blur
    airlight = (1.0 - params.t) * params.A
    clean_deill = useful_signal + airlight

    if illum_shape is None: illum_shape = np.ones_like(gt)
    full_field = illum_gain * illum_shape
    clean = np.clip(clean_deill * full_field, 0, 255)
    return clean, clean_deill, full_field

# ============================================================
# 无参考图随机模拟器
# ============================================================
class SyntheticScatteringSimulatorV3:
    def __init__(self, t_range=(0.15, 0.85), A_range=(20.0, 180.0), blur_sigma_range=(2.5, 3.5),
                 illum_gain_range=(0.55, 1.0), illum_shape_strength=0.25, noise_params=None):
        self.t_range = t_range
        self.A_range = A_range
        self.blur_sigma_range = blur_sigma_range
        self.illum_gain_range = illum_gain_range
        self.illum_shape_strength = illum_shape_strength
        self.noise_params = noise_params or NoiseModelParams(1.0, 3.0, 4.0, 2.0, 0.5)

    def simulate(self, gt_img: np.ndarray, rng: Optional[np.random.Generator] = None, **kwargs):
        if rng is None: rng = np.random.default_rng()

        # 🚨 安全校验逻辑：当传进来的参数是 None 时，立刻转入随机区间抽取
        t_val = kwargs.get("jitter_t")
        t = float(t_val) if t_val is not None else float(rng.uniform(*self.t_range))
        
        A_val = kwargs.get("jitter_A")
        A = float(A_val) if A_val is not None else float(rng.uniform(*self.A_range))
        
        s_val = kwargs.get("jitter_sigma")
        sigma = float(s_val) if s_val is not None else float(rng.uniform(*self.blur_sigma_range))

        scatter_params = ScatterParams(t=t, A=A, blur_sigma=sigma)
        H, W = gt_img.shape
        
        illum_gain = float(rng.uniform(*self.illum_gain_range)) * kwargs.get("illum_gain_scale", 1.0)
        _, illum_shape, _ = synthetic_illumination_components(H, W, rng, self.illum_shape_strength)

        clean, clean_deill, full_field = render_clean_scattering(
            gt_img, scatter_params, illum_gain=illum_gain, illum_shape=illum_shape, 
            signal_gain=kwargs.get("signal_gain", 1.0)
        )

        target_mean = kwargs.get("target_mean")
        if target_mean:
            scale = float(target_mean) / max(float(clean.mean()), 1e-6)
            clean *= scale
        
        degraded = np.clip(clean, 0, 255).astype(np.float32)
        
        # 抛出准确的物理参数供外界命名文件使用
        meta = {
            "t": t, 
            "A": A, 
            "blur": sigma,
            "scatter_params_used": asdict(scatter_params),
        }
        return degraded, meta, {"clean": clean, "illum_full": full_field}