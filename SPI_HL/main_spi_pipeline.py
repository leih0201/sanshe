# -*- coding: utf-8 -*-
from typing import Optional, Dict, Any
from dataclasses import asdict
import numpy as np
import os

# 导入加速重构工具
from spi_tval3_functions_v3 import (
    ForwardParams,
    TVAL3Opts,
    RunParams,
    run_pipeline,
)

# 导入物理模拟器组件
from scattering_model_v3 import (
    SyntheticScatteringSimulatorV3,
    NoiseModelParams,
)

if __name__ == "__main__":
    # ========================================================
    # 1. 核心路径与数据集规模配置 (13600K + 5060 专用)
    # ========================================================
    run_params = RunParams(
        # [路径] 投影矩阵文件路径 (.csv 或 .npy)
        pattern_path=r"F:\code\wangluo\wt_futong\rate_0p0400\A_01.csv",         
        # [路径] 原始 GT 清晰图所在文件夹 
        input_path=r"F:\code\wangluo\wt_futong\T_images\GT_kuochong",                   
        # [路径] 结果输出的总目录 
        output_root=r"F:\code\wangluo\wt_futong\T_results",                    
        
        # [核心] 每张输入图生成 100 张变体图
        num_repeats=100,
        resolution=256,
        
        # [命名] 父文件夹名
        final_folder_name="T_ONE_140",       
        debug_folder_name="T_ONE_COP",       
        
        random_seed=1234,
        save_scatter_debug=False,
        save_counts_csv=False
    )

    # ========================================================
    # 2. SPAD 硬件仿真参数
    # ========================================================
    forward_params = ForwardParams(
        DMD_frame_rate=65.0,                  
        dead_time=1e-7,                          
        photon_count_rate=3010720.0,             
        dark_count=700.0,                        
        detection_efficiency=0.8,                
        environmental_noise=0.02,                
        afterpulsing_prob=0.001                
    )

    # ========================================================
    # 3. 物理模拟随机波动区间 (控制退化的“程度”)
    # ========================================================
    scattering_simulator = SyntheticScatteringSimulatorV3(
        t_range=(0.15, 0.85),        # 透射率：值越小雾越浓
        A_range=(70.0, 115.0),       # 大气光强度：决定画面泛白程度
        blur_sigma_range=(1.5, 4.0), # 模糊核大小：模拟前向散射造成的模糊
        illum_gain_range=(0.55, 0.7),
        illum_shape_strength=0.3,    
        noise_params=NoiseModelParams(c0=1.0, c1=3.0, c2=4.0, c3=2.0, gain_power=0.5)
    )

    # ========================================================
    # 4. 微调控制 (将 jitter 设为 None 以开启随机化)
    # ========================================================
    scatter_sim_kwargs = {
        "signal_gain": 1.20,
        "noise_gain": 0.75,          
        "target_mean": 60.0,         # 自动均衡平均亮度 (防过曝/过暗)
        "output_gain": 0.90,
        "illum_blend": 0.70,
        
        # 🚨 设为 None 即可让系统在 100 次循环中，每次随机从上面的区间抽取参数
        "jitter_t": None,            
        "jitter_sigma": None,        
        "jitter_A": None,
    }

    # ========================================================
    # 5. TVAL3 重构参数 (GPU 加速下迭代依然极快)
    # ========================================================
    recon_opts = TVAL3Opts(
        TVL2=True,
        mu=2**8, 
        beta=2**5, 
        tol=1e-4,         
        maxit=50, 
        disp=False
    )

    # ========================================================
    # 6. 启动 GPU 加速流水线
    # ========================================================
    print(f"[DATASET PRODUCTION] 启动 RTX 5060 加速数据集生成...")
    print(f"模式：每张图片将生成专属的 'GT名_beat' 文件夹，并生成 {run_params.num_repeats} 张子图")
    
    run_pipeline(
        run_params=run_params,
        forward_params=forward_params,
        recon_opts=recon_opts,
        scattering_simulator=scattering_simulator,
        scatter_sim_kwargs=scatter_sim_kwargs
    )