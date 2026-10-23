import os
import numpy as np
import argparse
from tqdm import tqdm

# ========== 从训练脚本复用所有核心组件 ==========
from train_pao import (
    ForecastingPaOEnv,
    load_real_processed_dataset,
    calc_real_physical_settlement,
    set_env_seed,
    PENALTY_SHORTAGE_RATIO,
    PENALTY_SURPLUS_RATIO,
)
from optimization_model import MILPOpt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

import itertools


# ==============================================================
# 🎯 批量评估配置区（直接改这里就行）
# ==============================================================
# 1. 要评估的预测目标（支持多个，批量遍历）
TARGETS = ["pv"]                  # 可选: "price", "load", "pv"


# 2. 要遍历的超参数列表（自动笛卡尔积生成所有组合）
BETAS =       [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
BETAS =       [0.90]

LRS =         [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]  # 可同时扫多个学习率，如 [1e-05, 5e-05, 1e-04]
LRS =         [5e-5]  # 可同时扫多个学习率，如 [1e-05, 5e-05, 1e-04]

ENT_COEFS =   [0.00]   # 可同时扫多个熵系数，如 [0.00, 0.02]
NORM_OBS_LIST = [1]     # 可同时扫是否归一化观测，如 [0, 1]

# 3. 要评估的种子列表（支持多个，自动跨seed平均）
SEEDS = [42, 100, 888, 2001, 2026]

# 4. 模型检查点选择
MODEL_CHECKPOINT = "final"       # 可选: "best" 或 "final"

# 5. 其他固定配置
MODEL_TYPE = "milp"
DT = 2.0
SCHEDULE_DAYS = 1.0
MSE_NORM_METHOD = "sqrt"
USE_REAL_DATA = 1
DATA_SOURCE_NAME = "real"
EXP_TYPE = "batch"

# ==============================================================
# 单组参数单seed评估函数
# ==============================================================
def run_single_eval(target, beta, lr, ent_coef, norm_obs, seed):
    HORIZON_F = int(SCHEDULE_DAYS * 24 / DT)
    DATASET_DIR_SUFFIX = "real" if USE_REAL_DATA == 1 else "simu"

    # ========== 路径 ==========
    data_dir = f"./0_dataset_{DATASET_DIR_SUFFIX}_processed"
    test_npz_path = os.path.join(data_dir, f"test_dt{DT:.1f}h.npz")
    norm_bounds_path = os.path.join(data_dir, "train_norm_bounds.npz")
    pca_model_dir = f"./1_pca_{DATASET_DIR_SUFFIX}_{MODEL_TYPE}_auto"

    model_root = (
        f"./2_exp_{DATA_SOURCE_NAME}_{EXP_TYPE}_{MSE_NORM_METHOD}/"
        f"model_{MODEL_TYPE}/"
        f"lr_{lr:.0e}/ent_{ent_coef:.2f}/normobs_{int(norm_obs)}/"
        f"target_{target}/beta_{beta:.2f}/seed_{seed}/"
    )

    if MODEL_CHECKPOINT == "best":
        model_path = os.path.join(model_root, "best_model", "best_model.zip")
    else:
        model_path = os.path.join(model_root, f"pred_ppo_seed{seed}_final.zip")
    vec_norm_path = os.path.join(model_root, "vec_normalize.pkl")

    # ========== 加载数据 ==========
    test_data = load_real_processed_dataset(test_npz_path, dt=DT)
    price_all, pv_ub_all, load_all, weather_all = test_data
    train_bounds = np.load(norm_bounds_path)["norm_bounds"].astype(np.float32)

    # ========== 初始化环境 ==========
    env = ForecastingPaOEnv(
        data_tuple=test_data,
        beta=beta,
        target=target,
        norm_bounds=train_bounds,
        model_type=MODEL_TYPE,
        dt=DT,
        horizon_F=HORIZON_F,
        seed=seed + 10000,
        pca_model_dir=pca_model_dir,
        mse_norm_method=MSE_NORM_METHOD,
        start_date="2019-01-01"
    )

    # 加载归一化统计量
    if os.path.exists(vec_norm_path) and norm_obs:
        # 用 DummyVecEnv 包裹一层以兼容 VecNormalize
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_obs = True
        env.norm_reward = True

    # ========== 加载模型 ==========
    if not os.path.exists(model_path):
        print(f"  ⚠️ 模型不存在，跳过: {model_path}")
        return None
    model = PPO.load(model_path, env=env, device="cpu")

    # ========== 滚动评估 ==========
    obs = env.reset()
    done = False
    day_idx = 0

    daily_costs = []
    daily_mae = []
    daily_success = []
    # ✅ 新增：初始化每日最优解存储列表
    daily_solutions = []
    # ✅ 新增：每日时序数据存储
    daily_pred_pv = []      # 预测光伏功率（每天12点）
    daily_true_pv = []      # 真实光伏功率（每天12点）
    daily_true_load = []    # 真实负荷功率（每天12点）
    daily_true_price = []   # 真实电价（每天12点）

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_vec, info = env.step(action)
        
        # 兼容 VecEnv 和原生环境
        if isinstance(info, (list, tuple)):
            info_dict = info[0]
            done = bool(done_vec[0])
        else:
            info_dict = info
            done = bool(done_vec)

        real_cost = info_dict["real_cost"]
        pred_physical = info_dict["pred_physical"]
        solve_success = info_dict["solve_success"]
        # ✅ 新增：保存当天的完整最优解（micro_schedule 就是求解器输出的完整解字典）
        daily_solutions.append(info_dict["micro_schedule"])
        # ✅ 新增：保存当日完整时序数据
        # 注：当 target="pv" 时，pred_physical 就是预测的光伏功率序列
        daily_pred_pv.append(pred_physical.copy())
        daily_true_pv.append(info_dict["true_pv"].copy())
        daily_true_load.append(info_dict["true_load"].copy())
        daily_true_price.append(info_dict["true_price"].copy())

        # 计算物理量纲 MAE
        t_start = day_idx * HORIZON_F
        t_end = t_start + HORIZON_F
        if target == "price":
            true_vals = price_all[t_start:t_end]
        elif target == "load":
            true_vals = load_all[t_start:t_end]
        else:
            true_vals = pv_ub_all[t_start:t_end]
        
        mae = float(np.mean(np.abs(pred_physical - true_vals)))

        daily_costs.append(real_cost)
        daily_mae.append(mae)
        daily_success.append(solve_success)
        day_idx += 1

    env.close()

    return {
        "seed": seed,
        "target": target,
        "beta": beta,
        "lr": lr,
        "ent_coef": ent_coef,
        "norm_obs": norm_obs,
        "daily_costs": np.array(daily_costs, dtype=np.float64),
        "daily_mae": np.array(daily_mae, dtype=np.float64),
        "daily_success": np.array(daily_success, dtype=bool),
        "daily_solutions": daily_solutions,  # ✅ 新增：每日完整解列表
        # ✅ 新增：每日时序数据
        "daily_pred_pv": np.array(daily_pred_pv, dtype=np.float64),  # shape: (365, 12)
        "daily_true_pv": np.array(daily_true_pv, dtype=np.float64),  # shape: (365, 12)
        "daily_true_load": np.array(daily_true_load, dtype=np.float64), # shape: (365, 12)
        "daily_true_price": np.array(daily_true_price, dtype=np.float64), # shape: (365, 12)
        "total_cost": float(np.sum(daily_costs)),
        "mean_mae": float(np.mean(daily_mae)),
        "success_rate": float(np.mean(daily_success)),
        "num_days": day_idx,
    }


# ==============================================================
# 结果聚合：跨seed统计
# ==============================================================
def aggregate_seed_results(results_list):
    if not results_list:
        return None
    
    # 逐日跨seed平均
    daily_costs_stack = np.stack([r["daily_costs"] for r in results_list], axis=0)
    daily_mae_stack = np.stack([r["daily_mae"] for r in results_list], axis=0)
    
    agg = {
        "num_seeds": len(results_list),
        "seeds": [r["seed"] for r in results_list],
        # 逐日统计
        "daily_cost_mean": np.mean(daily_costs_stack, axis=0),
        "daily_cost_std": np.std(daily_costs_stack, axis=0),
        "daily_mae_mean": np.mean(daily_mae_stack, axis=0),
        "daily_mae_std": np.std(daily_mae_stack, axis=0),
        # 年度统计
        "total_cost_mean": float(np.mean([r["total_cost"] for r in results_list])),
        "total_cost_std": float(np.std([r["total_cost"] for r in results_list])),
        "mean_mae_mean": float(np.mean([r["mean_mae"] for r in results_list])),
        "mean_mae_std": float(np.std([r["mean_mae"] for r in results_list])),
        "success_rate_mean": float(np.mean([r["success_rate"] for r in results_list])),
        # 单seed原始数据
        "per_seed": results_list,
    }
    return agg

# ==============================================================
# 工具：单seed每日数据导出CSV（长表结构：day_num + name_data + t1~t12）
# ==============================================================
def save_seed_daily_csv(seed_result, output_dir):
    """
    长表格式：每行 = 某天 + 某个变量
    day_num 重复，name_data 列存变量名，t1~t12 列存对应时刻数值
    """
    import pandas as pd
    
    target = seed_result["target"]
    beta = seed_result["beta"]
    lr = seed_result["lr"]
    ent_coef = seed_result["ent_coef"]
    norm_obs = seed_result["norm_obs"]
    seed = seed_result["seed"]
    
    n_days = seed_result["num_days"]
    horizon = len(seed_result["daily_pred_pv"][0])  # 12个时刻
    n_cluster = len(seed_result["daily_solutions"][0]["E_m"])  # 2个集群
    
    rows = []
    t_cols = [f"t{i+1}" for i in range(horizon)]
    
    for day in range(n_days):
        day_num = day + 1
        sol = seed_result["daily_solutions"][day]
        p_ch_m = sol["p_ch_m"]
        p_dis_m = sol["p_dis_m"]
        E_m = sol["E_m"]
        
        # ========== 1. 时序变量：每个变量占一行 ==========
        # 预测光伏
        rows.append({
            "day_num": day_num,
            "name_data": "pred_pv",
            **dict(zip(t_cols, seed_result["daily_pred_pv"][day]))
        })
        # 真实光伏
        rows.append({
            "day_num": day_num,
            "name_data": "true_pv",
            **dict(zip(t_cols, seed_result["daily_true_pv"][day]))
        })
        # 真实负荷
        rows.append({
            "day_num": day_num,
            "name_data": "true_load",
            **dict(zip(t_cols, seed_result["daily_true_load"][day]))
        })
        # 真实电价
        rows.append({
            "day_num": day_num,
            "name_data": "true_price",
            **dict(zip(t_cols, seed_result["daily_true_price"][day]))
        })
        # 电网购电功率
        rows.append({
            "day_num": day_num,
            "name_data": "opt_P_grid",
            **dict(zip(t_cols, sol["P_grid"]))
        })
        # 光伏调度功率
        rows.append({
            "day_num": day_num,
            "name_data": "opt_P_pv",
            **dict(zip(t_cols, sol["P_pv"]))
        })
        # 集群1 充电功率
        rows.append({
            "day_num": day_num,
            "name_data": "unit1_P_ch",
            **dict(zip(t_cols, p_ch_m[0]))
        })
        # 集群1 放电功率
        rows.append({
            "day_num": day_num,
            "name_data": "unit1_P_dis",
            **dict(zip(t_cols, p_dis_m[0]))
        })
        # 集群1 电池能量
        rows.append({
            "day_num": day_num,
            "name_data": "unit1_E",
            **dict(zip(t_cols, E_m[0]))
        })
        # 集群2 充电功率
        rows.append({
            "day_num": day_num,
            "name_data": "unit2_P_ch",
            **dict(zip(t_cols, p_ch_m[1]))
        })
        # 集群2 放电功率
        rows.append({
            "day_num": day_num,
            "name_data": "unit2_P_dis",
            **dict(zip(t_cols, p_dis_m[1]))
        })
        # 集群2 电池能量
        rows.append({
            "day_num": day_num,
            "name_data": "unit2_E",
            **dict(zip(t_cols, E_m[1]))
        })
        
        # ========== 2. 单日标量变量：也各占一行，t列统一填该值 ==========
        rows.append({
            "day_num": day_num,
            "name_data": "solve_success",
            **dict(zip(t_cols, [seed_result["daily_success"][day]] * horizon))
        })
        rows.append({
            "day_num": day_num,
            "name_data": "daily_cost",
            **dict(zip(t_cols, [seed_result["daily_costs"][day]] * horizon))
        })
        rows.append({
            "day_num": day_num,
            "name_data": "daily_mae",
            **dict(zip(t_cols, [seed_result["daily_mae"][day]] * horizon))
        })
    
    df = pd.DataFrame(rows)
    # 列顺序固定
    df = df[["day_num", "name_data"] + t_cols]
    
    # 文件名
    filename = (
        f"daily_long_{target}_beta{beta:.2f}_lr{lr:.0e}"
        f"_ent{ent_coef:.2f}_norm{int(norm_obs)}_seed{seed}.csv"
    )
    out_path = os.path.join(output_dir, filename)
    df.to_csv(out_path, index=False, float_format="%.6f")



# ==============================================================
# 主函数：批量遍历所有参数组合
# ==============================================================
def main():
    print("=" * 70)
    print(f"📊 PaO 批量评估 | 目标: {TARGETS}")
    param_total = len(BETAS) * len(LRS) * len(ENT_COEFS) * len(NORM_OBS_LIST)
    print(f"   参数组合数: {param_total} | 种子数: {len(SEEDS)}")
    print("=" * 70)


    output_root = "./3_eval_results"
    output_root = "./3_eval_results_best"
    os.makedirs(output_root, exist_ok=True)

    for target in TARGETS:
        # 自动生成所有参数组合（笛卡尔积）
        param_combinations = [
            {"beta": b, "lr": lr, "ent_coef": e, "norm_obs": n}
            for b, lr, e, n in itertools.product(BETAS, LRS, ENT_COEFS, NORM_OBS_LIST)
        ]
        
        for param in param_combinations:
            beta = param["beta"]
            lr = param["lr"]
            ent_coef = param["ent_coef"]
            norm_obs = param["norm_obs"]

            print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"📍 Target: {target.upper()} | β={beta} | LR={lr:.0e} | ent={ent_coef:.2f} | norm_obs={norm_obs}")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            # 跑所有 seed
            seed_results = []
            for seed in tqdm(SEEDS, desc="  评估种子"):
                res = run_single_eval(target, beta, lr, ent_coef, norm_obs, seed)
                if res is not None:
                    seed_results.append(res)

            if not seed_results:
                print("  ❌ 所有seed均无有效结果，跳过")
                continue

            # 跨seed聚合
            agg = aggregate_seed_results(seed_results)

            # 打印汇总
            print(f"\n  📈 跨 {agg['num_seeds']} 个seed 汇总:")
            print(f"     年总成本 均值: {agg['total_cost_mean']:.4e} ± {agg['total_cost_std']:.4e}")
            print(f"     预测MAE 均值: {agg['mean_mae_mean']:.4f} ± {agg['mean_mae_std']:.4f}")
            print(f"     优化成功率: {agg['success_rate_mean']*100:.2f}%")

            # 保存结果
            param_tag = f"{target}_beta{beta:.2f}_lr{lr:.0e}_ent{ent_coef:.2f}_norm{norm_obs}"
            out_path = os.path.join(output_root, f"eval_{param_tag}.npz")
            np.savez_compressed(out_path, **agg)
            print(f"  💾 结果已保存: {out_path}")

            # ✅ 新增：逐seed导出每日CSV
            csv_dir = os.path.join(output_root, "daily_csvs")
            os.makedirs(csv_dir, exist_ok=True)
            for seed_res in agg["per_seed"]:
                save_seed_daily_csv(seed_res, csv_dir)
            print(f"  💾 每日CSV已保存到: {csv_dir}")


    print("\n" + "=" * 70)
    print("🎉 全部评估完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
