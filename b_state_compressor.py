import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
from optimization_model import MILPOpt 

def flatten_solution_vector(sol, model_type, num_clusters, horizon):

    h = horizon
    parts = [np.atleast_1d(sol["P_grid"])]
    if model_type in ["milp", "miqp"]:

        for i in range(num_clusters):
            p_net = sol["p_dis_m"][i, :] - sol["p_ch_m"][i, :]
            parts.append(np.atleast_1d(p_net))
        parts.append(np.atleast_1d(sol["P_pv"]))
        for i in range(num_clusters):
            parts.append(np.atleast_1d(sol["E_m"][i, :]))
    else:

        p_net = sol["P_dis"] - sol["P_ch"]
        parts.append(np.atleast_1d(p_net))
        parts.append(np.atleast_1d(sol["P_pv"]))
        parts.append(np.atleast_1d(sol["E_m"][0, :]))
    vec = np.concatenate(parts)
    expected_len = h * (2 + 2 * num_clusters)
    if len(vec) != expected_len:
        raise ValueError(
            f"展平维度错误: 期望 {expected_len}, 实际 {len(vec)} "
            f"model_type={model_type}, num_clusters={num_clusters}, horizon={h}"
        )
    return vec.astype(np.float32)

class SolutionPCACompressor:
    def __init__(self, n_components=0.99, random_state=42):

        self.n_components = n_components
        self.random_state = random_state

        self.scaler = None        
        self.pca = None           
        self.raw_dim = None        
        self.reduced_dim = None   
        self.explained_variance_ratio_ = None   
    def fit(self, X_raw):
        self.raw_dim = X_raw.shape[1]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw)

        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca.fit(X_scaled)
        self.reduced_dim = self.pca.n_components_
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        print(f"PCA拟合完成")
        print(f"   原始维度: {self.raw_dim}")
        print(f"   降维后维度: {self.reduced_dim}")
        print(f"   累计方差解释率: {np.sum(self.explained_variance_ratio_):.4f}")
        return self
    def transform(self, state_vec):
        if self.scaler is None or self.pca is None:
            raise RuntimeError("PCA模型未初始化，请先执行fit()或者load()加载预训练模型")
        is_single_sample = (state_vec.ndim == 1)
        if is_single_sample:
            state_vec = state_vec.reshape(1, -1)
        X_scaled = self.scaler.transform(state_vec)
        X_reduced = self.pca.transform(X_scaled)
        if is_single_sample:
            return X_reduced.flatten().astype(np.float32)
        return X_reduced.astype(np.float32)
    def fit_transform(self, X_raw):
        self.fit(X_raw)
        return self.transform(X_raw)
    def save(self, save_dir, meta_extra=None):
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(save_dir, "scaler.pkl"))
        joblib.dump(self.pca, os.path.join(save_dir, "pca.pkl"))

        meta = {
            "raw_dim": self.raw_dim,
            "reduced_dim": self.reduced_dim,
            "explained_variance_ratio": self.explained_variance_ratio_,
            "total_variance": float(np.sum(self.explained_variance_ratio_)),
        }
        if meta_extra is not None:
            meta.update(meta_extra)
        np.savez(os.path.join(save_dir, "pca_meta.npz"), **meta)
        print(f"   PCA模型已保存至: {save_dir}")
    def load(self, load_dir, expected_meta=None):
        scaler_path = os.path.join(load_dir, "scaler.pkl")
        pca_path = os.path.join(load_dir, "pca.pkl")
        if not os.path.exists(scaler_path) or not os.path.exists(pca_path):
            raise FileNotFoundError(f"PCA模型文件缺失，检查路径: {load_dir}")
        self.scaler = joblib.load(scaler_path)
        self.pca = joblib.load(pca_path)
        self.reduced_dim = self.pca.n_components_
        meta_path = os.path.join(load_dir, "pca_meta.npz")
        if os.path.exists(meta_path):
            meta = np.load(meta_path)
            self.raw_dim = int(meta["raw_dim"])
            self.explained_variance_ratio_ = meta["explained_variance_ratio"]

            if expected_meta is not None:
                for k, expect_val in expected_meta.items():
                    if k not in meta:
                        raise RuntimeError(f"PCA元数据缺失key: {k}，该PCA模型版本过旧，无法做参数校验")
                    disk_val = meta[k]

                    if hasattr(disk_val, "item"):
                        disk_val = disk_val.item()
                    if disk_val != expect_val:
                        raise RuntimeError(
                            f"PCA模型参数不匹配！key={k}，期望={expect_val}，磁盘保存={disk_val}\n"
                            "PCA训练超参和当前环境超参不一致，禁止运行，避免观测维度错乱。"
                        )
        else:

            if expected_meta is not None:
                raise RuntimeError(f"目录 {load_dir} 找不到pca_meta.npz，无法执行参数校验，请重新生成带meta的PCA模型")
        return self

def generate_solution_dataset(
    data_tuple=None,
    load_samples_npz: str = None,
    model_type="milp",
    horizon=12,
    dt=2.0,
    num_clusters=2,
    stride=None,
    seed=42,
    lambda_penalty=0.01,
    aggregate_unit=False,
    save_samples_npz: str = None
):

    if load_samples_npz is not None:
        print(f"直接加载历史解样本，跳过SCIP求解：{load_samples_npz}")
        npz_data = np.load(load_samples_npz)
        X_raw = npz_data["X_raw"]
        print(f"加载样本 | 有效样本:{X_raw.shape[0]}, 单样本维度:{X_raw.shape[1]}")
        if save_samples_npz is not None:
            os.makedirs(os.path.dirname(save_samples_npz), exist_ok=True)
            np.savez_compressed(save_samples_npz, X_raw=X_raw)
            print(f"样本另存到 {save_samples_npz}")
        return X_raw

    price, pv_ub, load, _ = data_tuple
    total_len = len(price)
    if stride is None:
        stride = horizon

    solver_map = {
        "milp": lambda: MILPOpt(horizon=horizon, dt=dt, num_clusters=num_clusters, seed=seed),
    }
    if model_type not in solver_map:
        raise ValueError(f"不支持的model_type={model_type}")
    opt = solver_map[model_type]()
    rng = np.random.default_rng(seed)
    sample_list = []
    t = 0

    while t + horizon <= total_len:

        win_price = price[t:t+horizon]
        win_pv = pv_ub[t:t+horizon]
        win_load = load[t:t+horizon]

        if model_type in ["milp", "miqp"]:
            init_soc = rng.uniform(0.3, 0.7, size=num_clusters)
        else:
            init_soc = np.array([rng.uniform(0.3, 0.7)])
        opt.set_initial_energy(init_soc)
        sol = opt.solve(win_price, win_pv, win_load)

        solve_success = opt.status in ["optimal", "gaplimit"]
        if solve_success:
            if aggregate_unit:

                p_es_net = (sol["p_dis_m"] - sol["p_ch_m"]).sum(axis=0)
                e_total = sol["E_m"].sum(axis=0)
                state_vec = np.concatenate([
                    sol["P_grid"], p_es_net, sol["P_pv"], e_total
                ])
            else:

                state_vec = flatten_solution_vector(sol, model_type, num_clusters, horizon)
            sample_list.append(state_vec)
        t += stride

    opt._safe_close()
    del opt
    if len(sample_list) == 0:
        raise RuntimeError("generate_solution_dataset：没有收集到任何有效最优解样本，请检查数据、SCIP求解器参数！")
    X_raw = np.array(sample_list, dtype=np.float32)

    print(f"generate_solution_dataset | 有效样本:{len(sample_list)}, 单样本维度:{X_raw.shape[1]}")

    if save_samples_npz is not None:
        os.makedirs(os.path.dirname(save_samples_npz), exist_ok=True)
        np.savez_compressed(save_samples_npz, X_raw=X_raw)
        print(f"解样本保存到 {save_samples_npz}")
    return X_raw

def build_pca_pipeline(
    data_tuple=None,
    load_samples_npz: str = None,
    save_dir="./pca_model",
    model_type="milp",
    horizon=12,
    dt=2.0,
    num_clusters=2,
    n_components=0.99,
    stride=None,
    seed=42,
    lambda_penalty=0.01,
    aggregate_unit=False,
    save_samples_npz=None
):

    print("=" * 60)
    print(f"   build_pca_pipeline start")
    print(f"   model_type={model_type}, num_clusters={num_clusters}, aggregate_unit={aggregate_unit}")
    print("=" * 60)

    if num_clusters is None:
        if model_type in ["miqp", "milp"]:
            num_clusters = 2
        else:
            num_clusters = 1
    X_raw = generate_solution_dataset(
        data_tuple=data_tuple,
        load_samples_npz=load_samples_npz,
        model_type=model_type,
        horizon=horizon,
        dt=dt,
        num_clusters=num_clusters,
        stride=stride,
        seed=seed,
        lambda_penalty=lambda_penalty,
        aggregate_unit=aggregate_unit,
        save_samples_npz=save_samples_npz
    )
    compressor = SolutionPCACompressor(n_components=n_components, random_state=seed)
    compressor.fit(X_raw)
    meta_extra = {
        "model_type": model_type,
        "horizon": horizon,
        "num_clusters": num_clusters,
        "aggregate_unit": aggregate_unit,
        "dt": dt
    }
    compressor.save(save_dir, meta_extra=meta_extra)
    print(f"   PCA模型已保存至: {save_dir}")
    print("=" * 60)
    return compressor

if __name__ == "__main__":

    DT = 2.0
    HORIZON = 12
    SEED = 42
    N_COMPONENTS = 0.99
    AGGREGATE_UNIT = False
    # ========== 1. 加载数据集 ==========
    dataset_map = {
        "real": "./0_dataset_real_processed/train_dt2.0h.npz",
    }
    loaded_data = {}
    for tag, path in dataset_map.items():
        data = np.load(path)
        loaded_data[tag] = (
            data["price"],
            data["pv_ub"],
            data["load"],
            data["weather"]
        )
        print(f"已加载 {tag} 数据集，序列长度: {len(data['price'])}")
    # ========== 2. 运行 milp ==========
    model_list = ["milp"]
    for dataset_tag, data_tuple in loaded_data.items():
        print("\n" + "=" * 70)
        print(f"开始处理数据集: {dataset_tag}")
        print("=" * 70)
        for model_type in model_list:
            print(f"\n{'─' * 50}")
            print(f"正在构建: {dataset_tag} - {model_type}")
            print(f"{'─' * 50}")

            save_dir = f"./1_pca_{dataset_tag}_{model_type}_auto"
            sample_cache_path = os.path.join(save_dir, "solution_samples.npz")
            build_pca_pipeline(
                data_tuple=data_tuple,
                save_dir=save_dir,
                model_type=model_type,
                horizon=HORIZON,
                dt=DT,
                num_clusters=None,    
                n_components=N_COMPONENTS,
                seed=SEED,
                aggregate_unit=AGGREGATE_UNIT,
                save_samples_npz=sample_cache_path
            )
    # ========== 3. 运行完成汇总 ==========
    print("\n" + "=" * 70)
    print("全部 PCA 模型构建完成！输出目录清单：")
    for dataset_tag in loaded_data.keys():
        for model_type in model_list:
            print(f"  ./1_pca_{dataset_tag}_{model_type}_auto")
    print("=" * 70)
    # srun -p AMD -N 1 -c 4 --mem=16G --pty /bin/bash
