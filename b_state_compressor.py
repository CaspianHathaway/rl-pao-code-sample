import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
from optimization_model import MILPOpt  # ✅ 修改：从optimization_model导入SCIP版MILPOpt

def flatten_solution_vector(sol, model_type, num_clusters, horizon):
    """
    统一的调度解展平函数，离线PCA训练和在线环境必须完全共用这一套
    拼接顺序（aggregate_unit=False）：
        P_grid(H) + net_power_unit0(H) + ... + P_pv(H) + E_unit0(H) + ...
    """
    h = horizon
    parts = [np.atleast_1d(sol["P_grid"])]
    if model_type in ["milp", "miqp"]:
        # 多单元：使用分簇变量 p_dis_m / p_ch_m / E_m，shape=(M, H)
        for i in range(num_clusters):
            p_net = sol["p_dis_m"][i, :] - sol["p_ch_m"][i, :]
            parts.append(np.atleast_1d(p_net))
        parts.append(np.atleast_1d(sol["P_pv"]))
        for i in range(num_clusters):
            parts.append(np.atleast_1d(sol["E_m"][i, :]))
    else:
        # 单单元 LP/QP：P_dis/P_ch 是一维，E_m 是 (1, H) 二维，取第0行
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
    """
    基于内层优化器调度解的PCA降维器
    适用场景：PaO框架，将MIQP/MILP/LP/QP调度解时序向量做标准化+PCA投影，作为RL观测输入
    支持两种模式：
        aggregate_unit=True：多储能单元聚合为总功率、总电量
        aggregate_unit=False：保留每个储能单元独立的功率、电量时序（MIQP主实验使用）
    使用流程：
        1. 离线阶段：用【训练集】数据批量调用SCIP求解，得到大量调度解样本X_raw
        2. fit()：对X_raw做标准化、拟合PCA投影矩阵
        3. save()：保存scaler、pca、元信息到磁盘
        4. RL训练阶段：load()直接加载模型，只调用transform()，不再拟合，杜绝数据泄露
    """
    def __init__(self, n_components=0.99, random_state=42):
        """
        Args:
            n_components:
                float(0~1): 保留对应比例总方差，自动选择输出维度（推荐0.99）
                int: 强制压缩到指定维度
            random_state: 随机种子，保证PCA结果可复现
        """
        self.n_components = n_components
        self.random_state = random_state
        # 下面成员在fit之后才会被赋值
        self.scaler = None          # StandardScaler标准化器
        self.pca = None             # PCA模型实例
        self.raw_dim = None         # 原始输入向量维度
        self.reduced_dim = None     # PCA降维之后输出维度
        self.explained_variance_ratio_ = None   # 各个主成分的方差占比
    def fit(self, X_raw):
        """
        使用原始调度解样本拟合标准化器与PCA
        Args:
            X_raw: np.ndarray shape=(n_samples, raw_dim)，仅使用训练集样本
        Returns: self
        """
        self.raw_dim = X_raw.shape[1]
        # 标准化：每个特征维度零均值、单位方差，PCA前必须做
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw)
        # 拟合PCA
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca.fit(X_scaled)
        self.reduced_dim = self.pca.n_components_
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        print(f"✅ PCA拟合完成")
        print(f"   原始维度: {self.raw_dim}")
        print(f"   降维后维度: {self.reduced_dim}")
        print(f"   累计方差解释率: {np.sum(self.explained_variance_ratio_):.4f}")
        return self
    def transform(self, state_vec):
        """
        对单条/批量调度解向量执行降维投影
        注意：只做transform，绝不fit！训练、测试阶段都调用这个。
        Args:
            state_vec:
                单样本: shape(raw_dim,)
                批量样本: shape(n_samples, raw_dim)
        Returns:
            降维后向量，输入维度格式保持一致，dtype=np.float32适配RL环境
        """
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
        """拟合+转换一步完成，离线生成样本时使用"""
        self.fit(X_raw)
        return self.transform(X_raw)
    def save(self, save_dir, meta_extra=None):
        """
        将scaler、pca、元信息全部保存到指定文件夹
        Args:
            save_dir: 输出文件夹路径，不存在自动创建
        """
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(save_dir, "scaler.pkl"))
        joblib.dump(self.pca, os.path.join(save_dir, "pca.pkl"))
        # 元信息：方便后续查看维度、方差，不参与计算
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
            # 关键：外部传入期望参数，不匹配直接报错
            if expected_meta is not None:
                for k, expect_val in expected_meta.items():
                    if k not in meta:
                        raise RuntimeError(f"PCA元数据缺失key: {k}，该PCA模型版本过旧，无法做参数校验")
                    disk_val = meta[k]
                    # numpy数组/标量转原生python值
                    if hasattr(disk_val, "item"):
                        disk_val = disk_val.item()
                    if disk_val != expect_val:
                        raise RuntimeError(
                            f"PCA模型参数不匹配！key={k}，期望={expect_val}，磁盘保存={disk_val}\n"
                            "⚠️ PCA训练超参和当前环境超参不一致，禁止运行，避免观测维度错乱。"
                        )
        else:
            # 没有meta文件，但用户传了expected_meta，说明用户想要校验，但是文件没有元数据，直接报错
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
    """
    两种工作模式二选一：
      模式A：传入data_tuple，滑动窗口逐窗口调用SCIP求解，生成调度解样本（现在用）
      模式B：传入load_samples_npz，直接读取已经保存好的X_raw，跳过SCIP求解，调试加速
    !!!! 重要约束：只能传入【训练集】数据，测试集绝对不能进这个函数，防止数据泄露 !!!!
    Args:
        data_tuple: (price, pv_ub, load, weather) 时序元组；load_samples_npz不为None时该参数无效
        load_samples_npz: str，若提供，直接读取npz中X_raw，不再跑SCIP
        model_type: 优化器类型 lp / qp / milp / miqp
        horizon: 调度时域步数，PaO中为horizon_F
        dt: 时间步长，单位小时
        num_clusters: 储能单元数量，MILP主实验为2
        stride: 滑动窗口移动步长；None时默认等于horizon，窗口无重叠
        seed: 求解器随机种子，用于初始化每个窗口储能SOC
        lambda_penalty: QP/MIQP二次惩罚系数（milp不使用）
        aggregate_unit:
            True：聚合所有储能单元为总功率总电量
            False：保留每个储能单元独立功率、电量（MILP主实验配置）
        save_samples_npz: str，不为None时，将生成/加载得到的X_raw保存到该路径，方便下次复用
    Returns:
        X_raw: np.ndarray shape=(n_valid_samples, feature_dim)，PCA的训练样本
    """
    # ========== 模式B：直接加载已保存样本，跳过SCIP求解 ==========
    if load_samples_npz is not None:
        print(f"📂 直接加载历史解样本，跳过SCIP求解：{load_samples_npz}")
        npz_data = np.load(load_samples_npz)
        X_raw = npz_data["X_raw"]
        print(f"📊 加载样本 | 有效样本:{X_raw.shape[0]}, 单样本维度:{X_raw.shape[1]}")
        if save_samples_npz is not None:
            os.makedirs(os.path.dirname(save_samples_npz), exist_ok=True)
            np.savez_compressed(save_samples_npz, X_raw=X_raw)
            print(f"💾 样本另存到 {save_samples_npz}")
        return X_raw
    # ========== 模式A：现场滑动窗口调用SCIP求解生成样本 ==========
    price, pv_ub, load, _ = data_tuple
    total_len = len(price)
    if stride is None:
        stride = horizon  # 默认无重叠滑动
    # 根据model_type实例化对应的优化求解器对象
    solver_map = {
        "milp": lambda: MILPOpt(horizon=horizon, dt=dt, num_clusters=num_clusters, seed=seed),
    }
    if model_type not in solver_map:
        raise ValueError(f"不支持的model_type={model_type}")
    opt = solver_map[model_type]()
    rng = np.random.default_rng(seed)
    sample_list = []
    t = 0
    # 滑动窗口遍历整个训练时序
    while t + horizon <= total_len:
        # 截取当前窗口的价格、光伏上限、负荷
        win_price = price[t:t+horizon]
        win_pv = pv_ub[t:t+horizon]
        win_load = load[t:t+horizon]
        # 随机初始化储能SOC，扩大解的分布多样性
        if model_type in ["milp", "miqp"]:
            init_soc = rng.uniform(0.3, 0.7, size=num_clusters)
        else:
            init_soc = np.array([rng.uniform(0.3, 0.7)])
        opt.set_initial_energy(init_soc)
        sol = opt.solve(win_price, win_pv, win_load)
        # ✅ 修改！SCIP状态判断，不再用Gurobi getStatus==2
        solve_success = opt.status in ["optimal", "gaplimit"]
        if solve_success:
            if aggregate_unit:
                # 聚合模式（你目前主实验不用，保留兼容）
                p_es_net = (sol["p_dis_m"] - sol["p_ch_m"]).sum(axis=0)
                e_total = sol["E_m"].sum(axis=0)
                state_vec = np.concatenate([
                    sol["P_grid"], p_es_net, sol["P_pv"], e_total
                ])
            else:
                # 分单元模式：统一调用标准展平函数
                state_vec = flatten_solution_vector(sol, model_type, num_clusters, horizon)
            sample_list.append(state_vec)
        t += stride
    # 安全释放SCIP模型资源，防止内存泄漏
    opt._safe_close()
    del opt
    if len(sample_list) == 0:
        raise RuntimeError("generate_solution_dataset：没有收集到任何有效最优解样本，请检查数据、SCIP求解器参数！")
    X_raw = np.array(sample_list, dtype=np.float32)

    print(f"📊 generate_solution_dataset | 有效样本:{len(sample_list)}, 单样本维度:{X_raw.shape[1]}")
    # 如果指定路径，把原始解样本保存npz，后续可以直接load跳过求解
    if save_samples_npz is not None:
        os.makedirs(os.path.dirname(save_samples_npz), exist_ok=True)
        np.savez_compressed(save_samples_npz, X_raw=X_raw)
        print(f"💾 解样本保存到 {save_samples_npz}")
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
    """
    PCA一站式流水线：
        1. 生成/加载调度解样本X_raw
        2. 拟合StandardScaler + PCA
        3. 保存全部模型文件到磁盘
    Args:
        data_tuple: 训练集时序元组；load_samples_npz不为None时无效
        load_samples_npz: 可选，直接读取历史npz样本，跳过SCIP
        save_dir: PCA模型输出文件夹
        其余参数透传给generate_solution_dataset
    Returns:
        compressor: SolutionPCACompressor实例，已经fit完成
    """
    print("=" * 60)
    print(f"🚀 build_pca_pipeline start")
    print(f"   model_type={model_type}, num_clusters={num_clusters}, aggregate_unit={aggregate_unit}")
    print("=" * 60)
    # 自动匹配 num_clusters
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
    """
    批量构建 PCA 降压器：
    - 数据集：真实数据(real)
    - 模型类型：milp
    - 自动匹配 num_clusters：milp=2
    """
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
        print(f"📦 已加载 {tag} 数据集，序列长度: {len(data['price'])}")
    # ========== 2. 运行 milp ==========
    model_list = ["milp"]
    for dataset_tag, data_tuple in loaded_data.items():
        print("\n" + "=" * 70)
        print(f"🎯 开始处理数据集: {dataset_tag}")
        print("=" * 70)
        for model_type in model_list:
            print(f"\n{'─' * 50}")
            print(f"▶ 正在构建: {dataset_tag} - {model_type}")
            print(f"{'─' * 50}")
            # 输出目录 & 缓存路径，自动带标识
            save_dir = f"./1_pca_{dataset_tag}_{model_type}_auto"
            sample_cache_path = os.path.join(save_dir, "solution_samples.npz")
            build_pca_pipeline(
                data_tuple=data_tuple,
                save_dir=save_dir,
                model_type=model_type,
                horizon=HORIZON,
                dt=DT,
                num_clusters=None,       # 自动匹配：milp=2
                n_components=N_COMPONENTS,
                seed=SEED,
                aggregate_unit=AGGREGATE_UNIT,
                save_samples_npz=sample_cache_path
            )
    # ========== 3. 运行完成汇总 ==========
    print("\n" + "=" * 70)
    print("✅ 全部 PCA 模型构建完成！输出目录清单：")
    for dataset_tag in loaded_data.keys():
        for model_type in model_list:
            print(f"   📁 ./1_pca_{dataset_tag}_{model_type}_auto")
    print("=" * 70)
    # srun -p AMD -N 1 -c 4 --mem=16G --pty /bin/bash
