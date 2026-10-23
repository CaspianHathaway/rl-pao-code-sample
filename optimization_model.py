import os
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from pyscipopt import Model, quicksum

# ==========================================
# 基类：通用优化框架，所有范式复用
# ==========================================
class BaseOptimization(ABC):
    def __init__(self, horizon=12, dt=2.0, seed=42):
        self.horizon = horizon
        self.dt = dt
        # 通用储能物理参数，子类可按需覆盖
        self.P_trans_max = 300.0
        self.E_max = 60.0
        self.E_min = 6.0
        self.eta_ch = 0.95
        self.eta_dis = 0.95
        # 求解器实例与状态
        self.model = Model()
        self.status = None
        self._setup_base_params(seed)
    def _setup_base_params(self, seed):
        """所有范式通用的求解器基础参数 - 强化学习极速优化版"""
        # 1. 关闭日志输出
        self.model.setParam("display/verblevel", 0)
        
        # 2. 压低单步硬超时（从 3.0s 改为 0.1s，足以应付日前调度，防止单步深挖）
        self.model.setParam("limits/time", 0.3)
        
        # 3. 放宽相对 MIP 收敛 Gap（1% 误差在能量调度中完全可接受，耗时直降 50%+）
        self.model.setParam("limits/gap", 0.01)
        
        # 4. 彻底关闭重型割平面生成（SCIP 相比 HiGHS 慢的核心根源）
        self.model.setParam("separating/maxrounds", 0)
        self.model.setParam("separating/maxroundsroot", 0)
        
        # 5. 限制预处理轮数
        self.model.setParam("presolving/maxrounds", 1)
        
        # 6. 启发式搜索采用快速模式
        # self.model.setParam("heuristics/emphasis", 0)
        
        # 7. 容差与随机种子保持原样
        self.model.setParam("randomization/randomseedshift", seed)
        self.model.setParam("numerics/feastol", 1e-4)
        
        # 8. 线程锁定
        threads = int(os.environ.get("SCIP_THREADS", 1))
        self.model.setParam("parallel/maxnthreads", threads)
    def _add_var(self, lb, ub, vtype="C", name=""):
        """轻量封装：统一浮点类型转换，返回变量对象"""
        return self.model.addVar(
            lb=float(lb),
            ub=None if ub is None else float(ub),
            vtype=vtype,
            name=name
        )
    def get_obj_value(self, default_val=np.nan):
        """通用目标值读取"""
        if self.status in ["optimal","gaplimit"]:
            return self.model.getObjVal()
        return default_val
    def _fallback_sol(self):
        """求解失败兜底零解，子类可覆盖"""
        num_m = getattr(self, "M", 1)
        h = self.horizon
        return {
            "obj_val": np.inf,  # 新增：无解标记为无穷大成本
            "P_grid": np.zeros(h), "P_ch": np.zeros(h), "P_dis": np.zeros(h),
            "P_pv": np.zeros(h), "slack_shortage": np.zeros(h), "slack_surplus": np.zeros(h),
            "p_ch_m": np.zeros((num_m, h)), "p_dis_m": np.zeros((num_m, h)),
            "u_m": np.zeros((num_m, h)), "E_m": np.zeros((num_m, h))
        }
    # ---------- 子类必须实现的抽象方法 ----------
    @abstractmethod
    def _build_model(self): pass
    @abstractmethod
    def update_parameters(self, price, pv_ub, load): pass
    @abstractmethod
    def set_initial_energy(self, socs): pass
    @abstractmethod
    def _extract_solution(self) -> dict: pass
    # ---------- 通用求解流程，所有范式复用 ----------
    def solve(self, price, pv_ub, load):
        self.update_parameters(price, pv_ub, load)
        # # =========导出模型，关键！update参数完成之后再导出！=========
        # self.model.writeProblem("model_dump.lp")
        # print("模型已经导出到 model_dump.lp，可以查看完整lp文件")

        # self.model.optimize()
        # self.status = self.model.getStatus()
        # if self.status in ["optimal", "gaplimit", "timelimit"]:
        #     return self._extract_solution()
        # return self._fallback_sol()

        self.model.optimize()
        self.status = self.model.getStatus()

        # 1. 最优/间隙限：一定有可行解，加异常保护
        if self.status in ["optimal", "gaplimit"]:
            try:
                return self._extract_solution()
            except Exception:
                return self._fallback_sol()
        # 2. 超时：可能还没找到可行解，尝试读取，失败则兜底
        elif self.status == "timelimit":
            try:
                return self._extract_solution()
            except Exception:
                return self._fallback_sol()
        # 3. 其他状态（不可行、出错等）直接兜底
        else:
            return self._fallback_sol()

    def _safe_close(self):
        if hasattr(self, "model") and self.model is not None:
            try: self.model.freeProb()
            except Exception: pass
    def __del__(self):
        self._safe_close()
# ==========================================
# MILP 子类：分簇储能整数调度模型
# ==========================================
class MILPOpt(BaseOptimization):
    def __init__(self, horizon=12, dt=2.0, num_clusters=2, seed=42):
        super().__init__(horizon, dt, seed)
        self.M = num_clusters
        # 分簇专属参数
        self.P_clus_ch_max = (self.E_max - self.E_min) / self.M
        self.P_clus_dis_max = (self.E_max - self.E_min) / self.M
        self.E_clus_max = self.E_max / self.M
        self.E_clus_min = self.E_min / self.M
        self.PENALTY = 1e5
        # 变量与约束句柄
        self.P_pv = []
        self.P_grid = []
        self.slack_short = []
        self.slack_sur = []
        self.p_ch = []
        self.p_dis = []
        self.u = []
        self.E = []
        self.con_init_e = {}
        self.con_balance = []
        self._build_model()
        self._setup_mip_params()
    def _setup_mip_params(self):
        """MIP 专属参数，SCIP10 标准 mip/gaprel，兼容老版本回退"""
        gap_val = 0.01
        try:
            self.model.setParam("mip/gaprel", gap_val)
        except KeyError:
            self.model.setParam("limits/gap", gap_val)
    def _build_model(self):
        h, M = self.horizon, self.M
        # 1. 公共变量
        for t in range(h):
            self.P_pv.append(self._add_var(0, 0))
            self.P_grid.append(self._add_var(0, self.P_trans_max))
            self.slack_short.append(self._add_var(0, None))
            self.slack_sur.append(self._add_var(0, None))
        # 2. 分簇变量
        for m in range(M):
            ch, dis, u, e = [], [], [], []
            for t in range(h):
                ch.append(self._add_var(0, self.P_clus_ch_max))
                dis.append(self._add_var(0, self.P_clus_dis_max))
                u.append(self._add_var(0, 1, vtype="B"))
            for t in range(h + 1):
                e.append(self._add_var(self.E_clus_min, self.E_clus_max))
            self.p_ch.append(ch)
            self.p_dis.append(dis)
            self.u.append(u)
            self.E.append(e)
        # 3. 初始 SOC 等式约束
        for m in range(M):
            rhs = 0.5 * self.E_clus_max
            self.con_init_e[m] = self.model.addCons(self.E[m][0] == rhs)
        # 4. 功率平衡约束
        for t in range(h):
            expr = (
                self.P_pv[t] + self.P_grid[t] 
                + self.slack_short[t] 
                - self.slack_sur[t]
                + quicksum(self.p_dis[m][t] for m in range(M))
                - quicksum(self.p_ch[m][t] for m in range(M))
            )
            self.con_balance.append(self.model.addCons(expr == 0))
        # 5. 充放电互斥约束
        for m in range(M):
            for t in range(h):
                self.model.addCons(self.p_ch[m][t] <= self.P_clus_ch_max * self.u[m][t])
                self.model.addCons(self.p_dis[m][t] <= self.P_clus_dis_max * (1 - self.u[m][t]))
        # 6. SOC 动力学
        for m in range(M):
            for t in range(h):
                self.model.addCons(
                    self.E[m][t+1] == self.E[m][t]
                    + self.eta_ch * self.dt * self.p_ch[m][t]
                    - (self.dt / self.eta_dis) * self.p_dis[m][t]
                )
        # # 7. 目标函数（电价系数滚动时更新）
        # obj = quicksum(
        #     0 * self.P_grid[t] + self.PENALTY * (self.slack_short[t] + self.slack_sur[t])
        #     for t in range(h)
        # )
        # self.model.setObjective(obj, "minimize")
    def set_initial_energy(self, socs):
        # 修复2：如果之前 solve 过一次，现在是重新滚动求解，必须调用 freeTransform
        if self.model.getStage() != "problem":
            self.model.freeTransform()
            
        for m in range(self.M):
            rhs = socs[m] * self.E_clus_max
            # 修复1：等式约束必须同时修改 LHS 和 RHS，否则变成不等式
            self.model.chgLhs(self.con_init_e[m], rhs) # ？？？？？
            self.model.chgRhs(self.con_init_e[m], rhs)
    def update_parameters(self, price, pv_ub, load):
        # print(f"[update] price={price}")
        # print(f"[update] pv_ub={pv_ub}")
        # print(f"[update] load={load}")
        # print(f"load sum: {sum(load)}")
        # 必须项：如果是滚动优化，释放先前的求解状态回到 problem 阶段
        if self.model.getStage() != "problem":
            self.model.freeTransform()
        h = self.horizon
        for t in range(h):
            self.model.chgVarUb(self.P_pv[t], float(pv_ub[t]))
            # 必须项：等式约束同时改 LHS 和 RHS
            self.model.chgLhs(self.con_balance[t], float(load[t])) # ？？？？
            self.model.chgRhs(self.con_balance[t], float(load[t]))
        # 用你原本正确的逻辑重新构造目标函数
        obj_expr = quicksum(
            float(price[t] * self.dt) * self.P_grid[t]
            + self.PENALTY * (self.slack_short[t] + self.slack_sur[t])
            for t in range(h)
        )
        # clear=True 强制清除掉底层的旧表达式缓存（兼容老版本关键）
        self.model.setObjective(obj_expr, "minimize", clear=True)
    def _extract_solution(self) -> dict:
        h, M = self.horizon, self.M
        val = self.model.getVal
        P_grid = np.array([val(v) for v in self.P_grid])
        P_pv = np.array([val(v) for v in self.P_pv])
        slack_short = np.array([val(v) for v in self.slack_short])
        slack_sur = np.array([val(v) for v in self.slack_sur])
        p_ch = np.zeros((M, h))
        p_dis = np.zeros((M, h))
        u = np.zeros((M, h))
        E = np.zeros((M, h))
        for m in range(M):
            for t in range(h):
                p_ch[m, t] = val(self.p_ch[m][t])
                p_dis[m, t] = val(self.p_dis[m][t])
                u[m, t] = val(self.u[m][t])
                E[m, t] = val(self.E[m][t+1])
        return {
            "obj_val": self.model.getObjVal(),   # 新增目标函数值
            "P_grid": P_grid, "P_pv": P_pv,
            "P_ch": p_ch.sum(axis=0), "P_dis": p_dis.sum(axis=0),
            "slack_shortage": slack_short, "slack_surplus": slack_sur,
            "p_ch_m": p_ch, "p_dis_m": p_dis, "u_m": u, "E_m": E
        }
    def get_final_energy(self):
        val = self.model.getVal
        return np.array([val(self.E[m][-1]) for m in range(self.M)], dtype=np.float32)



# ==========================================
# 测试运行
# ==========================================
if __name__ == "__main__":
    DATA = Path("./0_dataset_real_processed/test_dt2.0h.npz")
    data = np.load(DATA)

    h_steps = 12
    t_start = 96
    pred_pv = data["pv_ub"][t_start:t_start+h_steps].copy() * 1.0
    pred_load = data["load"][t_start:t_start+h_steps].copy() * 1.0
    pred_price = data["price"][t_start:t_start+h_steps].copy() * 1.0 

    opt = MILPOpt(horizon=h_steps, dt=2.0, num_clusters=2)
    opt.model.setParam("display/verblevel", 4)
    opt.model.setParam("limits/time", 20)
    opt.set_initial_energy(np.array([0.5, 0.5]))

    sol_dict = opt.solve(pred_price, pred_pv, pred_load)

    # =========新增打印调试=========
    print("=====输入参数检查=====")
    print(f"pred_price: {pred_price}")
    print(f"pred_pv:    {pred_pv}")
    print(f"pred_load:  {pred_load}")
    print(f"price shape:{pred_price.shape}, pv shape:{pred_pv.shape}, load shape:{pred_load.shape}")
    print(f"load最大值:{pred_load.max()}, load最小值:{pred_load.min()}")
    print("======================")

    print("最优目标值:", sol_dict["obj_val"])
    print("P_grid:", sol_dict["P_grid"])
    print("全年最大load:", data["load"].max())
    print("全年最大pv_ub:", data["pv_ub"].max())
    print("全年最大price:", data["price"].max())
    # if opt.status == "optimal":
    #     print(f"\n最优目标值: {opt.get_obj_value():.4f}")
    #     print("P_grid:", np.round(sol_dict["P_grid"], 2))
    #     print("最终SOC:", np.round(opt.get_final_energy(), 2))
    # else:
    #     print(f"求解失败，status={opt.status}")


        # srun -p AMD -N 1 -c 4 --mem=16G --pty /bin/bash
