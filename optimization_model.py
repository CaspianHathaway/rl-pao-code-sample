import os
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from pyscipopt import Model, quicksum

# ==========================================
# base class for optimization models
# ==========================================
class BaseOptimization(ABC):
    def __init__(self, horizon=12, dt=2.0, seed=42):
        self.horizon = horizon
        self.dt = dt

        self.P_trans_max = 300.0
        self.E_max = 60.0
        self.E_min = 6.0
        self.eta_ch = 0.95
        self.eta_dis = 0.95

        self.model = Model()
        self.status = None
        self._setup_base_params(seed)
    def _setup_base_params(self, seed):
        # 1. close verbose output 
        self.model.setParam("display/verblevel", 0)
        
        # 2. decrease time limit
        self.model.setParam("limits/time", 0.3)
        
        # 3. lose precision gap limit
        self.model.setParam("limits/gap", 0.01)
        
        # 4. close cutting planes
        self.model.setParam("separating/maxrounds", 0)
        self.model.setParam("separating/maxroundsroot", 0)
        
        # 5. limit presolving rounds
        self.model.setParam("presolving/maxrounds", 1)
        
        # 6. heuristics emphasis
        # self.model.setParam("heuristics/emphasis", 0)
        
        # 7. random seed for reproducibility
        self.model.setParam("randomization/randomseedshift", seed)
        self.model.setParam("numerics/feastol", 1e-4)
        
        # 8. set number of threads from environment variable
        threads = int(os.environ.get("SCIP_THREADS", 1))
        self.model.setParam("parallel/maxnthreads", threads)

    def _add_var(self, lb, ub, vtype="C", name=""):
        return self.model.addVar(
            lb=float(lb),
            ub=None if ub is None else float(ub),
            vtype=vtype,
            name=name
        )
    def get_obj_value(self, default_val=np.nan):
        if self.status in ["optimal","gaplimit"]:
            return self.model.getObjVal()
        return default_val
    def _fallback_sol(self):
        num_m = getattr(self, "M", 1)
        h = self.horizon
        return {
            "obj_val": np.inf,  # 
            "P_grid": np.zeros(h), "P_ch": np.zeros(h), "P_dis": np.zeros(h),
            "P_pv": np.zeros(h), "slack_shortage": np.zeros(h), "slack_surplus": np.zeros(h),
            "p_ch_m": np.zeros((num_m, h)), "p_dis_m": np.zeros((num_m, h)),
            "u_m": np.zeros((num_m, h)), "E_m": np.zeros((num_m, h))
        }

    @abstractmethod
    def _build_model(self): pass
    @abstractmethod
    def update_parameters(self, price, pv_ub, load): pass
    @abstractmethod
    def set_initial_energy(self, socs): pass
    @abstractmethod
    def _extract_solution(self) -> dict: pass

    def solve(self, price, pv_ub, load):
        self.update_parameters(price, pv_ub, load)

        self.model.optimize()
        self.status = self.model.getStatus()

        # 1. 
        if self.status in ["optimal", "gaplimit"]:
            try:
                return self._extract_solution()
            except Exception:
                return self._fallback_sol()
        # 2. 
        elif self.status == "timelimit":
            try:
                return self._extract_solution()
            except Exception:
                return self._fallback_sol()
        # 3. 
        else:
            return self._fallback_sol()

    def _safe_close(self):
        if hasattr(self, "model") and self.model is not None:
            try: self.model.freeProb()
            except Exception: pass
    def __del__(self):
        self._safe_close()

class MILPOpt(BaseOptimization):
    def __init__(self, horizon=12, dt=2.0, num_clusters=2, seed=42):
        super().__init__(horizon, dt, seed)
        self.M = num_clusters

        self.P_clus_ch_max = (self.E_max - self.E_min) / self.M
        self.P_clus_dis_max = (self.E_max - self.E_min) / self.M
        self.E_clus_max = self.E_max / self.M
        self.E_clus_min = self.E_min / self.M
        self.PENALTY = 1e5

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
        gap_val = 0.01
        try:
            self.model.setParam("mip/gaprel", gap_val)
        except KeyError:
            self.model.setParam("limits/gap", gap_val)
            
    def _build_model(self):
        h, M = self.horizon, self.M
        # 1. public
        for t in range(h):
            self.P_pv.append(self._add_var(0, 0))
            self.P_grid.append(self._add_var(0, self.P_trans_max))
            self.slack_short.append(self._add_var(0, None))
            self.slack_sur.append(self._add_var(0, None))
        # 2. storage cluster variables
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
        # 3. initial energy constraints
        for m in range(M):
            rhs = 0.5 * self.E_clus_max
            self.con_init_e[m] = self.model.addCons(self.E[m][0] == rhs)
        # 4. power balance constraints
        for t in range(h):
            expr = (
                self.P_pv[t] + self.P_grid[t] 
                + self.slack_short[t] 
                - self.slack_sur[t]
                + quicksum(self.p_dis[m][t] for m in range(M))
                - quicksum(self.p_ch[m][t] for m in range(M))
            )
            self.con_balance.append(self.model.addCons(expr == 0))
        # 5. charging/discharging constraints
        for m in range(M):
            for t in range(h):
                self.model.addCons(self.p_ch[m][t] <= self.P_clus_ch_max * self.u[m][t])
                self.model.addCons(self.p_dis[m][t] <= self.P_clus_dis_max * (1 - self.u[m][t]))
        # 6. soc dynamics
        for m in range(M):
            for t in range(h):
                self.model.addCons(
                    self.E[m][t+1] == self.E[m][t]
                    + self.eta_ch * self.dt * self.p_ch[m][t]
                    - (self.dt / self.eta_dis) * self.p_dis[m][t]
                )
        # # 7. objective function
        # obj = quicksum(
        #     0 * self.P_grid[t] + self.PENALTY * (self.slack_short[t] + self.slack_sur[t])
        #     for t in range(h)
        # )
        # self.model.setObjective(obj, "minimize")

    def set_initial_energy(self, socs):

        if self.model.getStage() != "problem":
            self.model.freeTransform()
            
        for m in range(self.M):
            rhs = socs[m] * self.E_clus_max
            # lhs rhs
            self.model.chgLhs(self.con_init_e[m], rhs) #
            self.model.chgRhs(self.con_init_e[m], rhs)

    def update_parameters(self, price, pv_ub, load):
        # print(f"[update] price={price}")
        # print(f"[update] pv_ub={pv_ub}")
        # print(f"[update] load={load}")
        # print(f"load sum: {sum(load)}")

        if self.model.getStage() != "problem":
            self.model.freeTransform()
        h = self.horizon
        for t in range(h):
            self.model.chgVarUb(self.P_pv[t], float(pv_ub[t]))
            self.model.chgLhs(self.con_balance[t], float(load[t]))
            self.model.chgRhs(self.con_balance[t], float(load[t]))

        obj_expr = quicksum(
            float(price[t] * self.dt) * self.P_grid[t]
            + self.PENALTY * (self.slack_short[t] + self.slack_sur[t])
            for t in range(h)
        )

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
            "obj_val": self.model.getObjVal(),  
            "P_grid": P_grid, "P_pv": P_pv,
            "P_ch": p_ch.sum(axis=0), "P_dis": p_dis.sum(axis=0),
            "slack_shortage": slack_short, "slack_surplus": slack_sur,
            "p_ch_m": p_ch, "p_dis_m": p_dis, "u_m": u, "E_m": E
        }
    def get_final_energy(self):
        val = self.model.getVal
        return np.array([val(self.E[m][-1]) for m in range(self.M)], dtype=np.float32)



# test
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

    # ==================
    print("======================")
    print(f"pred_price: {pred_price}")
    print(f"pred_pv:    {pred_pv}")
    print(f"pred_load:  {pred_load}")
    print(f"price shape:{pred_price.shape}, pv shape:{pred_pv.shape}, load shape:{pred_load.shape}")
    print(f"load max:{pred_load.max()}, load min:{pred_load.min()}")
    print("======================")

    print("optimal obj:", sol_dict["obj_val"])
    print("P_grid:", sol_dict["P_grid"])
    print("max load:", data["load"].max())
    print("max pv_ub:", data["pv_ub"].max())
    print("max price:", data["price"].max())


        # srun -p AMD -N 1 -c 4 --mem=16G --pty /bin/bash
