import os
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from pyscipopt import Model, quicksum

# ==========================================
# base class for optimization models
# ==========================================
class BaseOptimization(ABC):
    # commom parameters, creates the scip model
    def __init__(self, horizon=12, dt=2.0, seed=42):
        self.horizon = horizon
        self.dt = dt
        # The grid power limit is 300. Total battery energy has an upper limit of 60 and a lower limit of 6.
        # Both charging and discharging efficiency. Energy is lost in both directions.
        self.P_trans_max = 300.0
        self.E_max = 60.0
        self.E_min = 6.0
        self.eta_ch = 0.95
        self.eta_dis = 0.95

        # Create one solver model. 
        self.model = Model()
        self.status = None
        self._setup_base_params(seed)

    # These settings support many repeated solves. They trade solution quality for speed.
    def _setup_base_params(self, seed):
        # close verbose output 
        self.model.setParam("display/verblevel", 0)
        
        # time limit
        self.model.setParam("limits/time", 0.3)
        
        # set gap between feasible solution and optimum
        self.model.setParam("limits/gap", 0.01)
        
        # close cutting planes
        # EN: Set normal cutting-plane rounds to zero to reduce this work. This can also make some problems harder to solve.
        self.model.setParam("separating/maxrounds", 0)
        self.model.setParam("separating/maxroundsroot", 0)
        
        # limit presolving rounds, just 1 round
        self.model.setParam("presolving/maxrounds", 1)
        
        # heuristics emphasis, closed
        # self.model.setParam("heuristics/emphasis", 0)
        
        # random seed for reproducibility
        self.model.setParam("randomization/randomseedshift", seed)

        # feasibility tolerance allows small numeric errors. 
        self.model.setParam("numerics/feastol", 1e-4)
        
        # set number of threads from environment variable, just default 1 
        threads = int(os.environ.get("SCIP_THREADS", 1))
        self.model.setParam("parallel/maxnthreads", threads)

    # add variable helper.
    def _add_var(self, lb, ub, vtype="C", name=""):
        return self.model.addVar(
            lb=float(lb),
            ub=None if ub is None else float(ub),
            vtype=vtype,
            name=name
        )
    
    # Return the objective value only for optimal or gaplimit.
    def get_obj_value(self, default_val=np.nan):
        if self.status in ["optimal","gaplimit"]:
            return self.model.getObjVal()
        return default_val
    
    # predefined output.
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

    # defines the variables and constraints.
    @abstractmethod
    def _build_model(self): pass

    # updates the prices, PV limits, load, and objective.
    @abstractmethod
    def update_parameters(self, price, pv_ub, load): pass

    # sets the initial SOC.
    @abstractmethod
    def set_initial_energy(self, socs): pass

    # get solution dictionary.
    @abstractmethod
    def _extract_solution(self) -> dict: pass

    # input parameters, calls the solver, checks the status extracts results or returns the fallback solution
    def solve(self, price, pv_ub, load):
        self.update_parameters(price, pv_ub, load) 
        self.model.optimize() 
        self.status = self.model.getStatus() # get status

        if self.status in ("optimal", "gaplimit", "timelimit"):
            try:
                return self._extract_solution()
            except Exception:
                return self._fallback_sol()
        # any other reasons
        return self._fallback_sol()

    # release resources, mannul
    def _safe_close(self):
        if hasattr(self, "model") and self.model is not None:
            try: self.model.freeProb()
            except Exception: pass

    # auto release resources
    def __del__(self):
        self._safe_close()

# MILP
class MILPOpt(BaseOptimization):
    def __init__(self, horizon=12, dt=2.0, num_clusters=2, seed=42):

        # run the base setup 
        super().__init__(horizon, dt, seed)
        self.M = num_clusters

        # discharge and charge power limit
        self.P_clus_ch_max = (self.E_max - self.E_min) / self.M
        self.P_clus_dis_max = (self.E_max - self.E_min) / self.M

        # capacity limits
        self.E_clus_max = self.E_max / self.M
        self.E_clus_min = self.E_min / self.M

        # slack penalty setting to avoid infeasible
        self.PENALTY = 1e5

        # EN: These lists store solver variable objects, not solved numeric values.
        # EN: Shared variables use [t]. Battery variables use [m][t], where m is the group index.
        self.P_pv = []
        self.P_grid = []
        self.slack_short = []
        self.slack_sur = []
        self.p_ch = []
        self.p_dis = []
        self.u = []
        self.E = []

        # Keep references to the initial-energy and power-balance constraints so their bounds can be changed later.
        self.con_init_e = {}
        self.con_balance = []

        # build model.
        self._build_model()
    
    # variables + constraints.
    def _build_model(self):
        h, M = self.horizon, self.M
        # 1. public variables
        for t in range(h):
            # PV upper bound to zero for now. Update it with the PV forecast later. Unused PV is allowed.
            self.P_pv.append(self._add_var(0, 0))
            # grid power.
            self.P_grid.append(self._add_var(0, self.P_trans_max))
            # shortage, surplus slack 
            self.slack_short.append(self._add_var(0, None))
            self.slack_sur.append(self._add_var(0, None))

        # 2. storage cluster variables
        for m in range(M):
            ch, dis, u, e = [], [], [], []
            for t in range(h):
                ch.append(self._add_var(0, self.P_clus_ch_max))
                dis.append(self._add_var(0, self.P_clus_dis_max))
                # u=1 allows charging. u=0 allows discharging.
                u.append(self._add_var(0, 1, vtype="B"))

            for t in range(h + 1):
                e.append(self._add_var(self.E_clus_min, self.E_clus_max))
            self.p_ch.append(ch)
            self.p_dis.append(dis)
            self.u.append(u)
            self.E.append(e)

        # 3. initial energy constraints
        # default SOC is 0.5 for each group. env will updates init soc later.
        for m in range(M):
            rhs = 0.5 * self.E_clus_max
            self.con_init_e[m] = self.model.addCons(self.E[m][0] == rhs)

        # 4. power balance constraints
        # Power balance: PV + grid + discharge - charge + shortage - surplus = load.
        for t in range(h):
            expr = (
                self.P_pv[t] + self.P_grid[t] 
                + self.slack_short[t] 
                - self.slack_sur[t]
                + quicksum(self.p_dis[m][t] for m in range(M))
                - quicksum(self.p_ch[m][t] for m in range(M))
            )
            # zero when building. update_parameters load later.
            self.con_balance.append(self.model.addCons(expr == 0))

        # 5. charging/discharging constraints
        # no charging and discharging at the same time.
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

        # # 7. objective function / update_parameters done this job, commented out
        # obj = quicksum(
        #     0 * self.P_grid[t] + self.PENALTY * (self.slack_short[t] + self.slack_sur[t])
        #     for t in range(h)
        # )
        # self.model.setObjective(obj, "minimize")

    def set_initial_energy(self, socs):

        # 中文：SCIP 求解时会创建变换后的模型；freeTransform 用于返回可修改原模型的阶段。
        # EN: SCIP creates a transformed model for solving. freeTransform returns to the stage where the original model can be changed.
        # 中文：此判断按原代码保留；getStage 的返回类型应以所用 PySCIPOpt 版本为准。
        # EN: This check is kept as written. The return type of getStage depends on the PySCIPOpt interface in use.
        if self.model.getStage() != "problem":
            self.model.freeTransform()
            
        for m in range(self.M):
            # 中文：用 SOC 乘该组最大容量，得到初始能量的数值。
            # EN: Multiply SOC by the group's full capacity to get its starting energy.
            rhs = socs[m] * self.E_clus_max
            # lhs rhs
            # 中文：求解器把等式存成下界 <= 表达式 <= 上界；两边都设成 rhs 才保持等式。
            # EN: The solver stores a constraint as lower bound <= expression <= upper bound. Set both bounds to rhs to keep an equality.
            self.model.chgLhs(self.con_init_e[m], rhs) #
            self.model.chgRhs(self.con_init_e[m], rhs)

    # Update price pv_ub and load.
    def update_parameters(self, price, pv_ub, load):
        # print(f"[update] price={price}")
        # print(f"[update] pv_ub={pv_ub}")
        # print(f"[update] load={load}")
        # print(f"load sum: {sum(load)}")

        # problem -- haven't solved.
        if self.model.getStage() != "problem":
            self.model.freeTransform()

        # update
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
        # update obj
        self.model.setObjective(obj_expr, "minimize", clear=True)

    # get solution dictionary
    def _extract_solution(self) -> dict:
        h, M = self.horizon, self.M
        val = self.model.getVal # renamed

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
    
    # SOC, 。
    # Return the final energy of each group. The environment divides by capacity to get SOC for the next period.
    def get_final_energy(self):
        val = self.model.getVal
        return np.array([val(self.E[m][-1]) for m in range(self.M)], dtype=np.float32)



# test
if __name__ == "__main__":

    DATA = Path("./0_dataset_real_processed/test_dt2.0h.npz")
    data = np.load(DATA)

    # 12 slots from  96 to cover 24 hours.
    h_steps = 12
    t_start = 96

    pred_pv = data["pv_ub"][t_start:t_start+h_steps].copy() * 1.0
    pred_load = data["load"][t_start:t_start+h_steps].copy() * 1.0
    pred_price = data["price"][t_start:t_start+h_steps].copy() * 1.0 

    opt = MILPOpt(horizon=h_steps, dt=2.0, num_clusters=2)
    opt.model.setParam("display/verblevel", 4)
    opt.model.setParam("limits/time", 20)

    # 50% SOC.
    opt.set_initial_energy(np.array([0.5, 0.5]))

    sol_dict = opt.solve(pred_price, pred_pv, pred_load)


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
