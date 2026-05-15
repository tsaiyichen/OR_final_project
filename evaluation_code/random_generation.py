import numpy as np
import itertools
import time
import gurobipy as gp
from gurobipy import GRB
import copy


# ==========================================
# 1. 系統參數與資料生成模組 (Data Module)
# ==========================================
class BaseballConfig:
    def __init__(self):
        self.G = 300 # 測試場數
        self.P = 13  # 投手名單人數
        self.W = 5  # LMSH 的 Sliding Window 大小
        self.M_val = 50  # Big-M 的值
        self.epsilon = 0.001  # Gurobi 用的容忍度
        self.innings = [5, 2, 1, 1]  # SP, MR, CL1, CL2 的局數


def generate_data(config):
    C = np.random.uniform(2.0, 6.0, config.G)
    base_eri = np.random.uniform(0.1, 0.9, config.P)

    E = np.zeros((config.P, 4))
    for i in range(config.P):
        for j in range(4):
            E[i, j] = base_eri[i] * config.innings[j]

    print(f"[Data] Generated: {config.G} games, {config.P} pitchers.")
    return C, E, base_eri


# ==========================================
# 2. 狀態機模組 (State Machine for Heuristics)
# ==========================================
class RosterState:
    """統一管理投手的疲勞、休息與出賽次數，確保所有 Heuristics 遵守 OR Constraints"""

    def __init__(self, P):
        self.P = P
        self.rest_days = np.zeros(P, dtype=int)  # 必須休息的天數 (SP=4, MR=1)
        self.starts = np.zeros(P, dtype=int)  # 先發次數上限 (Constraint 8)
        self.pitched_yesterday = np.zeros(P, dtype=bool)  # 昨天是否出賽
        self.pitched_day_before = np.zeros(P, dtype=bool)  # 前天是否出賽

    def get_available_pitchers(self):
        """回傳今天可以合法出賽的投手名單"""
        available = []
        for i in range(self.P):
            # Constraint: 休息天數歸零 AND (不能連續三天出賽 -> 昨天和前天不能同時出賽)
            if self.rest_days[i] == 0 and not (self.pitched_yesterday[i] and self.pitched_day_before[i]):
                available.append(i)
        return available

    def get_valid_sp(self, available_pitchers):
        """從可用投手中濾出可以擔任先發的人 (未達 25 場限制)"""
        return [i for i in available_pitchers if self.starts[i] < 25]

    def apply_assignment(self, sp, mr, cl1, cl2):
        """套用指派，並將時間推進到下一天"""
        today_pitchers = np.zeros(self.P, dtype=bool)
        today_pitchers[[sp, mr, cl1, cl2]] = True

        # 更新狀態
        self.starts[sp] += 1

        # 推進時間：減少所有人的休息天數
        self.rest_days = np.maximum(0, self.rest_days - 1)

        # 套用新的休息限制 (Constraint 6 & 7)
        self.rest_days[sp] = max(self.rest_days[sp], 4)
        self.rest_days[mr] = max(self.rest_days[mr], 1)

        # 推進連續出賽紀錄 (Constraint 5)
        self.pitched_day_before = self.pitched_yesterday.copy()
        self.pitched_yesterday = today_pitchers

    def clone(self):
        """提供給 Lookahead Simulation 深拷貝使用"""
        return copy.deepcopy(self)


# ==========================================
# 3. 演算法模組 - Gurobi (Exact IP)
# ==========================================
def solve_gurobi_exact(config, C, E):
    print("\n--- Running Gurobi Exact Model ---")
    start_time = time.time()
    m = gp.Model("Pitcher_Scheduling")
    m.Params.OutputFlag = 0

    x = m.addVars(config.P, 4, config.G, vtype=GRB.BINARY, name="x")
    w = m.addVars(config.G, vtype=GRB.BINARY, name="w")
    m.setObjective(gp.quicksum(w[k] for k in range(config.G)), GRB.MAXIMIZE)

    for k in range(config.G):
        m.addConstr(
            gp.quicksum(E[i, j] * x[i, j, k] for i in range(config.P) for j in range(4))
            - C[k] + config.epsilon <= config.M_val * (1 - w[k])
        )
        for j in range(4):
            m.addConstr(gp.quicksum(x[i, j, k] for i in range(config.P)) == 1)

    for i in range(config.P):
        for k in range(config.G):
            m.addConstr(gp.quicksum(x[i, j, k] for j in range(4)) <= 1)

            # [修正!] Constraint 5: 全局疲勞限制 (所有 j in range(4)，不是只有 Closers)
            if k <= config.G - 3:
                m.addConstr(gp.quicksum(x[i, j, k + t] for j in range(4) for t in range(3)) <= 2)

            if k <= config.G - 2:
                m.addConstr(gp.quicksum(x[i, j, k + 1] for j in range(4)) <= 1 * (1 - x[i, 1, k]))

            future_days = min(4, config.G - 1 - k)
            if future_days > 0:
                m.addConstr(
                    gp.quicksum(x[i, j, k + t] for j in range(4) for t in range(1, future_days + 1))
                    <= future_days * (1 - x[i, 0, k])
                )
        m.addConstr(gp.quicksum(x[i, 0, k] for k in range(config.G)) <= 25)

    m.optimize()
    run_time = time.time() - start_time
    if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        wins = int(round(m.objVal))
        print(f"[Gurobi] Result: {wins} Wins | Time: {run_time:.4f} sec")
        return wins, run_time
    else:
        print(f"[Gurobi] Failed or Infeasible. Status: {m.status}")
        return 0, run_time


# ==========================================
# 4. 演算法模組 - Benchmark (Naive Greedy)
# ==========================================
def solve_benchmark_greedy(config, C, E, base_eri):
    print("\n--- Running Benchmark Greedy ---")
    start_time = time.time()
    state = RosterState(config.P)
    wins = 0

    for k in range(config.G):
        avail = state.get_available_pitchers()
        avail.sort(key=lambda i: base_eri[i])  # 依能力排序 (越低越強)

        valid_sp = state.get_valid_sp(avail)
        if not valid_sp or len(avail) < 4:
            # 極端情況防呆：如果可用投手不足，直接判定輸掉此局並隨便推進狀態
            state.apply_assignment(0, 1, 2, 3)
            continue

        # 貪婪指派：最強的當先發、次強當中繼、最爛兩個當終結
        sp = valid_sp[0]
        avail.remove(sp)
        mr = avail[0]
        avail.remove(mr)
        cl1, cl2 = avail[-1], avail[-2]

        if E[sp, 0] + E[mr, 1] + E[cl1, 2] + E[cl2, 3] < C[k]:
            wins += 1

        state.apply_assignment(sp, mr, cl1, cl2)

    run_time = time.time() - start_time
    print(f"[Benchmark] Result: {wins} Wins | Time: {run_time:.4f} sec")
    return wins, run_time

# ==========================================
# 4.5 演算法模組 - Pure Greedy (純粹貪婪)
# ==========================================
def solve_pure_greedy(config, C, E, base_eri):
    print("\n--- Running Pure Greedy ---")
    start_time = time.time()
    state = RosterState(config.P)
    wins = 0

    for k in range(config.G):
        avail = state.get_available_pitchers()
        avail.sort(key=lambda i: base_eri[i])  # 依能力排序 (越低越強)

        valid_sp = state.get_valid_sp(avail)
        if not valid_sp or len(avail) < 4:
            # 極端情況防呆：如果可用投手不足，直接判定輸掉此局並隨便推進狀態
            state.apply_assignment(0, 1, 2, 3)
            continue

        # 純粹貪婪指派：最強的當先發，剩下「最強的三個」去當中繼與終結
        sp = valid_sp[0]
        avail.remove(sp)

        mr = avail[0]
        cl1 = avail[1]
        cl2 = avail[2]

        if E[sp, 0] + E[mr, 1] + E[cl1, 2] + E[cl2, 3] < C[k]:
            wins += 1

        state.apply_assignment(sp, mr, cl1, cl2)

    run_time = time.time() - start_time
    print(f"[Pure Greedy] Result: {wins} Wins | Time: {run_time:.4f} sec")
    return wins, run_time

# ==========================================
# 5. 演算法模組 - LMSH (Heuristic)
# ==========================================
def get_valid_combinations(avail, valid_sp):
    """產生所有合法的投手組合 (SP, MR, CL1, CL2)"""
    combs = []
    for sp in valid_sp:
        rem1 = [p for p in avail if p != sp]
        for mr in rem1:
            rem2 = [p for p in rem1 if p != mr]
            for cls in itertools.combinations(rem2, 2):
                combs.append((sp, mr, cls[0], cls[1]))
    return combs


def solve_lmsh(config, C, E):
    print("\n--- Running LMSH (Lookahead Minimal-Slack) ---")
    start_time = time.time()
    state = RosterState(config.P)
    global_wins = 0

    for k in range(config.G):
        avail = state.get_available_pitchers()
        valid_sp = state.get_valid_sp(avail)
        all_combs = get_valid_combinations(avail, valid_sp)

        if not all_combs:  # 防呆機制
            state.apply_assignment(0, 1, 2, 3)
            continue

        comb_data = [(c, E[c[0], 0] + E[c[1], 1] + E[c[2], 2] + E[c[3], 3]) for c in all_combs]

        # 分類並產生 Candidates
        wins_combs = sorted([x for x in comb_data if x[1] < C[k]], key=lambda x: x[1], reverse=True)
        loss_combs = sorted([x for x in comb_data if x[1] >= C[k]], key=lambda x: x[1], reverse=True)

        candidates = wins_combs[:2] + loss_combs[:1]
        if not candidates: candidates = [comb_data[0]]

        best_cand, best_sim_wins = None, -1

        # Lookahead Simulation
        for cand, cand_er in candidates:
            sim_wins = 1 if cand_er < C[k] else 0
            sim_state = state.clone()
            sim_state.apply_assignment(*cand)

            for t in range(k + 1, min(k + config.W, config.G)):
                s_avail = sim_state.get_available_pitchers()
                s_valid_sp = sim_state.get_valid_sp(s_avail)
                s_combs = get_valid_combinations(s_avail, s_valid_sp)

                if not s_combs: break  # 若模擬中崩潰則提早結束

                s_data = [(c, E[c[0], 0] + E[c[1], 1] + E[c[2], 2] + E[c[3], 3]) for c in s_combs]
                f_wins = sorted([x for x in s_data if x[1] < C[t]], key=lambda x: x[1], reverse=True)

                if f_wins:
                    chosen, _ = f_wins[0]
                    sim_wins += 1
                else:
                    s_data.sort(key=lambda x: x[1], reverse=True)
                    chosen, _ = s_data[0]

                sim_state.apply_assignment(*chosen)

            if sim_wins > best_sim_wins:
                best_sim_wins = sim_wins
                best_cand = cand
                best_cand_er = cand_er

        # Commit 最佳決策
        if best_cand_er < C[k]:
            global_wins += 1
        state.apply_assignment(*best_cand)

    run_time = time.time() - start_time
    print(f"[LMSH] Result: {global_wins} Wins | Time: {run_time:.4f} sec")
    return global_wins, run_time


# ==========================================
# 主程式執行區
# ==========================================
if __name__ == "__main__":
    cfg = BaseballConfig()
    C, E, base_eri = generate_data(cfg)

    # 執行四個模型進行比較
    pure_w, pure_t = solve_pure_greedy(cfg, C, E, base_eri)
    bench_w, bench_t = solve_benchmark_greedy(cfg, C, E, base_eri)
    lmsh_w, lmsh_t = solve_lmsh(cfg, C, E)
    gurobi_w, gurobi_t = solve_gurobi_exact(cfg, C, E)