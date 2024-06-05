import pandas as pd
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# 读取储能需求数据
file_path = '储能需求.xlsx'  # 确保路径与您的环境相匹配
hess_data = pd.read_excel(file_path)

# 提取储能出力作为净功率需求
Phess = hess_data['储能出力(MW)'].values

# 系统参数
SOC_B_min, SOC_B_max = 0.1, 0.9
SOC_SC_min, SOC_SC_max = 0.05, 0.95
eta_B_charge, eta_B_discharge = 0.9, 0.85
eta_SC_charge, eta_SC_discharge = 0.95, 0.9
E_BN, E_SCN = 100, 50  # 额定容量，MWh
P_BN, P_SCN = 10, 20  # 额定功率，MW
Ts = 15 / 60  # 采样时间，小时

N = len(Phess)  # 时间步数量

# 定义优化变量
SOC_B = cp.Variable(N+1)
SOC_SC = cp.Variable(N+1)
P_B_charge = cp.Variable(N, nonneg=True)
P_B_discharge = cp.Variable(N, nonneg=True)
P_SC_charge = cp.Variable(N, nonneg=True)
P_SC_discharge = cp.Variable(N, nonneg=True)

# 初始SOC约束
constraints = [SOC_B[0] == 0.3, SOC_SC[0] == 0.3]

# 构建目标函数和约束
cost = 0
for t in range(N):
    # 使用新的变量构建目标函数
    cost += P_B_charge[t] / eta_B_charge + P_B_discharge[t] * eta_B_discharge
    cost += P_SC_charge[t] / eta_SC_charge + P_SC_discharge[t] * eta_SC_discharge
    cost += cp.square(SOC_B[t] - 0.5) + cp.square(SOC_SC[t] - 0.5)

    # SOC更新约束，以及新的充放电功率约束
    constraints += [
        SOC_B[t+1] == SOC_B[t] + Ts * (P_B_charge[t] / eta_B_charge - P_B_discharge[t] * eta_B_discharge) / E_BN,
        SOC_SC[t+1] == SOC_SC[t] + Ts * (P_SC_charge[t] / eta_SC_charge - P_SC_discharge[t] * eta_SC_discharge) / E_SCN,
        SOC_B_min <= SOC_B[t+1], SOC_B[t+1] <= SOC_B_max,
        SOC_SC_min <= SOC_SC[t+1], SOC_SC[t+1] <= SOC_SC_max,
        P_B_charge[t] <= P_BN, P_B_discharge[t] <= P_BN,
        P_SC_charge[t] <= P_SCN, P_SC_discharge[t] <= P_SCN,
        P_B_charge[t] - P_B_discharge[t] + P_SC_charge[t] - P_SC_discharge[t] + Phess[t] == 0,
    ]

# 优化问题
prob = cp.Problem(cp.Minimize(cost), constraints)

# 求解
prob.solve()

# 结果可视化
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(SOC_B.value[:-1], label='Battery SOC')
plt.plot(SOC_SC.value[:-1], label='Supercapacitor SOC')
plt.ylabel('SOC')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(P_B_charge.value - P_B_discharge.value, label='Battery Power')
plt.plot(P_SC_charge.value - P_SC_discharge.value, label='Supercapacitor Power')
plt.plot(Phess, label='Net Power Demand', linestyle='--')
plt.ylabel('Power (MW)')
plt.legend()

plt.show()

