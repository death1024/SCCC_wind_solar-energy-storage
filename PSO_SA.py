import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# PSO 参数
num_particles = 30
dim = 2  # 维度：[电池容量, 超级电容容量]
max_iter = 100
w = 0.7  # 惯性权重
c1 = 2.0  # 个人最优学习因子
c2 = 2.0  # 全局最优学习因子

# 模拟退火参数
T_initial = 100  # 初始温度
T_min = 1e-10  # 最小温度
alpha = 0.95  # 冷却系数

# 约束条件参数
P_B_max = 100  # 电池最大功率输出
P_SC_max = 50  # 超级电容最大功率输出
SOC_B_min, SOC_B_max = 0.1, 0.9  # 电池最小和最大SOC
SOC_SC_min, SOC_SC_max = 0.05, 0.95  # 超级电容最小和最大SOC

# 评价函数权重
omega_1 = 1500
omega_2 = 1200
omega_3 = 1
omega_4 = 30



def load_and_plot_wind_power_data(file_path, selected_date):
    wind_power_df = pd.read_excel(file_path, header=0, skiprows=[1], parse_dates=['时间'])
    
    # 选择指定日期的数据
    wind_power_df['日期'] = wind_power_df['时间'].dt.date
    selected_date = pd.to_datetime(selected_date).date()
    selected_day_data = wind_power_df[wind_power_df['日期'] == selected_date]
    
    # 实际功率(kW)
    selected_day_data = selected_day_data.copy()
    selected_day_data['实际功率(kW)'] = selected_day_data['实际功率(MW)'] * 1000
    
    plt.figure(figsize=(12, 6))
    plt.plot(selected_day_data['时间'], selected_day_data['实际功率(kW)'], label='Wind Power', marker='o', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.title(f'Wind Power on {selected_date}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    return selected_day_data['实际功率(kW)'].values

def adjust_wind_power(Pw):
    Po = np.zeros_like(Pw)
    Po[0:4] = Pw[0:4]
    dP_15 = 0.03
    dP_60 = 0.10
    Pr = 60
    
    for i in range(4, len(Pw)):
        if Pw[i] > Po[i-1]:
            Poi = max(Pw[i-4:i+1]) - dP_15 * Pw[i]
        elif Pw[i] < Po[i-1]:
            Poi = min(Pw[i-4:i+1]) + dP_15 * Pw[i]
        else:
            Poi = Pw[i]
        
        a = max(0, min((Poi - Po[i-1]) / (Pw[i] - Po[i-1]), 1))
        Po[i] = (1-a) * Po[i-1] + a * Pw[i]
        
        dP_15_real = (Po[i] - Po[i-1]) / Pr
        dP_60_real = (max(Po[i-4:i+1]) - min(Po[i-4:i+1])) / Pr
        
        if dP_15_real > dP_15 or dP_60_real > dP_60:
            if Po[i] < Po[i-1]:
                Po[i] = min(max(Po[i-1] - dP_15*Pr, Po[i-4] - dP_60*Pr), Po[i-1])
            elif Po[i] > Po[i-1]:
                Po[i] = max(min(Po[i-1] + dP_15*Pr, Po[i-4] + dP_60*Pr), Po[i-1])
    
    return Po


def storage_control_and_evaluation(Pw, E_B_r, E_SC_r):
    length = len(Pw)
    Po = adjust_wind_power(Pw)  # 调整后的输出功率
    Phess = Pw - Po  # 储能系统需求功率
    Ts = 15  # 采样时间，单位分钟


    # 初始化参数
    SOC_B = np.zeros(length)
    SOC_SC = np.zeros(length)
    P_B = np.zeros(length)
    P_SC = np.zeros(length)
    SOC_B[0] = 0.3  # 电池初始SOC
    SOC_SC[0] = 0.3  # 超级电容初始SOC
    eta_B = 0.8  # 电池充放电效率
    eta_SC = 0.95  # 超级电容充放电效率

    for i in range(1, length):
        # 电池充放电逻辑
        if Phess[i] > 0:  # 需要充电
            if SOC_B[i-1] >= SOC_B_max:  # 如果电池SOC过高，不再充电
                P_B[i] = 0
            else:
                P_B[i] = min(Phess[i], P_B_max)  # 根据需求和最大充电功率确定充电功率
        elif Phess[i] < 0:  # 需要放电
            if SOC_B[i-1] <= SOC_B_min:  # 如果电池SOC过低，不再放电
                P_B[i] = 0
            else:
                P_B[i] = max(Phess[i], -P_B_max)  # 根据需求和最大放电功率确定放电功率
        
        # 更新电池SOC
        if P_B[i] > 0:  # 充电
            SOC_B[i] = SOC_B[i-1] + (P_B[i] * eta_B * Ts / 60) / E_B_r
        elif P_B[i] < 0:  # 放电
            SOC_B[i] = SOC_B[i-1] + (P_B[i] / eta_B * Ts / 60) / E_B_r
        else:
            SOC_B[i] = SOC_B[i-1]

        SOC_B[i] = np.clip(SOC_B[i], SOC_B_min, SOC_B_max)  # 确保SOC在合理范围内

       # 超级电容充放电逻辑
        P_SC_remaining = Phess[i] - P_B[i]  # 剩余需求功率分配给超级电容
        if P_SC_remaining > 0:
            if SOC_SC[i-1] < SOC_SC_max:
                P_SC[i] = min(P_SC_remaining, P_SC_max)
            else:
                P_SC[i] = 0  # 如果超级电容SOC过高，不再充电
        elif P_SC_remaining < 0:
            if SOC_SC[i-1] > SOC_SC_min:
                P_SC[i] = max(P_SC_remaining, -P_SC_max)
            else:
                P_SC[i] = 0  # 如果超级电容SOC过低，不再放电
        else:
            P_SC[i] = 0

        # 更新超级电容SOC
        if P_SC[i] > 0:  # 充电
            SOC_SC[i] = SOC_SC[i-1] + (P_SC[i] * eta_SC * Ts / 60) / E_SC_r
        elif P_SC[i] < 0:  # 放电
            SOC_SC[i] = SOC_SC[i-1] + (P_SC[i] / eta_SC * Ts / 60) / E_SC_r
        else:
            SOC_SC[i] = SOC_SC[i-1]
        SOC_SC[i] = np.clip(SOC_SC[i], SOC_SC_min, SOC_SC_max)  # 确保SOC在合理范围内

    # 计算评价指标
    Delta_SOC_B = abs(SOC_B[-1] - SOC_B[0])
    Delta_SOC_SC = abs(SOC_SC[-1] - SOC_SC[0])
    cost = omega_1 * Delta_SOC_B + omega_2 * Delta_SOC_SC + omega_3 * E_B_r + omega_4 * E_SC_r  # 总成本


    return cost


class Particle:
    def __init__(self):
        self.position = np.array([random.uniform(0, P_B_max), random.uniform(0, P_SC_max)])
        self.velocity = np.array([0, 0])
        self.pbest_position = self.position.copy()
        self.pbest_value = float('inf')
    
    def evaluate(self, wind_power):
        # 计算成本
        cost = storage_control_and_evaluation(wind_power,self.position[0],self.position[1])
        if cost < self.pbest_value:
            self.pbest_position = self.position.copy()
            self.pbest_value = cost
    
    def update_velocity_position(self, gbest_position):
        r1, r2 = np.random.rand(2)
        self.velocity = w * self.velocity + c1 * r1 * (self.pbest_position - self.position) + c2 * r2 * (gbest_position - self.position)
        self.position += self.velocity
        self.position = np.clip(self.position, [0, 0], [P_B_max, P_SC_max])



def simulated_annealing(particle, T, wind_power):
    # 随机生成新位置
    new_position = particle.position + np.random.uniform(-1, 1, particle.position.shape) * T
    # 确保新位置在约束范围内
    new_position = np.clip(new_position, [0, 0], [P_B_max, P_SC_max])
    
    # 计算新位置的成本
    new_cost = storage_control_and_evaluation(wind_power,new_position[0],new_position[1])
    
    # 新成本更低/根据SA准则接受较差的解
    if new_cost < particle.pbest_value or np.exp((particle.pbest_value - new_cost) / T) > np.random.rand():
        particle.position = new_position
        particle.pbest_position = new_position.copy()
        particle.pbest_value = new_cost

def update_global_best(particles):
    global gbest_value, gbest_position
    for particle in particles:
        if particle.pbest_value < gbest_value:
            gbest_position = particle.pbest_position.copy()
            gbest_value = particle.pbest_value
    return gbest_position, gbest_value

def pso_with_sa(wind_power):
    global gbest_position, gbest_value
    particles = [Particle() for _ in range(num_particles)]
    
    gbest_value = float('inf')
    gbest_position = None

    # 确保在算法开始前，每个粒子都被评估一次，并尝试更新全局最佳值
    for particle in particles:
        particle.evaluate(wind_power)
    gbest_position, gbest_value = update_global_best(particles)
    
    for it in range(max_iter):
        for particle in particles:
            particle.update_velocity_position(gbest_position)
            particle.evaluate(wind_power)
        gbest_position, gbest_value = update_global_best(particles)
        
        T = T_initial * (alpha ** it)
        for particle in particles:
            simulated_annealing(particle, T, wind_power)

        print(f"Iteration {it+1}: Global Best Value = {gbest_value}")

    return gbest_position, gbest_value
if __name__ == "__main__":
    file_path = '附件2-场站出力.xlsx' 
    selected_date = '2019-01-02'  
    wind_power = load_and_plot_wind_power_data(file_path, selected_date)

    gbest_position, gbest_value = pso_with_sa(wind_power)

    print(f"Optimal Configuration: Battery Capacity = {gbest_position[0]} MWh, Supercapacitor Capacity = {gbest_position[1]} MWh")
    print(f"Minimum Cost: {gbest_value}")
