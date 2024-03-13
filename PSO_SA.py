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
P_B_max = 10  # 电池最大功率输出，单位MW
P_SC_max = 20  # 超级电容最大功率输出，单位MW

E_BN = 100  # 电池额定容量，单位MWh
E_SCN = 50  # 超级电容额定容量，单位MWh
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
    Tmin = 60 # 最小充放电循环时间(min)
    StateCount = 1
    StateChangeCount = 0 # 充放电转换次数

    # 初始化参数
    SOC_B = np.zeros(length)
    SOC_SC = np.zeros(length)
    P_B = np.zeros(length)
    P_SC = np.zeros(length)
    E_B = np.zeros(length)
    E_SC = np.zeros(length)
    C = np.zeros(length) # 电池损耗系数
    SOC_B[0] = 0.3  # 电池初始SOC
    SOC_SC[0] = 0.3  # 超级电容初始SOC
    eta_B = 0.8  # 电池充放电效率
    eta_SC = 0.95  # 超级电容充放电效率

    if (Phess[0]<0):
        C[0] = SOC_B_max/(SOC_B[0]^2)
    else:
        C[0] = SOC_B[0]/(SOC_B[0]*SOC_B_max)

    E_B[0] = E_BN * SOC_B[0]
    E_SC[0] = E_SCN * SOC_SC[0]

    for i in range(1, len(Phess)):
        if Phess[i] > 0 and SOC_B[i-1] >= SOC_B_max:
            # Po<Pw,需要充电,但是SOC过高
            P_B[i] = 0  # 不吸收功率
            SOC_B[i] = SOC_B[i-1]
        elif Phess[i] < 0 and SOC_B[i-1] <= SOC_B_min:
            # Po>Pw,需要放电,但是SOC过低
            P_B[i] = 0  # 不释放功率
            SOC_B[i] = SOC_B[i-1]
        else:
            # 确定参考滤波系数，低频功率
            a = min((2 * np.pi * 15 / (2 * np.pi * 15 + Tmin)) / (2 * C[i-1]), 1/15)
            P_B[i] = min((1-a) * P_B[i-1] + a * Phess[i], P_B_max)
            # 检查充放电间隔
            if StateCount < 2 and np.sign(P_B[i]) != np.sign(P_B[i-1]):
                P_B[i] = P_B[i-1]  # 如果时间过短，不能变功率

            # 更新荷电状态
            if P_B[i] > 0:  # 充电
                SOC_B[i] = SOC_B[i-1] + (eta_B * P_B[i] * Ts / 60) / E_B_r
            elif P_B[i] < 0:  # 放电
                SOC_B[i] = SOC_B[i-1] + ((1 / eta_B) * P_B[i] * Ts / 60) / E_B_r
            else:
                SOC_B[i] = SOC_B[i-1]

            # 检查荷电状态并修正功率
            if SOC_B[i] > SOC_B_max:
                P_B[i] = (SOC_B_max - SOC_B[i-1]) * E_B_r / eta_B
                SOC_B[i] = SOC_B_max
            elif SOC_B[i] < SOC_B_min:
                P_B[i] = (SOC_B[i-1] - SOC_B_min) * E_B_r * eta_B
                SOC_B[i] = SOC_B_min

    # 更新充放电间隔
        if P_B[i] * P_B[i-1] > 0 or (P_B[i] == 0 and P_B[i-1] == 0):
            StateCount += 1
        else:
            StateChangeCount += 1
            StateCount = 0

        # 超级电容
        P_SC[i] = min(Phess[i] - P_B[i], P_SC_max)
        if P_SC[i] > 0:  # 充电
            SOC_SC[i] = SOC_SC[i-1] + (eta_SC * P_SC[i] * Ts / 60) / E_SC_r
        elif P_SC[i] < 0:  # 放电
            SOC_SC[i] = SOC_SC[i-1] + ((1 / eta_SC) * P_SC[i] * Ts / 60) / E_SC_r
        else:
            SOC_SC[i] = SOC_SC[i-1]

    # 超级电容状态检验
        if SOC_SC[i] > SOC_SC_max:
            P_SC[i] = (SOC_SC_max - SOC_SC[i-1]) * E_SC_r / eta_SC
            SOC_SC[i] = SOC_SC_max
        elif SOC_SC[i] < SOC_SC_min:
            P_SC[i] = (SOC_SC[i-1] - SOC_SC_min) * E_SC_r * eta_SC
            SOC_SC[i] = SOC_SC_min

    # 更新损耗系数
        if P_B[i] > 0:
            C[i] = SOC_B_max / (SOC_B[i] * SOC_B[i-1])
        else:
            C[i] = SOC_B[i-1] / (SOC_B_max * SOC_B[i])

    # 计算评价指标
    Delta_SOC_B = abs(SOC_B[-1] - SOC_B[0])
    Delta_SOC_SC = abs(SOC_SC[-1] - SOC_SC[0])
    cost = omega_1 * Delta_SOC_B + omega_2 * Delta_SOC_SC + omega_3 * E_B_r + omega_4 * E_SC_r  # 总成本


    return cost


class Particle:
    def __init__(self):
        self.position = np.array([random.uniform(0, E_BN), random.uniform(0, E_SCN)])
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
        self.position = np.clip(self.position, [0, 0], [E_BN,  E_SCN])



def simulated_annealing(particle, T, wind_power):
    # 随机生成新位置
    new_position = particle.position + np.random.uniform(-1, 1, particle.position.shape) * T
    # 确保新位置在约束范围内
    new_position = np.clip(new_position, [0, 0], [E_BN, E_SCN])
    
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
