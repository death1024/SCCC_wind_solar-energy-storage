import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = '附件2-场站出力.xlsx'
wind_power_data = pd.read_excel(data_path, header=0, skiprows=[1], parse_dates=['时间'])

#起止时间修改就行
start_date = pd.Timestamp('2019-01-28')
end_date = pd.Timestamp('2019-01-30')
daily_data = wind_power_data[(wind_power_data['时间'] >= start_date) & (wind_power_data['时间'] <= end_date)]

daily_power_data = daily_data['实际功率(MW)'].astype(float).values
daily_time_data = daily_data['时间']

# 卡尔曼滤波器参数
initial_state_mean = daily_power_data[0]
initial_state_covariance = 1.0  # 初始状态协方差
observation_covariance = np.var(daily_power_data) * 0.1  # 观测协方差
transition_covariance = np.var(daily_power_data) * 0.01  # 过程协方差

n = len(daily_power_data)

filtered_state_estimates = np.zeros(n)


current_state_estimate = initial_state_mean
current_covariance_estimate = initial_state_covariance

for t in range(n):
    kalman_gain = current_covariance_estimate / (current_covariance_estimate + observation_covariance)
    current_state_estimate = current_state_estimate + kalman_gain * (daily_power_data[t] - current_state_estimate)
    current_covariance_estimate = (1 - kalman_gain) * current_covariance_estimate
    current_covariance_estimate += transition_covariance
    filtered_state_estimates[t] = current_state_estimate


plt.figure(figsize=(15, 7))
plt.plot(daily_time_data, daily_power_data, label='Original Data', color='lightgray')
plt.plot(daily_time_data, filtered_state_estimates, label='Kalman Filter Estimate', color='darkorange', linewidth=2)
plt.title('Wind Power Data - Kalman Filter Smoothing')
plt.xlabel('Time')
plt.ylabel('Power (MW)')
plt.xticks(rotation=45)  
plt.legend()
plt.tight_layout()  
plt.show()




