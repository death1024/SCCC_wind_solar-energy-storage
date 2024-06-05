import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# 设置matplotlib的字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体为微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 读取Excel文件中的数据
Pw = pd.read_excel('附件2-场站出力.xlsx', usecols='B', skiprows=386, nrows=288, engine='openpyxl').to_numpy().flatten()

# 生成每分钟的功率波动特征
def generate_features_labels(Pw, step=15):
    features = []
    labels = []
    for i in range(len(Pw) - 1):
        for j in range(1, step):
            features.append([Pw[i], Pw[i + 1], j])
            delta = (Pw[i + 1] - Pw[i]) * (j / step) * (0.5 + np.random.rand())  # 随机波动
            labels.append(Pw[i] + delta)
    return np.array(features), np.array(labels)

# 训练随机森林模型
def train_random_forest(features, labels):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features, labels)
    return model

# 生成每分钟波动的特征和标签
features_1min, labels_1min = generate_features_labels(Pw,step=15)
model_1min = train_random_forest(features_1min, labels_1min)

# 使用模型预测
Pw1 = np.zeros(len(Pw) * 15)
for i in range(1, len(Pw) * 15):
    if i % 15 == 1:
        Pw1[i - 1] = Pw[i // 15]
    else:
        Pw1[i - 1] = model_1min.predict([[Pw[i // 15], Pw[min(i // 15 + 1, len(Pw) - 1)], i % 15]])[0]
    Pw1[i - 1] = Pw1[i - 1]  # 调整到30MW

# 生成每10秒波动的特征和标签
features_10sec, labels_10sec = generate_features_labels(Pw1, step=6)
model_10sec = train_random_forest(features_10sec, labels_10sec)

# 使用模型预测

Pw2 = np.zeros(len(Pw1) * 6)
for i in range(1, len(Pw1) * 6):
    if i % 6 == 1:
        Pw2[i - 1] = Pw1[i // 6]
    else:
        Pw2[i - 1] = model_10sec.predict([[Pw1[i // 6], Pw1[min(i // 6 + 1, len(Pw1) - 1)], i % 6]])[0]

plt.figure(1)
plt.plot(range(len(Pw2)), Pw2)
plt.grid(True)
plt.xticks(ticks=np.arange(0, len(Pw2)+1, 720*6), labels=['0', '12', '24', '36', '48', '60', '72'])
plt.xlabel('时间/h')
plt.ylabel('有功功率/MW')
plt.title('2019-01-05 ~ 2019-01-07 功率输入')

# 滑动平均
dP_1 = 80/10
dP_10 = 80/3
Pg = Pw2.copy()
N0 = 26
Nmax = 120
Nmin = 25
N = np.full(len(Pw2), 26)
k = 60
while k < len(Pw2) :
    Pg[k] = np.mean(Pw2[k - N[k] + 1:k + 1])  # 初始滑动平均并网功率
    # 计算功率波动
    dP_1_real = max(Pg[k - 5:k + 1]) - min(Pg[k - 5:k + 1])
    dP_10_real = max(Pg[k - 59:k + 1]) - min(Pg[k - 59:k + 1])
        
    # 不满足约束
    if dP_1_real > dP_1 or dP_10_real > dP_10:
        if N[k] < N0:
            N[k] += 1
        elif N[k] == Nmax:
            N[k] = Nmax
        else:
            N[k] += 1
            k -= 1
    else:  # 满足约束
        if N[k] > N0:
            pass
        elif N[k] == Nmin:
            N[k] = Nmin
        else:
            N[k] -= 1
            k -= 1
    k += 1

plt.figure(2)
plt.plot(range(len(Pw2)), Pg, label='滑动平均功率')
plt.plot(range(len(Pw2)), Pw2 - Pg, label='功率波动')
plt.legend()
plt.grid(True)
plt.xticks(ticks=np.arange(0, len(Pw2)+1, 720*6), labels=['0', '12', '24', '36', '48', '60', '72'])
plt.xlabel('时间/h')
plt.ylabel('有功功率/MW')
plt.title('功率波动平滑处理')

# 计算功率波动
dP_1_final = np.zeros(len(Pw2))
dP_10_final = np.zeros(len(Pw2))
for k in range(59, len(Pw2)):
    dP_1_final[k] = np.max(Pg[max(0, k - 5):k + 1]) - np.min(Pg[max(0, k - 5):k + 1])
    dP_10_final[k] = np.max(Pg[max(0, k - 59):k + 1]) - np.min(Pg[max(0, k - 59):k + 1])

plt.figure(3)
plt.plot(range(len(Pw2)), dP_1_final, label='一分钟功率波动')
plt.plot(range(len(Pw2)), dP_10_final, label='十分钟功率波动')
plt.legend()
plt.grid(True)
plt.xticks(ticks=np.arange(0, len(Pw2)+1, 720*6), labels=['0', '12', '24', '36', '48', '60', '72'])
plt.xlabel('时间/h')
plt.ylabel('有功功率/MW')
plt.title('功率波动分析')

plt.show()
