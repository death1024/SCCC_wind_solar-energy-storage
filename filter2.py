import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# 设置matplotlib的字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体为微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 读取Excel文件中的数据
Pw = pd.read_excel('附件2-场站出力.xlsx', usecols='B', skiprows=386, nrows=288, engine='openpyxl').to_numpy().flatten()
Pw_3 = pd.read_excel('附件2-场站出力.xlsx', usecols='B', skiprows=386, nrows=288, engine='openpyxl').to_numpy().flatten()

# 为数据添加随机噪声
noise_level = 1.5
Pw_noisy = Pw + noise_level * np.random.randn(len(Pw))


# 生成每分钟的功率波动特征
def generate_features_labels(Pw, step):
    features = []
    labels = []
    # Introduce rolling and exponential moving averages
    rolling_avg = pd.Series(Pw).rolling(window=3, min_periods=1).mean().values
    exp_moving_avg = pd.Series(Pw).ewm(span=3, adjust=False).mean().values
    
    for i in range(len(Pw) - step):
        for j in range(1, step):
            deltaP = Pw[i + 1] - Pw[i]
            ratioP = Pw[i + 1] / (Pw[i] + 0.001)
            features.append([Pw[i], Pw[i + 1], deltaP, ratioP, rolling_avg[i], exp_moving_avg[i], j])
            delta = deltaP * (j / step) * (0.5 + np.random.rand())  # 随机波动
            labels.append(Pw[i] + delta)
    return np.array(features), np.array(labels)

# 训练随机森林模型
def train_random_forest(features, labels):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(features, labels)
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    return best_model

Pw1 = np.zeros(len(Pw) * 6)
# 生成每分钟波动的特征和标签
features_1min, labels_1min = generate_features_labels(Pw,step=15)
model_1min = train_random_forest(features_1min, labels_1min)


def predict_with_features(Pw, model, features_1min):
    Pw_extended = np.zeros(len(Pw) * 15)
    rolling_avg = pd.Series(Pw).rolling(window=3, min_periods=1).mean().values
    exp_moving_avg = pd.Series(Pw).ewm(span=3, adjust=False).mean().values
    
    for i in range(1, len(Pw) * 15):
        index = i // 15
        next_index = min(index + 1, len(Pw) - 1)
        deltaP = Pw[next_index] - Pw[index]
        ratioP = Pw[next_index] / (Pw[index] + 0.001)
        if i % 15 == 1:
            Pw_extended[i - 1] = Pw[index]
        else:
            features = [Pw[index], Pw[next_index], deltaP, ratioP, rolling_avg[index], exp_moving_avg[index], i % 15]
            predicted = model.predict([features])[0]
            Pw_extended[i - 1] = predicted
    return Pw_extended

Pw1 = predict_with_features(Pw, model_1min, features_1min)
# 生成每10秒波动的特征和标签
features_10sec, labels_10sec = generate_features_labels(Pw1,step=6)
model_10sec = train_random_forest(features_10sec, labels_10sec)

Pw2 = np.zeros(len(Pw1) * 6)

def predict_with_features(Pw, model, features_1min):
    Pw_extended = np.zeros(len(Pw) * 6)
    rolling_avg = pd.Series(Pw).rolling(window=3, min_periods=1).mean().values
    exp_moving_avg = pd.Series(Pw).ewm(span=3, adjust=False).mean().values
    
    for i in range(1, len(Pw) * 6):
        index = i // 6
        next_index = min(index + 1, len(Pw) - 1)
        deltaP = Pw[next_index] - Pw[index]
        ratioP = Pw[next_index] / (Pw[index] + 0.001)
        if i % 6 == 1:
            Pw_extended[i - 1] = Pw[index]
        else:
            features = [Pw[index], Pw[next_index], deltaP, ratioP, rolling_avg[index], exp_moving_avg[index], i % 6]
            predicted = model.predict([features])[0]
            Pw_extended[i - 1] = predicted
    return Pw_extended

Pw2 = predict_with_features(Pw1, model_10sec, features_10sec)

Pw3 = np.zeros(len(Pw_noisy) * 6)
# 生成每分钟波动的特征和标签
features_1min, labels_1min = generate_features_labels(Pw_noisy,step=15)
model_1min = train_random_forest(features_1min, labels_1min)


def predict_with_features(Pw, model, features_1min):
    Pw_extended = np.zeros(len(Pw) * 15)
    rolling_avg = pd.Series(Pw).rolling(window=3, min_periods=1).mean().values
    exp_moving_avg = pd.Series(Pw).ewm(span=3, adjust=False).mean().values
    
    for i in range(1, len(Pw) * 15):
        index = i // 15
        next_index = min(index + 1, len(Pw) - 1)
        deltaP = Pw[next_index] - Pw[index]
        ratioP = Pw[next_index] / (Pw[index] + 0.001)
        if i % 15 == 1:
            Pw_extended[i - 1] = Pw[index]
        else:
            features = [Pw[index], Pw[next_index], deltaP, ratioP, rolling_avg[index], exp_moving_avg[index], i % 15]
            predicted = model.predict([features])[0]
            Pw_extended[i - 1] = predicted
    return Pw_extended

Pw3 = predict_with_features(Pw_noisy, model_1min, features_1min)
# 生成每10秒波动的特征和标签
features_10sec, labels_10sec = generate_features_labels(Pw3,step=6)
model_10sec = train_random_forest(features_10sec, labels_10sec)

Pw4 = np.zeros(len(Pw3) * 6)

def predict_with_features(Pw, model, features_1min):
    Pw_extended = np.zeros(len(Pw) * 6)
    rolling_avg = pd.Series(Pw).rolling(window=3, min_periods=1).mean().values
    exp_moving_avg = pd.Series(Pw).ewm(span=3, adjust=False).mean().values
    
    for i in range(1, len(Pw) * 6):
        index = i // 6
        next_index = min(index + 1, len(Pw) - 1)
        deltaP = Pw[next_index] - Pw[index]
        ratioP = Pw[next_index] / (Pw[index] + 0.001)
        if i % 6 == 1:
            Pw_extended[i - 1] = Pw[index]
        else:
            features = [Pw[index], Pw[next_index], deltaP, ratioP, rolling_avg[index], exp_moving_avg[index], i % 6]
            predicted = model.predict([features])[0]
            Pw_extended[i - 1] = predicted
    return Pw_extended

Pw4 = predict_with_features(Pw3, model_10sec, features_10sec)

plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(range(len(Pw_3)), Pw_3, label='原始数据', linestyle='-', color='blue')
plt.title('原始数据')
plt.xlabel('时间/h')
plt.ylabel('有功功率/MW')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(range(len(Pw_noisy)), Pw_noisy, label='原始数据', linestyle='-', color='blue')
plt.title('原始数据')
plt.xlabel('时间/h')
plt.ylabel('有功功率/MW')
plt.grid(True)

plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(range(len(Pw2)), Pw2, label='预测数据', linestyle='-', color='red')
plt.xticks(ticks=np.arange(0, len(Pw2)+1, 720*6), labels=['0', '12', '24', '36', '48', '60', '72'])
plt.title('随机森林预测数据')
plt.xlabel('时间/h')
plt.ylabel('有功功率/MW')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(range(len(Pw4)), Pw4, label='预测数据', linestyle='-', color='red')
plt.xticks(ticks=np.arange(0, len(Pw4)+1, 720*6), labels=['0', '12', '24', '36', '48', '60', '72'])
plt.title('随机森林预测数据')
plt.xlabel('时间/h')
plt.ylabel('有功功率/MW')
plt.grid(True)


plt.tight_layout()
plt.show()