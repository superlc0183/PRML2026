import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ==========================================
# 1. 数据预处理 (与刚刚一致)
# ==========================================
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f'var{j+1}(t-{i})' for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [f'var{j+1}(t)' for j in range(n_vars)]
        else:
            names += [f'var{j+1}(t+{i})' for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

dataset = pd.read_csv('LSTM-Multivariate_pollution.csv', header=0, index_col=0)
values = dataset.values
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
values = values.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)

# ==========================================
# 2. 划分训练集与测试集
# ==========================================
values = reframed.values
# 我们用前 4 年的数据来训练 (365天 * 24小时 * 4年)，剩下的第 5 年作为测试集
n_train_hours = 365 * 24 * 4
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# 拆分输入(X)和标签(y)
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# ★ 关键步骤：将数据重塑为 LSTM 期待的 3D 格式 [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print("训练集维度:", train_X.shape)
print("测试集维度:", test_X.shape)

# ==========================================
# 3. 构建并训练 LSTM 模型
# ==========================================
model = Sequential()
# 添加一个包含 50 个神经元的 LSTM 层
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
# 添加一个全连接输出层，输出 1 个预测值
model.add(Dense(1))

# 使用 MAE (平均绝对误差) 作为损失函数，Adam 作为优化器
model.compile(loss='mae', optimizer='adam')

print("\n🚀 开始训练 LSTM 模型...")
# 训练 50 个 epoch
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# ==========================================
# 4. 可视化训练过程
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test/Validation Loss')
plt.title('LSTM Model Loss (Training vs Validation)')
plt.ylabel('Loss (MAE)')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('lstm_loss.png', dpi=300)
print("\n✅ 训练完成！Loss 曲线已保存为 'lstm_loss.png'")

import math
from sklearn.metrics import mean_squared_error

# ==========================================
# 5. 模型预测与反归一化
# ==========================================
print("\n=== 开始进行测试集预测 ===")
yhat = model.predict(test_X)
# 将 3D 的 test_X 变回 2D，方便提取后面的气象特征列
test_X_reshaped = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# ★ 反归一化预测值 (yhat)
# 我们必须把预测出来的 1 列 PM2.5，和测试集原有的 7 列气象数据拼起来，凑够 8 列，才能调用 inverse_transform
inv_yhat = np.concatenate((yhat, test_X_reshaped[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0] # 提取出第一列，也就是还原后的真实刻度下的 PM2.5 预测值

# ★ 反归一化真实标签 (test_y)
test_y_reshaped = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y_reshaped, test_X_reshaped[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0] # 提取出真实的 PM2.5 值

# 计算 RMSE (均方根误差)
rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
print(f'===> 测试集最终 RMSE (均方根误差): {rmse:.3f}')

# ==========================================
# 6. 预测结果可视化对比
# ==========================================
# 测试集有 8000 多个小时，全画在一张图里会变成黑乎乎的一团
# 我们只截取测试集的前 200 个小时（大约 8 天）的连续数据来对比细节
plt.figure(figsize=(15, 6))
plt.plot(inv_y[:200], label='Actual PM2.5 (True Values)', color='blue', alpha=0.6, linewidth=2)
plt.plot(inv_yhat[:200], label='Predicted PM2.5 (LSTM)', color='red', alpha=0.8, linestyle='--', linewidth=2)
plt.title('PM2.5 Forecasting: Actual vs Predicted (First 200 Hours of Test Set)', fontsize=16)
plt.xlabel('Time (Hours)', fontsize=12)
plt.ylabel('PM2.5 Concentration', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('prediction_comparison.png', dpi=300)
print("\n✅ 预测对比图已保存为：prediction_comparison.png")