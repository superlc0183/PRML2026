import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

# 1. 导入数据
train_df = pd.read_excel('Data4Regression.xlsx', sheet_name=0) # 读取第一个表单 (训练集)
test_df = pd.read_excel('Data4Regression.xlsx', sheet_name=1)  # 读取第二个表单 (测试集)

# 提取特征并 reshape
X_train = train_df['x'].values.reshape(-1, 1)
y_train = train_df['y_complex'].values.reshape(-1, 1)
X_test = test_df['x_new'].values.reshape(-1, 1)
y_test = test_df['y_new_complex'].values.reshape(-1, 1)

# 添加偏置项 (bias) x_0 = 1
X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]
X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]

# ================= 任务一：线性拟合 ================= #
# 1.1 最小二乘法 (Least Squares)
w_ls = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)
y_train_pred_ls = X_train_b.dot(w_ls)
y_test_pred_ls = X_test_b.dot(w_ls)

# 1.2 梯度下降法 (Gradient Descent)
np.random.seed(42)
w_gd = np.random.randn(2, 1)
lr = 0.01
epochs = 1000
for epoch in range(epochs):
    gradients = 2/len(X_train) * X_train_b.T.dot(X_train_b.dot(w_gd) - y_train)
    w_gd -= lr * gradients
y_train_pred_gd = X_train_b.dot(w_gd)
y_test_pred_gd = X_test_b.dot(w_gd)

# 1.3 牛顿法 (Newton's Method)
w_newton = np.random.randn(2, 1)
H = 2/len(X_train) * X_train_b.T.dot(X_train_b) # Hessian Matrix
grad = 2/len(X_train) * X_train_b.T.dot(X_train_b.dot(w_newton) - y_train)
w_newton = w_newton - np.linalg.inv(H).dot(grad)
y_train_pred_newton = X_train_b.dot(w_newton)
y_test_pred_newton = X_test_b.dot(w_newton)

print("--- 线性模型评估 (MSE) ---")
print(f"Least Squares  - Train: {mean_squared_error(y_train, y_train_pred_ls):.4f}, Test: {mean_squared_error(y_test, y_test_pred_ls):.4f}")
print(f"Gradient Desc. - Train: {mean_squared_error(y_train, y_train_pred_gd):.4f}, Test: {mean_squared_error(y_test, y_test_pred_gd):.4f}")
print(f"Newton's Method- Train: {mean_squared_error(y_train, y_train_pred_newton):.4f}, Test: {mean_squared_error(y_test, y_test_pred_newton):.4f}")

# ================= 任务二：非线性拟合 ================= #
# 使用多层感知机 (MLP)
mlp = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', max_iter=2000, random_state=42)
mlp.fit(X_train, y_train.ravel())

y_train_pred_mlp = mlp.predict(X_train)
y_test_pred_mlp = mlp.predict(X_test)

print("\n--- 非线性模型评估 (MSE) ---")
print(f"MLP Regressor  - Train: {mean_squared_error(y_train, y_train_pred_mlp):.4f}, Test: {mean_squared_error(y_test, y_test_pred_mlp):.4f}")

# ================= 绘图与可视化 ================= #
plt.figure(figsize=(12, 5))

# 图1：线性拟合展示
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, label='Train Data', color='gray', alpha=0.5)
plt.plot(X_train, y_train_pred_ls, label='Linear Fit (LS)', color='red', linewidth=2)
plt.title('Linear Models (Underfitting)')
plt.legend()

# 图2：非线性模型拟合展示
X_range = np.linspace(X_train.min(), X_train.max(), 500).reshape(-1, 1)
y_range_mlp = mlp.predict(X_range)
plt.subplot(1, 2, 2)
plt.scatter(X_train, y_train, label='Train Data', color='gray', alpha=0.5)
plt.plot(X_range, y_range_mlp, label='MLP Regression', color='blue', linewidth=2)
plt.title('Non-Linear Fit (MLP)')
plt.legend()

plt.tight_layout()
plt.savefig('fit_results.png') # 保存图表供Latex使用
plt.show()