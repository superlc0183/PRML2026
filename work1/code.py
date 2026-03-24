import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# 1. 读取数据
train_df = pd.read_excel('Data4Regression.xlsx', sheet_name=0)
test_df = pd.read_excel('Data4Regression.xlsx', sheet_name=1)

X_train = train_df['x'].values.reshape(-1, 1)
y_train = train_df['y_complex'].values.reshape(-1, 1)
X_test = test_df['x_new'].values.reshape(-1, 1)
y_test = test_df['y_new_complex'].values.reshape(-1, 1)

N_train = len(X_train)
X_train_b = np.c_[np.ones((N_train, 1)), X_train]
X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]

# ================= M1: 线性模型 (LS, GD, Newton 结果一致) =================
w_ls = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)
y_train_ls = X_train_b.dot(w_ls)
y_test_ls = X_test_b.dot(w_ls)

# ================= M2: 基础非线性 - 多项式回归 (Degree=6) =================
poly = PolynomialFeatures(degree=6)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model_poly = LinearRegression().fit(X_train_poly, y_train)
y_train_poly = model_poly.predict(X_train_poly)
y_test_poly = model_poly.predict(X_test_poly)

# ================= M3: 贝叶斯多项式回归 =================
model_bayes = BayesianRidge()
model_bayes.fit(X_train_poly, y_train.ravel())
y_train_bayes = model_bayes.predict(X_train_poly)
y_test_bayes = model_bayes.predict(X_test_poly)

# ================= M4: 核岭回归 (RBF Kernel) =================
model_kr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.5)
model_kr.fit(X_train, y_train)
y_train_kr = model_kr.predict(X_train)
y_test_kr = model_kr.predict(X_test)

# ================= M5: 支持向量回归 (SVR) =================
model_svr = SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma=0.5)
model_svr.fit(X_train, y_train.ravel())
y_train_svr = model_svr.predict(X_train)
y_test_svr = model_svr.predict(X_test)

# ================= M6: 多层感知机 (MLP 神经网络) =================
model_mlp = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', max_iter=3000, random_state=42)
model_mlp.fit(X_train, y_train.ravel())
y_train_mlp = model_mlp.predict(X_train)
y_test_mlp = model_mlp.predict(X_test)

# ================= 评估与打印输出 =================
def evaluate(name, y_tr, y_te):
    mse_tr = mean_squared_error(y_train, y_tr)
    mse_te = mean_squared_error(y_test, y_te)
    r2_te = r2_score(y_test, y_te)
    print(f"{name:20s} | Train MSE: {mse_tr:.4f} | Test MSE: {mse_te:.4f} | Test R2: {r2_te:.4f}")

print("-" * 65)
print("模型名称                 | 训练 MSE  | 测试 MSE  | 测试 R2")
print("-" * 65)
evaluate("Linear (LS/GD/Newton)", y_train_ls, y_test_ls)
evaluate("Polynomial (Deg=6)", y_train_poly, y_test_poly)
evaluate("Bayesian Poly", y_train_bayes, y_test_bayes)
evaluate("Kernel Ridge (RBF)", y_train_kr, y_test_kr)
evaluate("SVR (RBF)", y_train_svr, y_test_svr)
evaluate("MLP Neural Network", y_train_mlp, y_test_mlp)
print("-" * 65)

# ================= 可视化绘图 (双子图对比) =================
X_range = np.linspace(X_train.min()-0.5, X_train.max()+0.5, 500).reshape(-1, 1)
y_range_ls = np.c_[np.ones((500, 1)), X_range].dot(w_ls)
y_range_poly = model_poly.predict(poly.transform(X_range))
y_range_bayes, y_std = model_bayes.predict(poly.transform(X_range), return_std=True)
y_range_kr = model_kr.predict(X_range)
y_range_svr = model_svr.predict(X_range)
y_range_mlp = model_mlp.predict(X_range)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 图1：线性 vs 多项式 vs 贝叶斯
axes[0].scatter(X_train, y_train, color='lightgray', label='Train Data')
axes[0].scatter(X_test, y_test, color='gray', marker='x', label='Test Data')
axes[0].plot(X_range, y_range_ls, 'r--', label='Linear Fit')
axes[0].plot(X_range, y_range_poly, 'b-', label='Polynomial (Deg=6)')
axes[0].plot(X_range, y_range_bayes, 'g-', label='Bayesian Poly')
axes[0].fill_between(X_range.ravel(), y_range_bayes - y_std, y_range_bayes + y_std, color='green', alpha=0.1, label='Bayesian Uncertainty')
axes[0].set_title('Linear vs Polynomial vs Bayesian')
axes[0].legend()
axes[0].grid(True, linestyle=':', alpha=0.6)

# 图2：高级核方法 vs 深度学习
axes[1].scatter(X_train, y_train, color='lightgray', label='Train Data')
axes[1].scatter(X_test, y_test, color='gray', marker='x', label='Test Data')
axes[1].plot(X_range, y_range_kr, 'c-', linewidth=2, label='Kernel Ridge (RBF)')
axes[1].plot(X_range, y_range_svr, 'm-', linewidth=2, label='SVR (RBF)')
axes[1].plot(X_range, y_range_mlp, 'purple', linestyle='-.', linewidth=2, label='MLP Network')
axes[1].set_title('Kernel Methods vs Deep Learning')
axes[1].legend()
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('fit_results.png', dpi=300)
plt.show()