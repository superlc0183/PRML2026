import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge

# 1. 读取数据 (请确保文件名与你本地一致)


train_df = pd.read_excel('Data4Regression.xlsx', sheet_name=0)
test_df = pd.read_excel('Data4Regression.xlsx', sheet_name=1)


X_train = train_df['x'].values.reshape(-1, 1)
y_train = train_df['y_complex'].values.reshape(-1, 1)
X_test = test_df['x_new'].values.reshape(-1, 1)
y_test = test_df['y_new_complex'].values.reshape(-1, 1)

# 添加偏置项 x_0 = 1
X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]
X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]

# ================= 1. 线性模型 (LS, GD, Newton) =================
# 1.1 Least Squares
w_ls = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)
y_train_pred_ls = X_train_b.dot(w_ls)
y_test_pred_ls = X_test_b.dot(w_ls)

# 1.2 Gradient Descent
np.random.seed(42)
w_gd = np.random.randn(2, 1)
lr = 0.01
for epoch in range(2000):
    gradients = 2/len(X_train) * X_train_b.T.dot(X_train_b.dot(w_gd) - y_train)
    w_gd -= lr * gradients
y_test_pred_gd = X_test_b.dot(w_gd)

# 1.3 Newton's Method
w_newton = np.random.randn(2, 1)
H = 2/len(X_train) * X_train_b.T.dot(X_train_b)
grad = 2/len(X_train) * X_train_b.T.dot(X_train_b.dot(w_newton) - y_train)
w_newton = w_newton - np.linalg.inv(H).dot(grad)
y_test_pred_newton = X_test_b.dot(w_newton)

print(f"--- Linear Model Parameters ---")
print(f"Optimal Weights (LS): Intercept={w_ls[0][0]:.4f}, Slope={w_ls[1][0]:.4f}")

# ================= 2. 非线性模型 (Polynomial & RBF Kernel) =================
# 2.1 多项式回归 (Degree = 6)
poly = PolynomialFeatures(degree=6)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_train_pred_poly = poly_reg.predict(X_train_poly)
y_test_pred_poly = poly_reg.predict(X_test_poly)

# 2.2 RBF 核回归 (Kernel Ridge)
kr_model = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.5)
kr_model.fit(X_train, y_train)
y_train_pred_kr = kr_model.predict(X_train)
y_test_pred_kr = kr_model.predict(X_test)

# ================= 3. 评估与输出 =================
def evaluate(name, y_true_tr, y_pred_tr, y_true_te, y_pred_te):
    mse_tr = mean_squared_error(y_true_tr, y_pred_tr)
    mse_te = mean_squared_error(y_true_te, y_pred_te)
    r2_te = r2_score(y_true_te, y_pred_te)
    print(f"{name:15s} | Train MSE: {mse_tr:.4f} | Test MSE: {mse_te:.4f} | Test R2: {r2_te:.4f}")

print(f"\n--- Model Performance ---")
evaluate("Least Squares", y_train, y_train_pred_ls, y_test, y_test_pred_ls)
evaluate("Grad Descent", y_train, X_train_b.dot(w_gd), y_test, y_test_pred_gd)
evaluate("Newton Method", y_train, X_train_b.dot(w_newton), y_test, y_test_pred_newton)
evaluate("Polynomial(d=6)", y_train, y_train_pred_poly, y_test, y_test_pred_poly)
evaluate("RBF Kernel Reg", y_train, y_train_pred_kr, y_test, y_test_pred_kr)

# ================= 4. 可视化绘制 =================
X_range = np.linspace(X_train.min()-0.5, X_train.max()+0.5, 500).reshape(-1, 1)
y_range_poly = poly_reg.predict(poly.transform(X_range))
y_range_kr = kr_model.predict(X_range)
y_range_ls = np.c_[np.ones((len(X_range), 1)), X_range].dot(w_ls)

plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='lightgray', label='Training Data', alpha=0.7)
plt.scatter(X_test, y_test, color='darkgray', marker='x', label='Testing Data', alpha=0.7)

plt.plot(X_range, y_range_ls, color='red', linestyle='--', linewidth=2, label='Linear Fit (LS)')
plt.plot(X_range, y_range_poly, color='blue', linewidth=2, label='Polynomial Fit (Deg=6)')
plt.plot(X_range, y_range_kr, color='green', linewidth=2, label='RBF Kernel Fit')

plt.title('Comparison of Linear and Non-Linear Regression Models')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('fit_results.png', dpi=300)
plt.show()