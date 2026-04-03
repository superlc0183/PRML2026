import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# 1. 修正后的数据生成函数
# ==========================================
def make_moons_3d(n_samples=500, noise=0.1):
    n_samples_per_class = int(n_samples / 2)
    t = np.linspace(0, 2 * np.pi, n_samples_per_class)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t) 
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    labels = np.hstack([np.zeros(n_samples_per_class), np.ones(n_samples_per_class)])
    X += np.random.normal(scale=noise, size=X.shape)
    return X, labels

# 生成数据
X_train, y_train = make_moons_3d(n_samples=1000, noise=0.2)
X_test, y_test = make_moons_3d(n_samples=500, noise=0.2)

# ==========================================
# 2. 初始化分类器
# ==========================================
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "AdaBoost + DT": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=42), 
        n_estimators=50, 
        random_state=42
    ),
    "SVM (Linear)": SVC(kernel='linear', random_state=42),
    "SVM (Poly, d=3)": SVC(kernel='poly', degree=3, random_state=42),
    "SVM (RBF)": SVC(kernel='rbf', random_state=42)
}

# 存储预测结果供画图使用
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

# ==========================================
# 3. 绘制并保存图片：混淆矩阵对比图
# ==========================================
fig, axes = plt.subplots(1, 5, figsize=(25, 4))
fig.suptitle('Confusion Matrices of Different Classifiers', fontsize=16)

for ax, (name, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False, 
                xticklabels=['C0', 'C1'], yticklabels=['C0', 'C1'])
    ax.set_title(f"{name}\nAcc: {accuracy_score(y_test, y_pred):.4f}")
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300)
plt.close() # 关闭当前图，准备画下一张

# ==========================================
# 4. 绘制并保存图片：错误点可视化 (选 Linear 和 RBF 对比)
# ==========================================
fig = plt.figure(figsize=(14, 6))
fig.suptitle('Visualization of Misclassified Points (Test Set)', fontsize=16)

models_to_plot = ["SVM (Linear)", "SVM (RBF)"]

for i, name in enumerate(models_to_plot):
    ax = fig.add_subplot(1, 2, i+1, projection='3d')
    y_pred = predictions[name]
    
    # 找出分类正确的点和分类错误的点
    correct_idx = (y_test == y_pred)
    wrong_idx = (y_test != y_pred)
    
    # 画出分类正确的点 (半透明)
    ax.scatter(X_test[correct_idx, 0], X_test[correct_idx, 1], X_test[correct_idx, 2], 
               c='gray', alpha=0.3, label='Correctly Classified')
    
    # 画出分类错误的点 (红色高亮)
    ax.scatter(X_test[wrong_idx, 0], X_test[wrong_idx, 1], X_test[wrong_idx, 2], 
               c='red', s=50, alpha=0.8, label='Misclassified')
    
    ax.set_title(f"{name} (Errors: {sum(wrong_idx)} / {len(y_test)})")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

plt.tight_layout()
plt.savefig('error_visualization.png', dpi=300)
plt.close()

print("图片生成完毕：已在当前文件夹保存 'confusion_matrices.png' 和 'error_visualization.png'")