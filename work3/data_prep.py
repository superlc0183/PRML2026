import pandas as pd
import matplotlib.pyplot as plt

# 1. 加载数据
file_path = 'LSTM-Multivariate_pollution.csv'
# 将第一列（通常是时间）解析为日期时间格式，并设置为索引
df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')

# 2. 基本信息检查
print("=== 数据集基本信息 ===")
print(df.info())
print("\n=== 前 5 行数据 ===")
print(df.head())

# 3. 检查缺失值
print("\n=== 缺失值统计 ===")
print(df.isnull().sum())

# 4. 简单可视化：看看这五年的 PM2.5 整体走势
plt.figure(figsize=(15, 5))
plt.plot(df.index, df['pollution'], linewidth=0.5, color='red')
plt.title('PM2.5 Concentration Over 5 Years', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('PM2.5', fontsize=12)
plt.tight_layout()
plt.savefig('pm25_trend.png', dpi=300)
print("\n图片已保存：pm25_trend.png")