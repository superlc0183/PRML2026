import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# === 1. 将时间序列转换为监督学习问题的函数 ===
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
    # 输入序列 (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f'var{j+1}(t-{i})' for j in range(n_vars)]
        
    # 预测序列 (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [f'var{j+1}(t)' for j in range(n_vars)]
        else:
            names += [f'var{j+1}(t+{i})' for j in range(n_vars)]
            
    # 拼接起来
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# === 2. 加载与清洗数据 ===
print("加载数据...")
dataset = pd.read_csv('LSTM-Multivariate_pollution.csv', header=0, index_col=0)
values = dataset.values

# 类别编码 (把风向转为数字)
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
# 确保所有数据都是浮点数
values = values.astype('float32')

# 归一化 (压缩到 0 到 1 之间)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# === 3. 重构为监督学习格式 ===
# 使用前1个小时的数据 (t-1) 来预测当前小时 (t)
reframed = series_to_supervised(scaled, 1, 1)

# 注意：我们只想预测污染值 (var1(t))，所以我们要把其它变量在 t 时刻的列删掉
# 我们的原始列有 8 个 (pollution, dew, temp, press, wnd_dir, wnd_spd, snow, rain)
# 所以 var1(t) 是污染，后面的 var2(t) 到 var8(t) 我们不预测，直接丢弃
reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)

print("\n=== 重构后的数据前 5 行 ===")
print(reframed.head())
print(f"\n最终准备输入神经网络的数据维度: {reframed.shape}")