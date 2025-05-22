# 导入所需的库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from keras.layers import *
from keras.models import *
from sklearn import preprocessing
from pandas import DataFrame

'''
设置lstm的时间窗等参数
'''
window = 9  # 时间窗口大小，用于定义输入数据的序列长度
lstm_units = 16  # LSTM层的神经元单元数量
dropout = 0.1  # Dropout的概率
epoch = 1500  # 训练迭代次数

'''
读取数据
'''
df00 = pd.read_csv('宝安区data.csv', encoding='ISO-8859-1')
df1 = df00.iloc[:, 3:12]  # 选择数据中的列范围3~12映射到0~9
'''
数据归一化
'''
# 使用MinMaxScaler进行数据归一化
min_max_scaler0 = preprocessing.MinMaxScaler()
min_max_scaler1 = preprocessing.MinMaxScaler()
df0 = min_max_scaler0.fit_transform(df1.iloc[:, 0:8])  # 对特征数据归一化
df2 = min_max_scaler1.fit_transform(df1.iloc[:, 8:9])  # 对被预测流量归一化
df3 = np.concatenate((df0, df2), axis=1)#axis=1：列。将各列合并
df = pd.DataFrame(df3, columns=df1.columns)#创建新的数据库，列名和df1相同 
input_size = len(df.iloc[1, :])#数据框df中第一行的长度，即特征的个数。

'''
构建网络输入
'''
stock = df
seq_len = window
amount_of_features = len(stock.columns)
print(amount_of_features)
data = stock.values
print(data.shape)
sequence_length = seq_len + 1
result = []

# 创建时间序列数据，每个序列包含 seq_len+1 个数据点
for index in range(len(data) - sequence_length):
    result.append(data[index: index + sequence_length])
result = np.array(result)#转换为numpy数组
row = round(0.9 * result.shape[0])
print(row)
print('~~~~~~~')
train = result[:int(row), :]
print(train.shape)
x_train = train[:, :-1]
y_train = train[:, -1][:, -1]
print(y_train.shape)
x_test = result[int(row):, :-1]
print(x_test.shape)
y_test = result[int(row):, -1][:, -1]

'''
搭建LSTM+Transformer层网络模型
'''
# 定义LSTM网络模型
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, LayerNormalization, MultiHeadAttention, Flatten
window = 9  # 设置窗口大小
input_size = 9 # 输入维度大小
# 输入层
inputs = Input(shape=(window, input_size))
# LSTM层
lstm_output1 = LSTM(8, activation='tanh', return_sequences=True, name='LSTM0')(inputs)
lstm_output2 = LSTM(8, activation='tanh', return_sequences=True, name='LSTM1')(lstm_output1)
lstm_output3 = LSTM(8, activation='tanh', return_sequences=True, name='LSTM2')(lstm_output2)
lstm_output = LSTM(8, activation='tanh', return_sequences=True, name='LSTM3')(lstm_output3)

# Transformer层
num_heads = 8  # 设置Transformer中的头数
transformer_output = MultiHeadAttention(num_heads=num_heads, key_dim=64, value_dim=64)(lstm_output, lstm_output, lstm_output)  # 传递三个相同的参数作为key, query, value
transformer_output = LayerNormalization(epsilon=1e-5)(transformer_output)
# 全连接层
transformer_output = Flatten()(transformer_output)
outputs = Dense(1, activation='sigmoid')(transformer_output)

# 创建模型
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# 编译模型，定义损失函数和优化器
model.compile(loss='mse', optimizer='adam')

# 打印模型结构
model.summary()


# 训练模型
history = model.fit(x_train, y_train, 64, epoch, shuffle=False, validation_data=(x_test, y_test))

# 绘制训练过程中的损失曲线
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(loss, label='Train Loss')
plt.legend(loc='upper right')
plt.title('Train and Val Loss')
plt.show()

# 在训练集上的拟合结果
y_train_predict = model.predict(x_train)
y_train_predict = y_train_predict[:, 0]
draw = pd.concat([pd.DataFrame(y_train), pd.DataFrame(y_train_predict)], axis=1)
draw = min_max_scaler1.inverse_transform(draw)
draw = DataFrame(draw)
draw.iloc[0:, 0].plot(figsize=(12, 6))
draw.iloc[0:, 1].plot(figsize=(12, 6))
plt.legend(('real', 'predict'), fontsize='15')
plt.title("Train Data", fontsize='30')
plt.show()

# 在测试集上的预测
y_test_predict = model.predict(x_test)
y_test_predict = y_test_predict[:, 0]
draw = pd.concat([pd.DataFrame(y_test), pd.DataFrame(y_test_predict)], axis=1)
draw = min_max_scaler1.inverse_transform(draw)
draw = DataFrame(draw)

# 将测试集的结果保存为 'LSTM.csv' 文件
draw.to_csv('LSTM_Transformer.csv')

# 绘制测试集的真实数据和预测结果
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(np.array(draw.iloc[-287:, 0]), label='Data', linestyle='--')
ax.plot(np.array(draw.iloc[-287:, 1]), label='Prediction', linestyle='-')

# 对横坐标轴进行标注

ax.legend()
ax.set_xlabel('Date-time')
ax.set_ylabel('Counts')
plt.title("Test Data", fontsize='30')
plt.show()

# 输出模型性能指标
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) 

print('训练集上的MAE/MSE/MAPE')
print(sqrt(mean_absolute_error(y_train_predict, y_train) ))  # 平均绝对误差
print(mean_squared_error(y_train_predict, y_train) )  # 均方误差
print(mape(y_train_predict, y_train))  

print('测试集上的MAE/MSE/MAPE')
print(sqrt(mean_absolute_error(y_test_predict, y_test) ))
print(mean_squared_error(y_test_predict, y_test))
print(mape(y_test_predict, y_test))
