# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 20:20:58 2024

@author: pony
"""

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
epoch = 1500 # 训练迭代次数

'''
读取数据
'''
df00 = pd.read_csv('宝安区data.csv', encoding='ISO-8859-1')
df1 = df00.iloc[:, 3:12]  # 选择数据中的列范围


'''
数据归一化
'''
# 使用MinMaxScaler进行数据归一化
min_max_scaler0 = preprocessing.MinMaxScaler()
min_max_scaler1 = preprocessing.MinMaxScaler()
df0 = min_max_scaler0.fit_transform(df1.iloc[:, 0:8])  # 对特征数据归一化
df2 = min_max_scaler1.fit_transform(df1.iloc[:, 8:9])  # 对被预测流量归一化
df3 = np.concatenate((df0, df2), axis=1)
df = pd.DataFrame(df3, columns=df1.columns)
input_size = len(df.iloc[1, :])

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
result = np.array(result)
print(result.shape)
row = round(0.9 * result.shape[0])
print(row)
train = result[:int(row), :]
print(train.shape)
x_train = train[:, :-1]
y_train = train[:, -1][:, -1]
print(y_train.shape)
x_test = result[int(row):, :-1]
print(x_test.shape)
y_test = result[int(row):, -1][:, -1]

'''
搭建LSTM网络模型
'''
# 定义LSTM网络模型
from keras.layers import LSTM

inputs = Input(shape=(window, input_size))


# 第二层LSTM
lstm_2 = LSTM(16, activation='relu', return_sequences=True, name='LSTM2')(inputs)

# 第三层LSTM
lstm_3 = LSTM(8, activation='relu', name='LSTM3')(lstm_2)

# 输出层
outputs = Dense(1, activation='sigmoid')(lstm_3)

# 构建模型
model = Model(inputs=inputs, outputs=outputs)


# 编译模型，定义损失函数和优化器
model.compile(loss='mse', optimizer='Adam')

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

# 将测试集的结果保存为 '测试集.csv' 文件
draw.to_csv('测试集真实值与预测值.csv')

# 绘制测试集的真实数据和预测结果
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(np.array(draw.iloc[:, 0]), label='Data', linestyle='--')
ax.plot(np.array(draw.iloc[:, 1]), label='Prediction', linestyle='-')

# 对横坐标轴进行标注
row1 = round(0.97 * result.shape[0])
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
print(mean_squared_error(y_test_predict, y_test) )
print(mape(y_test_predict, y_test))
