# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pylab import mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False

x = pd.read_csv('LSTM_Transformer.csv')
x0 = pd.read_csv('LSTM_Transformer.csv')
x1 = pd.read_csv('CNN_GRU.csv')
x2 = pd.read_csv('GRU.csv')
x3 = pd.read_csv('测试集真实值与预测值.csv')

df00 = pd.read_csv('宝安区data.csv', encoding='ISO-8859-1')
row1 = round(0.97 * df00.shape[0])
fig, ax = plt.subplots(figsize=(10, 7))

# 绘制主图
ax.plot(np.array(x.iloc[-287:, 1:2]), label='真实值', linestyle='--', color='blue')
ax.plot(np.array(x0.iloc[-287:, 2:3]), label='LSTM_Transformer', linestyle='-', color='red')
ax.plot(np.array(x1.iloc[-287:, 2:3]), label='CNN_GRU', linestyle=':', color='black')
ax.plot(np.array(x2.iloc[-287:, 2:3]), label='GRU', linestyle='-.', color='green')
ax.plot(np.array(x3.iloc[-287:, 2:3]), label='LSTM', linestyle='-', color='yellow')
# ax.set_ylim(0, 600)
ax.legend(loc='upper left')
ax.set_xlabel('时间')
ax.set_ylabel('数量')

# 对横坐标轴进行标注
plt.title('预测值与真实值')

 # 创建放大区域
# axins = inset_axes(ax, width="30%", height="30%", loc="upper right")
# axins.plot(np.array(x.iloc[-287:, 1:2]), linestyle='--', color='blue')
# axins.plot(np.array(x0.iloc[-287:, 2:3]), linestyle='-', color='red')
# axins.plot(np.array(x1.iloc[-287:, 2:3]), linestyle=':', color='black')
# axins.plot(np.array(x2.iloc[-287:, 2:3]), linestyle='-.', color='green')
# axins.plot(np.array(x3.iloc[-287:, 2:3]), linestyle='-', color='yellow')

# # 设置放大区域的坐标范围
# x1, x2, y1, y2 = 100,150,150,300
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)

# # 在主图中标记放大区域
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.savefig("结果对比图.png", bbox_inches="tight", dpi=600)
plt.show()
