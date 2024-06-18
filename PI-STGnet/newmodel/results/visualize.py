# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
# plt.rcParams['figure.dpi'] = 600  # plt.show显示分辨率
import numpy as np
from inits import *
from matplotlib.ticker import MaxNLocator
import seaborn as sns

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 17.,
}

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}

# mean=  29.996377569846402
# std=  31.95731962018177

ASTGCN_NINGDE = np.load('ASTGCN-NINGDE.npz', allow_pickle=True)['prediction'][20:]
LABEL_ASTGCN_NINGDE = np.load('ASTGCN-NINGDE.npz', allow_pickle=True)['truth'][20:]
GMAN_NINGDE = np.load('GMAN-NINGDE.npz', allow_pickle=True)['prediction']
GraphWaveNet_NINGDE = np.load('GraphWaveNet-NINGDE.npz')['prediction'][20:]
LSTM_BILSTM_NINDE = np.load('LSTM-BILSTM-NINDE.npz')['prediction']
STGNN_NINGDE = np.load('STGNN-NINGDE.npz')['prediction'][20:].transpose(0,2,1)
DELA_NINGDE = np.load("DELA-NINGDE.npz")['prediction']
PI_STGnet_NINGDE = np.load('PI-STGnet-NINGDE.npz')['prediction'].transpose(0,2,1,3)
ST_GRAT_NINGDE = np.load('ST-GRAT-NINGDE.npz')['prediction'][20:].transpose(0,2,1)
STGIN_NINGDE = np.load('STGIN-NINGDE.npz', allow_pickle=True)['prediction']
TGCN_NINGDE = np.load('TGCN-NINGDE.npz', allow_pickle=True)['prediction']


print(ASTGCN_NINGDE.shape, 'ASTGCN_NINGDE')
print(LABEL_ASTGCN_NINGDE.shape, 'LABEL')
print(GMAN_NINGDE.shape, 'GMAN_NINGDE')
print(GraphWaveNet_NINGDE.shape, 'GraphWaveNet_NINGDE')
print(LSTM_BILSTM_NINDE.shape, 'LSTM_BILSTM_NINDE')
print(STGNN_NINGDE.shape,'STGNN_NINGDE')
print(DELA_NINGDE.shape,'DELA_NINGDE')
print(PI_STGnet_NINGDE.shape, 'PI_STGnet_NINGDE')
print(ST_GRAT_NINGDE.shape, 'ST_GRAT_NINGDE')
print(STGIN_NINGDE.shape, 'STGIN_NINGDE')
print(TGCN_NINGDE.shape, 'TGCN_NINGDE')


#---------------------------------------------  画不同站点的预测-实际曲线图 -------------------------------
fig1, ax1 = plt.subplots(1, 1)
road_index = 3 # --------------------站点选择3/8/12/15----------------------------
total=1200 # 预测的结束步长

# plt.rcParams['figure.dpi'] = 700

ax1.plot(np.concatenate([list(LABEL_ASTGCN_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)], axis=-1), color='#FC8002', linestyle='-',
         linewidth=0.7, label='Observed')
ax1.plot(np.concatenate([list(ASTGCN_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#ADDB88', linestyle='-',
         linewidth=0.5, label='ASTGCN')
ax1.plot(np.concatenate([list(GMAN_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1),  color='#369F2D', linestyle='-', linewidth=0.5,
         label='GMAN')
ax1.plot(np.concatenate([list(GraphWaveNet_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#EE4431', linestyle='-',
         linewidth=0.5, label='GraphWaveNet')
ax1.plot(np.concatenate([list(LSTM_BILSTM_NINDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#B9181A', linestyle='-',
         linewidth=0.5, label='LSTM-BILSTM')
ax1.plot(np.concatenate([list(STGNN_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#FABB6E', linestyle='-', linewidth=0.5,
         label='STGNN')
ax1.plot(np.concatenate([list(DELA_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#FAC7B3', linestyle='-', linewidth=0.5,
         label='DELA')
ax1.plot(np.concatenate([list(ST_GRAT_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#92C2DD', linestyle='-', linewidth=0.5,
         label='ST-GRAT')
ax1.plot(np.concatenate([list(STGIN_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#4995C6', linestyle='-', linewidth=0.5,
         label='STGIN')
ax1.plot(np.concatenate([list(TGCN_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#1663A9', linestyle='-', linewidth=0.5,
         label='T-GCN')
ax1.plot(np.concatenate([list(PI_STGnet_NINGDE[sample_index, road_index, :,1]) for sample_index in range(12, total, 12)],axis=-1), color='#614099', linestyle='-', linewidth=0.7,
         label='PI-STGnet')

# ---------------------------- 局部放大（不要了）--------------------------------

# 绘制缩放图
# axins = ax1.inset_axes((0.4, 0.6, 0.2, 0.2))  # 第一个表示放大图左右的位置在原图的多少百分比，第二个表示放大图上下的位置所占原图的百分比，第三/四个放大图长/宽
#
# # 在缩放图中也绘制主图所有内容，然后根据限制横纵坐标来达成局部显示的目的
#
# axins.plot(np.concatenate([list(ASTGCN_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#ff5b00', linestyle='-',
#          linewidth=0.7, label='ASTGCN')
# axins.plot(np.concatenate([list(LABEL_ASTGCN_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)], axis=-1), color='black', linestyle='-',
#          linewidth=0.5, label='Observed')
# axins.plot(np.concatenate([list(GMAN_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1),  color='red', linestyle='-', linewidth=0.5,
#          label='GMAN')
# axins.plot(np.concatenate([list(GraphWaveNet_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#d0c101', linestyle='-',
#          linewidth=0.5, label='GraphWaveNet')
# axins.plot(np.concatenate([list(LSTM_BILSTM_NINDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#0cdc73', linestyle='-',
#          linewidth=0.5, label='LSTM-BILSTM')
# axins.plot(np.concatenate([list(MTGNN_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#f504c9', linestyle='-', linewidth=0.5,
#          label='MTGNN')
# axins.plot(np.concatenate([list(PI_STGnet_NINGDE[sample_index, road_index, :,1]) for sample_index in range(12, total, 12)],axis=-1), color='blue', linestyle='-', linewidth=0.5,
#          label='PI_STGnet')
# axins.plot(np.concatenate([list(ST_GRAT_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='orange', linestyle='-', linewidth=0.5,
#          label='ST-GRAT')
# axins.plot(np.concatenate([list(STGIN_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#a55af4', linestyle='-', linewidth=0.5,
#          label='STGIN')
# axins.plot(np.concatenate([list(TGCN_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='skyblue', linestyle='-', linewidth=0.5,
#          label='TGCN')
#
#
# # 设置放大区间
# zone_left = 300
# zone_right = 450
# x_axis_data = list(range(500))
# # 坐标轴的扩展比例（根据实际数据调整）
# x_ratio = 0  # x轴显示范围的扩展比例
# y_ratio = 0  # y轴显示范围的扩展比例
#
# # X轴的显示范围
# xlim0 = x_axis_data[zone_left]-(x_axis_data[zone_right]-x_axis_data[zone_left])*x_ratio
# xlim1 = x_axis_data[zone_right]+(x_axis_data[zone_right]-x_axis_data[zone_left])*x_ratio
#
# # Y轴的显示范围
# y1=np.hstack(np.concatenate([list(ASTGCN_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1)).flatten()
# y2=np.hstack(np.concatenate([list(LABEL_ASTGCN_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)], axis=-1)).flatten()
# y3=np.hstack(np.concatenate([list(GMAN_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1)).flatten()
# y4=np.hstack(np.concatenate([list(GraphWaveNet_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1)).flatten()
# y5=np.hstack(np.concatenate([list(LSTM_BILSTM_NINDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1)).flatten()
# y6=np.hstack(np.concatenate([list(MTGNN_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1)).flatten()
# y7=np.hstack(np.concatenate([list(PI_STGnet_NINGDE[sample_index, road_index, :,1]) for sample_index in range(12, total, 12)],axis=-1)).flatten()
# y8=np.hstack(np.concatenate([list(ST_GRAT_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1)).flatten()
# y9=np.hstack(np.concatenate([list(STGIN_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1)).flatten()
# y10=np.hstack(np.concatenate([list(TGCN_NINGDE[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1)).flatten()
#
# y = np.hstack((y1[zone_left:zone_right], y2[zone_left:zone_right],
#                y3[zone_left:zone_right],y4[zone_left:zone_right],
#                y5[zone_left:zone_right],y6[zone_left:zone_right],
#               y7[zone_left:zone_right],y8[zone_left:zone_right],y9[zone_left:zone_right],y10[zone_left:zone_right]))
# ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
# ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio
#
# # 调整子坐标系的显示范围
# axins.set_xlim(xlim0, xlim1)
# axins.set_ylim(ylim0, ylim1)

# ------------------------

plt.legend(loc='upper left', prop=font1)
# plt.grid(axis='y')
plt.ylabel('Traffic flow', font2)
# plt.xlabel('Target time steps', font2)
# plt.title('Exit tall dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()

#------------------------------------------------------ 第一幅图绘制完毕 --------------------------------------------------


#------------------------------------------------------ 绘制气泡图 -------------------------------------------------------

begin = 0 # 开始步长
total=1000 # 结束步长

LABEL_obs=np.concatenate([list(LABEL_ASTGCN_NINGDE[sample_index]) for sample_index in range(begin, total, 12)],axis=-1)
STGNN_pre=np.concatenate([list(STGNN_NINGDE[sample_index]) for sample_index in range(begin, total, 12)],axis=-1)
TGCN_pre=np.concatenate([list(TGCN_NINGDE[sample_index]) for sample_index in range(begin, total, 12)],axis=-1)
ST_GRAT_pre=np.concatenate([list(ST_GRAT_NINGDE[sample_index]) for sample_index in range(begin, total, 12)],axis=-1)
GraphWaveNet_pre=np.concatenate([list(GraphWaveNet_NINGDE[sample_index]) for sample_index in range(begin, total, 12)],axis=-1)
GMAN_pre=np.concatenate([list(GMAN_NINGDE[sample_index]) for sample_index in range(begin, total, 12)],axis=-1)
PI_STGnet_pre=np.concatenate([list(PI_STGnet_NINGDE[sample_index, :, :,1]) for sample_index in range(begin, total, 12)],axis=-1)

#-------------------------------------------------------- 子图，内容为黑色的散点图 ------------------------------------------
plt.subplot(2,3,1)
plt.scatter(LABEL_obs,STGNN_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'STGNN',linewidths=1)
a=[i for i in range(300)]
b=[i for i in range(300)]
plt.plot(a,b,'black',linewidth=2)
plt.ylabel("Predicted traffic flow", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(2,3,2)
plt.scatter(LABEL_obs,TGCN_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'TGCN',linewidths=1)
a=[i for i in range(300)]
b=[i for i in range(300)]
plt.plot(a,b,'black',linewidth=2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(2,3,3)
plt.scatter(LABEL_obs,ST_GRAT_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'ST-GRAT',linewidths=1)
a=[i for i in range(300)]
b=[i for i in range(300)]
plt.plot(a,b,'black',linewidth=2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(2,3,4)
plt.scatter(LABEL_obs,GMAN_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'GMAN',linewidths=1)
a=[i for i in range(300)]
b=[i for i in range(300)]
plt.plot(a,b,'black',linewidth=2)
plt.xlabel("Observed traffic flow", font2)
plt.ylabel("Predicted traffic flow", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(2,3,5)
plt.scatter(LABEL_obs,GraphWaveNet_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'GraphWaveNet',linewidths=1)
plt.plot(a,b,'black',linewidth=2)
plt.xlabel("Observed traffic flow", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(2,3,6)
plt.scatter(LABEL_obs,PI_STGnet_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'PI-STGnet',linewidths=1)
a=[i for i in range(300)]
b=[i for i in range(300)]
plt.plot(a,b,'black',linewidth=2)
plt.xlabel("Observed traffic flow", font2)
plt.legend(loc='upper left',prop=font2)
plt.show()
#--------------------------------------------------- 子图绘制结束 --------------------------------------------------------

#--------------------------------------------------- 绘制单个预测-真实散点图，带上分布情况 ------------------------------------
sns.set_theme(style='ticks', font_scale=2.,font='Times New Roman')
data_df = pd.DataFrame(np.concatenate([np.reshape(LABEL_obs,[-1,1]),np.reshape(STGNN_pre, [-1,1])],axis=-1), columns=['truth','prediction'])
g = sns.jointplot(x="truth", y="prediction", data=data_df, label=u'DCRNN',color="g",alpha=0.5,marginal_kws=dict(bins=15, #hist箱子个数
                                    kde=True,#开启核密度图
                                    color='#c72e29',#直方图hist填充色
                                   ))
g.set_axis_labels(xlabel='Observed traffic flow', ylabel='Predicted traffic flow')
a=[i for i in range(300)]
b=[i for i in range(300)]
plt.plot(a,b,'black',linewidth=2)
plt.legend(loc='upper left',prop=font2)
plt.show()

data_df = pd.DataFrame(np.concatenate([np.reshape(LABEL_obs,[-1,1]),np.reshape(TGCN_pre, [-1,1])],axis=-1), columns=['truth','prediction'])
g = sns.jointplot(x="truth", y="prediction", data=data_df, label=u'TGCN',color="g",alpha=0.5,marginal_kws=dict(bins=15, #hist箱子个数
                                    kde=True,#开启核密度图
                                    color='#c72e29',#直方图hist填充色
                                   ))
g.set_axis_labels(xlabel='Observed traffic flow', ylabel='Predicted traffic flow')
a=[i for i in range(300)]
b=[i for i in range(300)]
plt.plot(a,b,'black',linewidth=2)
plt.legend(loc='upper left',prop=font2)
plt.show()

data_df = pd.DataFrame(np.concatenate([np.reshape(LABEL_obs,[-1,1]),np.reshape(ST_GRAT_pre, [-1,1])],axis=-1), columns=['truth','prediction'])
g = sns.jointplot(x="truth", y="prediction", data=data_df, label=u'ST-GRAT',color="g",alpha=0.5,marginal_kws=dict(bins=15, #hist箱子个数
                                    kde=True,#开启核密度图
                                    color='#c72e29',#直方图hist填充色
                                   ))
g.set_axis_labels(xlabel='Observed traffic flow', ylabel='Predicted traffic flow')
a=[i for i in range(300)]
b=[i for i in range(300)]
plt.plot(a,b,'black',linewidth=2)
plt.legend(loc='upper left',prop=font2)
plt.show()

data_df = pd.DataFrame(np.concatenate([np.reshape(LABEL_obs,[-1,1]),np.reshape(GMAN_pre, [-1,1])],axis=-1), columns=['truth','prediction'])
g = sns.jointplot(x="truth", y="prediction", data=data_df, label=u'GMAN',color="g",alpha=0.5,marginal_kws=dict(bins=15, #hist箱子个数
                                    kde=True,#开启核密度图
                                    color='#c72e29',#直方图hist填充色
                                   ))
g.set_axis_labels(xlabel='Observed traffic flow', ylabel='Predicted traffic flow')
a=[i for i in range(300)]
b=[i for i in range(300)]
plt.plot(a,b,'black',linewidth=2)
plt.legend(loc='upper left',prop=font2)
plt.show()

data_df = pd.DataFrame(np.concatenate([np.reshape(LABEL_obs,[-1,1]),np.reshape(GraphWaveNet_pre, [-1,1])],axis=-1), columns=['truth','prediction'])
g = sns.jointplot(x="truth", y="prediction", data=data_df, label=u'GraphWaveNet',color="g",alpha=0.5,marginal_kws=dict(bins=15, #hist箱子个数
                                    kde=True,#开启核密度图
                                    color='#c72e29',#直方图hist填充色
                                   ))
g.set_axis_labels(xlabel='Observed traffic flow', ylabel='Predicted traffic flow')
a=[i for i in range(300)]
b=[i for i in range(300)]
plt.plot(a,b,'black',linewidth=2)
plt.legend(loc='upper left',prop=font2)
plt.show()

data_df = pd.DataFrame(np.concatenate([np.reshape(LABEL_obs,[-1,1]),np.reshape(PI_STGnet_pre, [-1,1])],axis=-1), columns=['truth','prediction'])
g = sns.jointplot(x="truth", y="prediction", data=data_df, label=u'PI-STGnet',color="g",alpha=0.5,marginal_kws=dict(bins=15, #hist箱子个数
                                    kde=True,#开启核密度图
                                    color='#c72e29',#直方图hist填充色
                                   ))
g.set_axis_labels(xlabel='Observed traffic flow', ylabel='Predicted traffic flow')
a=[i for i in range(300)]
b=[i for i in range(300)]
plt.plot(a,b,'black',linewidth=2)
plt.legend(loc='upper left',prop=font2)
plt.show()
#--------------------------------------------------------- 绘制结束 -----------------------------------------------------


#--------------------------------------------------- 绘制热力图 ----------------------------------------------------------
LABEL_ASTGCN_NINGDE = np.load('ASTGCN-NINGDE.npz', allow_pickle=True)['truth'][20:] #真实值
LABEL_ASTGCN_NINGDE_flow = np.concatenate([list(LABEL_ASTGCN_NINGDE[sample_index]) for sample_index in range(665, 953, 12)], axis=-1) #4月5号->（89，377，12）
                                                                                                                                    #4月7号->（665，953，12）
                                                                                                                                    #4月9号->（1241，1529，12）
PI_STGnet_prediction = np.load('PI-STGnet-NINGDE.npz')['prediction'].transpose(0,2,1,3) #预测值
PI_STGnet_prediction_flow = np.concatenate([list(PI_STGnet_NINGDE[sample_index, :, :,1]) for sample_index in range(665, 953, 12)],axis=-1) #4月5号->（89，377，12）
                                                                                                                                    #4月7号->（665，953，12）
                                                                                                                                    #4月9号->（1241，1529，12）
print(LABEL_ASTGCN_NINGDE_flow.shape, 'LABEL_ASTGCN_NINGDE_flow')
print(PI_STGnet_prediction_flow.shape, 'PI_STGnet_prediction_flow')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
fig, ax = plt.subplots(figsize=(8,6))
ax.tick_params(axis='both', which='major', labelsize=20)
h = ax.imshow(LABEL_ASTGCN_NINGDE_flow, interpolation='nearest', cmap='rainbow', #想绘制预测值的热力图，把“LABEL_ASTGCN_NINGDE_flow” -> "PI_STGnet_prediction_flow"
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cax.tick_params(labelsize=20)
fig.colorbar(h, cax=cax)
ax.set_xlabel('Time (h)', fontsize = 20)
ax.set_ylabel('Station', fontsize = 20)
# ax.set_yticks([0,5,10,15,20,25,30])
# ax.set_xticks([0,3,6,9,12,15,18,21,24])
ax.set_xticklabels(['0','0','4','8','12','16','20','24'])
ax.set_yticklabels(['0','0','5','10','15','20','25','30'])
plt.show()
#------------------------------------------ 绘制完毕 ------------------------------------------------------------------


#-------------------------------------------------- 绘制基本图 ---------------------------------------------------------
PI_STGnet_fun = np.load('PI-STGnet-NINGDE.npz', allow_pickle=True)['truth'].transpose(0,2,1,3)  # 真实值
PI_STGnet_truth_flow_fun = np.concatenate([list(PI_STGnet_fun[sample_index, 12, :,1]) for sample_index in range(665, 953, 12)],axis=-1)
PI_STGnet_truth_flow_fun =  PI_STGnet_truth_flow_fun *  12
PI_STGnet_truth_velocity_fun = np.concatenate([list(PI_STGnet_fun[sample_index, 12, :,0]) for sample_index in range(665, 953, 12)],axis=-1)

PI_STGnet_pre_fun = np.load('PI-STGnet-NINGDE.npz')['prediction'].transpose(0,2,1,3) #预测值
PI_STGnet_pre_flow_fun = np.concatenate([list(PI_STGnet_pre_fun[sample_index, 12, :,1]) for sample_index in range(665, 953, 12)],axis=-1)
PI_STGnet_pre_flow_fun =  PI_STGnet_pre_flow_fun *  12
PI_STGnet_pre_velocity_fun = np.concatenate([list(PI_STGnet_pre_fun[sample_index, 12, :,0]) for sample_index in range(665, 953, 12)],axis=-1)

plt.scatter(PI_STGnet_truth_velocity_fun,PI_STGnet_truth_flow_fun,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'Obversed',linewidths=1)
plt.scatter(PI_STGnet_pre_velocity_fun,PI_STGnet_pre_flow_fun,alpha=0.7,color='blue',edgecolor = "black",marker='o',label=u'PI-STGnet',linewidths=1)
plt.xlabel('Velocity')
plt.ylabel("Flow")
plt.legend()
plt.show()
# from scipy import optimize as op
# # 需要拟合的函数
# def f_1(x, A, B, C):
#     return A * x**2 + B * x + C
# # 得到返回的A，B值
# A, B, C = op.curve_fit(f_1, PI_STGnet_truth_velocity_fun,PI_STGnet_truth_flow_fun )[0]
# print(A,B,C,'A,B,C')
# x = np.arange(0, 150, 0.01)
# y = A * x**2 + B *x + C
# plt.plot(x, y,color='red',label='fit curve')
# plt.legend() # 显示label

# plt.show()