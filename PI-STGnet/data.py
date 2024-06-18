import numpy as np
import pandas as pd
time_df = pd.read_csv('D:/博士阶段/师弟paper/尤哥电力项目/ST-EVCDP-main/ST-EVCDP-main/datasets/time.csv', index_col=None, header=0)
traffic_df  = pd.read_csv('D:/博士阶段/师弟paper/尤哥电力项目/ST-EVCDP-main/ST-EVCDP-main/datasets/duration.csv',index_col=None, header=0)
repeated_df = time_df.reindex(time_df.index.repeat(247))
df1 = traffic_df
df = df1.iloc[:,1:]
volume = []
for station, traffic_value in df.iterrows():
    volume.append(traffic_value.values)
my_list = [item for sublist in volume for item in sublist]

# 生成一个新列表，其中每个元素是由三个列表对应位置的数字构成的字符串
data_1 = np.array(2022).repeat(2134080)
data_2 = repeated_df['month']
data_3 = repeated_df['day']
result = ['{}/{}/{}'.format(d1, d2, d3) for d1, d2, d3 in zip(data_1, data_2, data_3)]

train = pd.DataFrame({'node':np.tile(np.arange(0,247,1), 8640),'date':result,'day':repeated_df['day'].values,'hour':repeated_df['hour'].values,'minute':repeated_df['minute'].values,'volume':my_list})

train.to_csv('D:/博士阶段/师弟paper/尤哥电力项目/ST-EVCDP-main/ST-EVCDP-main/datasets/train_duration.csv',index=False)