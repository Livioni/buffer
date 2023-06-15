import pandas as pd
import numpy as np
import sys
sys.path.append('/Users/livion/Documents/GitHub/Sources/buffer') 
from baselines.cost_function import Ali_function_cost_usd

# Read the Excel file
scene_name = 'scene_04'
ktc = pd.read_excel('data/ktc/' + scene_name + '/' + scene_name + '.xlsx')
no_batch = pd.read_excel('data/no_batch/' + scene_name + '/' + scene_name + '.xlsx')
partition = pd.read_excel('data/vanilla-partitions/' + scene_name + '/' + scene_name + '.xlsx')
vanilla = pd.read_excel('data/vanilla/' + scene_name + '/' + scene_name + '.xlsx')

frame_cnt = 48 #134
patches_cnt = 611 #1216
column_name = 'Average Latency'

# Read a specific column from the dataframe
avg_cost_ktc = ktc[column_name].values
avg_cost_no_batch = no_batch[column_name].values
avg_cost_partition = partition[column_name].values
avg_cost_vanilla = vanilla[column_name].values

cost_ktc = avg_cost_ktc * frame_cnt
cost_no_batch = avg_cost_no_batch * patches_cnt
cost_partition = avg_cost_partition * frame_cnt
cost_vanilla = avg_cost_vanilla * frame_cnt

for cost in [cost_ktc,cost_no_batch,cost_partition,cost_vanilla]:
    for index,time in enumerate(cost):
        money = Ali_function_cost_usd(time/1000, Mem=4, CPU=2, GPU=6) 
        cost[index] = money

#normalize the data

import matplotlib.pyplot as plt

# Random test data
plt.style.use("seaborn-v0_8-darkgrid")
all_data = [np.mean(cost_ktc),np.mean(cost_partition),np.mean(cost_vanilla),np.mean(cost_no_batch)]
error_bar = [np.std(cost_ktc),np.std(cost_partition),np.std(cost_vanilla),np.std(cost_no_batch)]
labels = ['Koutuchuan', 'Patches Frame', 'Full Frame','No Batch']

fig, ax1 = plt.subplots(figsize=(5, 5))

colors = [(253/255,185/255,107/255), (254/255, 162/255, 158/255), (114/255,170/255,207/255) ,'#2AB34A']

# rectangular box plot
bplot1 = ax1.bar(labels,all_data,yerr=error_bar, color = colors)
                 

# adding horizontal grid lines
ax1.yaxis.grid(True)
ax1.set_xlabel('Scene 01')
ax1.set_ylabel('Normalized (USD)')

plt.show()
