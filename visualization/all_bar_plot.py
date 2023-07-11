# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys,matplotlib
sys.path.append('/Users/livion/Documents/GitHub/Sources/buffer') 
from baselines.cost_function import Ali_function_cost_usd
plt.style.use("seaborn-v0_8-darkgrid")

matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
matplotlib.rcParams['font.family'] = "sans-serif"

# Read the Excel file
all_scene_name = ['scene_01','scene_02','scene_03','scene_04','scene_05','scene_06',\
                  'scene_07','scene_08','scene_09','scene_10'] #,
scene_label = ['scene_01 (#134)','scene_02 (#134)','scene_03 (#134)','scene_04 (#48)','scene_05 (#33)','scene_06 (#122)',\
                  'scene_07 (#80)','scene_08 (#134)','scene_09 (#134)','scene_10 (#134)'] #,
scene_number = len(all_scene_name)
ktc_10 = np.zeros(scene_number)
no_batch_10 = np.zeros(scene_number)
partition_10 = np.zeros(scene_number)
vanilla_10 = np.zeros(scene_number)

ktc_std = np.zeros(scene_number)
no_batch_std = np.zeros(scene_number)
partition_std = np.zeros(scene_number)
vanilla_std = np.zeros(scene_number)

for index,scene_name in enumerate(all_scene_name):
    ktc = pd.read_excel('data/ktc_copy/' + scene_name + '/' + scene_name + '.xlsx')
    no_batch = pd.read_excel('data/no_batch/' + scene_name + '/' + scene_name + '.xlsx')
    partition = pd.read_excel('data/vanilla-partitions/' + scene_name + '/' + scene_name + '.xlsx')
    vanilla = pd.read_excel('data/vanilla/' + scene_name + '/' + scene_name + '.xlsx')

    frame_cnt = [134,134,134,48,33,122,80,134,134,134] 
    patches_cnt = [1216,1150,1234,610,460,1040,799,1365,1323,1299] 
    column_name = 'Average Latency'

    # Read a specific column from the dataframe
    avg_cost_ktc = ktc[column_name].values
    avg_cost_no_batch = no_batch[column_name].values
    avg_cost_partition = partition[column_name].values
    avg_cost_vanilla = vanilla[column_name].values

    cost_ktc = avg_cost_ktc * frame_cnt[index]
    cost_no_batch = avg_cost_no_batch * patches_cnt[index]
    cost_partition = avg_cost_partition * frame_cnt[index]
    cost_vanilla = avg_cost_vanilla * frame_cnt[index]

    for cost in [cost_ktc,cost_no_batch,cost_partition,cost_vanilla]:
        for ind,time in enumerate(cost):
            money = Ali_function_cost_usd(time/1000, Mem=4, CPU=2, GPU=6) 
            cost[ind] = money

    ktc_10[index] = np.around(np.mean(cost_ktc),3)
    no_batch_10[index] = np.around(np.mean(cost_no_batch),3)
    partition_10[index] = np.around(np.mean(cost_partition),3)
    vanilla_10[index] = np.around(np.mean(cost_vanilla),3)

    ktc_std[index] = np.around(np.std(cost_ktc),3)
    no_batch_std[index] = np.around(np.std(cost_no_batch),3)
    partition_std[index] = np.around(np.std(cost_partition),3)
    vanilla_std[index] = np.around(np.std(cost_vanilla),3)


penguin_means = {
    r'Tangram ($4\times4$)': ktc_10,
    'Patches': partition_10,
    'Full Frame': vanilla_10,
    'ELF': no_batch_10,
}

x = np.linspace(1,55,10)  # the label locations
width = 1  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize=(25, 5))
colors = [(253/255,185/255,107/255), (254/255, 162/255, 158/255), (114/255,170/255,207/255) ,'#2AB34A']
errors = [ktc_std,partition_std,vanilla_std,no_batch_std]

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement,width, yerr = errors[multiplier], label=attribute, capsize=4,linewidth = 1.5,\
                   edgecolor = 'k', color = colors[multiplier])
    ax.bar_label(rects, padding=3,fontsize='16')
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Function Cost $ (USD)',fontsize='20')
ax.set_title('Cost of Different Methods on PANDA4K Dataset',fontsize='20')
ax.set_xticks(x + width, scene_label)
ax.legend(loc='upper left', ncols=4,fontsize='18')
plt.tick_params(axis='both', which='major', labelsize=18)
plt.savefig('figures/experiment1.pdf',format='pdf',bbox_inches='tight')
plt.show()
