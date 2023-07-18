
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys,matplotlib
import seaborn as sns
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
frame_cnt = [134,134,134,48,33,122,80,134,134,134] 

scene_number = len(all_scene_name)

scenes = []

for index,scene_name in enumerate(all_scene_name):
    if index == 9:
        ktc = pd.read_csv('data/ktc_copy/' + scene_name + '/' + 'ktc_partitions_10_' + str(index+1) + '.csv')
    else:
        ktc = pd.read_csv('data/ktc_copy/' + scene_name + '/' + 'ktc_partitions_0' + str(index+1) +'_' + str(index+1) + '.csv')

    column_name = 'Canvas Efficiency'

    # Read a specific column from the dataframe
    scenes.append(ktc[column_name].values)


fig, ax = plt.subplots(figsize=(9, 8))
n_bins = 100
labels = ['scene_01','scene_02','scene_03','scene_04','scene_05','scene_06',\
                  'scene_07','scene_08','scene_09','scene_10']
# plot the cumulative histogram
for index,x in enumerate(scenes):
    sns.ecdfplot(data=x,ax=ax,label=labels[index],linewidth=3)
    # n, bins, patches = ax.hist(x, n_bins, density=True, histtype='step',alpha=0.5,
    #                         cumulative=True, label=labels[index],linewidth=2,stacked=True)

# tidy up the figure
ax.grid(True)
ax.legend(loc='upper left',fontsize='21')
ax.set_title('ECDF of Bin Packing Efficiency in Different Scene',fontsize='24')
ax.set_xlabel(r'Canvas Efficiency ($4\times4,1024$)',fontsize='24')
ax.set_ylabel('Likelihood of Occurrence',fontsize='24')
ax.set_ylim(0,1.01)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.savefig('figures/cdf.pdf',format='pdf',bbox_inches='tight')
plt.show()

