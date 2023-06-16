
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sys,matplotlib
sys.path.append('/Users/livion/Documents/GitHub/Sources/buffer') 
plt.style.use("seaborn-v0_8-darkgrid")

matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
matplotlib.rcParams['font.family'] = "sans-serif"

# Read the Excel file
all_scene_name = ['scene_01','scene_02','scene_03','scene_04','scene_05','scene_06',\
                  'scene_07','scene_08','scene_09','scene_10'] #,
scene_label = ['scene_01 (#134)','scene_02 (#134)','scene_03 (#134)','scene_04 (#48)','scene_05 (#33)','scene_06 (#122)',\
                  'scene_07 (#80)','scene_08 (#134)','scene_09 (#134)','scene_10 (#134)'] #,
scene_number = len(all_scene_name)


ktc_patches = []

for index,scene_name in enumerate(all_scene_name):
    if index == 9:
        ktc = pd.read_csv('data/ktc_copy/scene_10/ktc_partitions_10_1.csv')
    else:
        ktc = pd.read_csv('data/ktc_copy/scene_0'+str(index+1)+ '/ktc_partitions_0' + str(index+1) + '_1.csv')

    frame_cnt = [134,134,134,48,33,122,80,134,134,134] 
    patches_cnt = [1216,1150,1234,610,460,1040,799,1365,1323,1299] 
    column_name = 'Image Number'

    # Read a specific column from the dataframe
    ktc_patches.append(ktc[column_name].values)


fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(9,8))
# make a little extra space between the subplots
fig.subplots_adjust(hspace=0.2)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']




for index,scene_name in enumerate(all_scene_name):
    if index <= 4:
        ax1.plot(ktc_patches[index], label=all_scene_name[index],color = colors[index],linewidth=2)   
        ax1.set_xlabel('Frame Index',fontsize='20')
        ax1.set_ylabel('Number of Patches',fontsize='20')
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.grid(True)
        ax1.legend(loc='upper right', ncols=2,fontsize='18')
        ax1.set_title('Patches Per Frame of Koutucuan in Different Scene',fontsize='22')
    else:
        ax2.plot(ktc_patches[index], label=all_scene_name[index],color = colors[index],linewidth=2)
        ax2.set_xlabel('Frame Index',fontsize='20')
        ax2.set_ylabel('Number of Patches',fontsize='20')
        ax2.tick_params(axis='both', which='major', labelsize=16)
        ax2.grid(True)
        ax2.legend(loc='upper center', ncols=3,fontsize='18')


plt.savefig('figures/patches.pdf',format='pdf',bbox_inches='tight')
plt.show()