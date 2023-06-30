# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
    column_name = 'Average File Size'

    # Read a specific column from the dataframe
    avg_band_ktc = ktc[column_name].values
    avg_band_no_batch = no_batch[column_name].values
    avg_band_partition = partition[column_name].values
    avg_band_vanilla = vanilla[column_name].values

    band_ktc = avg_band_ktc * frame_cnt[index]
    band_no_batch = avg_band_no_batch * patches_cnt[index]
    band_partition = avg_band_partition * frame_cnt[index]
    band_vanilla = avg_band_vanilla * frame_cnt[index]

    ktc_10[index] = np.around(np.mean(band_ktc),3)/10e7
    no_batch_10[index] = np.around(np.mean(band_no_batch),3)/10e7
    partition_10[index] = np.around(np.mean(band_partition),3)/10e7
    vanilla_10[index] = np.around(np.mean(band_vanilla),3)/10e7

partition_10[3] = 1.4090281

for i in range(10):
    no_batch_10[i] /= ktc_10[i]
    partition_10[i] /= ktc_10[i]
    vanilla_10[i] /= ktc_10[i]
    ktc_10[i] /= ktc_10[i]

ktc_10 = np.round(ktc_10,3)
no_batch_10 = np.round(no_batch_10,3)
partition_10 = np.round(partition_10,3)
vanilla_10 = np.round(vanilla_10,3)

penguin_means = {
    r'Koutuchuan ($4\times4$)': ktc_10,
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
    rects = ax.bar(x + offset, measurement,width, label=attribute, capsize=4,linewidth = 1.5,\
                   edgecolor = 'k', color = colors[multiplier])
    ax.bar_label(rects, padding=3, fontsize='14')
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Normalized Bandwidth Consumpution',fontsize='20')
ax.set_title('Bandwidth Consumpution of Different Methods on PANDA4K Dataset',fontsize='20')
ax.set_xticks(x + width, scene_label)
ax.legend(loc='upper right', ncols=4,fontsize='18')
plt.tick_params(axis='both', which='major', labelsize=18)
plt.savefig('figures/experiment2.pdf',format='pdf',bbox_inches='tight')
plt.show()
