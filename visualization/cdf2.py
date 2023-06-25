
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
df1 = pd.read_csv('logs/4x4/bandwidth=20/B20S1_1.csv')
df2 = pd.read_csv('logs/4x4/bandwidth=40/B40S1_1.csv')
df3 = pd.read_csv('logs/4x4/bandwidth=80/B80S1_1.csv')

df4 = pd.read_csv('logs/4x4/bandwidth=20/B20S11_1.csv')
df5 = pd.read_csv('logs/4x4/bandwidth=20/B20S12_1.csv')
df6 = pd.read_csv('logs/4x4/bandwidth=20/B20S13_1.csv')
df7 = pd.read_csv('logs/4x4/bandwidth=20/B20S14_1.csv')

df8 = pd.read_csv('logs/4x4/bandwidth=40/B40S08_1.csv')
df9 = pd.read_csv('logs/4x4/bandwidth=40/B40S09_1.csv')
df10 = pd.read_csv('logs/4x4/bandwidth=40/B40S11_1.csv')
df11 = pd.read_csv('logs/4x4/bandwidth=40/B40S12_1.csv')

df12 = pd.read_csv('logs/4x4/bandwidth=80/B80S06_1.csv')
df13 = pd.read_csv('logs/4x4/bandwidth=80/B80S07_1.csv')
df14 = pd.read_csv('logs/4x4/bandwidth=80/B80S08_1.csv')
df15 = pd.read_csv('logs/4x4/bandwidth=80/B80S09_1.csv')

column_name = 'Canvas efficiency'
data1 = []
data2 = []
data3 = []
data4 = []

fig, axes = plt.subplots(2, 2,figsize=(11,9))
###################fig1####################
labels1 = ['1.0(s)','1.1(s)','1.2(s)','1.3(s)','1.4(s)']
for df in [df1,df4,df5,df6,df7]:
    data1.append(df[column_name].values)
for index,x in enumerate(data1):
    sns.ecdfplot(data=x,ax=axes[0][0],label=labels1[index],linewidth=3)
axes[0][0].grid(True)
axes[0][0].legend(loc='upper left',fontsize='20')
axes[0][0].set_title('Bandwidth=20Mbps',fontsize='20')
axes[0][0].set_xlabel('(a)',fontsize='20')
axes[0][0].set_ylabel('Likelihood',fontsize='20')
axes[0][0].set_ylim(0,1.01)
axes[0][0].tick_params(axis='both', which='major', labelsize=18)
###################fig2####################
labels2 = ['0.8(s)','0.9(s)','1.0(s)','1.1(s)','1.2(s)']
for df in [df8,df9,df2,df10,df11]:
    data2.append(df[column_name].values)
for index,x in enumerate(data2):
    sns.ecdfplot(data=x,ax=axes[0][1],label=labels2[index],linewidth=3)
axes[0][1].grid(True)
axes[0][1].legend(loc='upper left',fontsize='20')
axes[0][1].set_title('Bandwidth=40Mbps',fontsize='22')
axes[0][1].set_xlabel('(b)',fontsize='20')
axes[0][1].set_ylabel('Likelihood',fontsize='20')
axes[0][1].set_ylim(0,1.01)
axes[0][1].tick_params(axis='both', which='major', labelsize=18)
###################fig3####################
labels3 = ['0.6(s)','0.7(s)','0.8(s)','0.9(s)','1.0(s)']
for df in [df12,df13,df14,df15,df3]:
    data3.append(df[column_name].values)
for index,x in enumerate(data3):
    sns.ecdfplot(data=x,ax=axes[1][0],label=labels3[index],linewidth=3)
axes[1][0].grid(True)
axes[1][0].legend(loc='upper left',fontsize='20')
axes[1][0].set_title('Bandwidth=80Mbps',fontsize='20')
axes[1][0].set_xlabel('(c)',fontsize='20')
axes[1][0].set_ylabel('Likelihood',fontsize='20')
axes[1][0].set_ylim(0,1.01)
axes[1][0].tick_params(axis='both', which='major', labelsize=18)
###################fig4####################
n_bins = 100
labels4 = ['20Mbps','40Mbps','80Mbps']
for df in [df1,df2,df3]:
    data4.append(df[column_name].values)
# plot the cumulative histogram
for index,x in enumerate(data4):
    sns.ecdfplot(data=x,ax=axes[1][1],label=labels4[index],linewidth=3)
    # n, bins, patches = ax.hist(x, n_bins, density=True, histtype='step',alpha=0.5,
    #                         cumulative=True, label=labels[index],linewidth=2,stacked=True)

# tidy up the figure
axes[1][1].grid(True)
axes[1][1].legend(loc='upper left',fontsize='20')
axes[1][1].set_title('SLO=1(s)',fontsize='20')
axes[1][1].set_xlabel('(d)',fontsize='20')
axes[1][1].set_ylabel('Likelihood',fontsize='20')
axes[1][1].set_ylim(0,1.01)
axes[1][1].tick_params(axis='both', which='major', labelsize=18)
########################################
fig.tight_layout()
plt.savefig('figures/cdf1.pdf',format='pdf',bbox_inches='tight')
plt.show()

