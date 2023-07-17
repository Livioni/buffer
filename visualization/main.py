import matplotlib.pyplot as plt
import numpy as np
import matplotlib
plt.style.use("seaborn-v0_8-darkgrid")

matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
matplotlib.rcParams['font.family'] = "sans-serif"
#data
x1 = ['1.0','1.1','1.2','1.3','1.4']
y1 = [0.016598176,0.016173985,0.015627292,0.015186822,0.014868162]
y1_max = [0.016779688,0.01631424,0.015800225,0.015324217,0.01502121]
y1_min = [0.016408453,0.016071705,0.015534208,0.015048425,0.014653791]
slo_violation_1 = [1.887,1.075,0.000,1.492,1.613]

x2 = ['0.8','0.9','1.0','1.1','1.2']
y2 = [0.016085388,0.015922392,0.015059366,0.014772899,0.014570032]
y2_max = [0.016162658,0.016067041,0.015471404,0.015103054,0.014633924]
y2_min = [0.01598764,0.015716349,0.014780704,0.014635615,0.014468143]
slo_violation_2 = [2.083,3.896,4.412,1.695,1.754]

x3 = ['0.6','0.7','0.8','0.9','1.0']
y3 = [0.017508029,0.016384024,0.015721095,0.015368874,0.014531955]
y3_max = [0.017996467,0.016800564,0.016272065,0.015632604,0.014648029]
y3_min = [0.016799645,0.016046366,0.015433856,0.015167176,0.014415261]
slo_violation_3 = [4.902,2.564,3.030,3.389,4.255]

p1 = [0.029902468,0.030135898,0.028736093,0.029315424,0.02950509]
p1_max = [0.031558387,0.031342071,0.029368755,0.030981787,0.030241878]
p1_min = [0.026789615,0.028705702,0.027497851,0.027697149,0.028187484]
slo_violation_p1 = [16.393,9.836,2.459,0.000,0.000]

p2 = [0.031717045,0.030974161,0.030057918,0.033461611,0.031684848]
p2_max = [0.033687458,0.032127716,0.032852908,0.034374358,0.032774213]
p2_min = [0.030310395,0.029768728,0.028644734,0.033405382,0.030303062]
slo_violation_p2 = [9.917,3.333,0.826,0.000,0.000]

p3 = [0.037712753,0.037032846,0.038510806,0.037393617,0.03745247]
p3_max = [0.038352047,0.037778673,0.039141318,0.038306363,0.038170961]
p3_min = [0.037334501,0.035923015,0.038019537,0.036180731,0.036906156]
slo_violation_p3 = [57.850,33.884,10.744,1.653,0.000]

elf20 = [0.021316865,0.021193354,0.020878363,0.020984068,0.020304336]
elf20_min = [0.021042562,0.0209810,0.020654686,0.020015867,0.019836311]
elf20_max = [0.021534835,0.021576973,0.021038715,0.021179756,0.020545842]
elf20_violation = [0.164,0.000,0.000,0.000,0.000]

elf40 = [0.0205295,0.02100607,0.021260421,0.021492422,0.021031821]
elf40_min = [0.019913664,0.020586252,0.020939205,0.020793109,0.02033037]
elf40_max = [0.021146165,0.021758735,0.021795807,0.021888349,0.0220710]
elf40_violation = [1.727,0.247,0.082,0.000,0.000]

elf80 = [0.021289617,0.02108824,0.021304991,0.021294594,0.021069619]
elf80_min = [0.02072772,0.020449237,0.020364332,0.021033777,0.020547722]
elf80_max = [0.02182615,0.021610325,0.02219744,0.021490713,0.021547199]
elf80_violation = [19.079,6.743,0.905,0.164,0.000]


fig, axes = plt.subplots(2, 3,figsize=(16,10))
ax1 = axes[0,0]
ax2 = axes[0,1]
ax3 = axes[0,2]
ax4 = axes[1,0]
ax5 = axes[1,1]
ax6 = axes[1,2]
color_koutuchun = [253/255,181/255,102/255]
color_padding = [1,153/255,149/255]
color_elf = '#2AB34A'
##############################ax1###########################################
ax1.plot(x1, y1, '.-',linewidth=3,markersize=15,color=color_koutuchun,label='Tangram')
ax1.fill_between(x1, y1_min, y1_max, alpha=0.5,interpolate=True,color=color_koutuchun)
ax1.plot(x1, p1, '.-',linewidth=3,markersize=15,color=color_padding,label='Fixed Batch 10')
ax1.fill_between(x1, p1_min, p1_max, alpha=0.5,interpolate=True,color=color_padding)
ax1.plot(x1, elf20, '.-',linewidth=3,markersize=15,color=color_elf,label='ELF')
ax1.fill_between(x1, elf20_min, elf20_max, alpha=0.5,interpolate=True,color=color_elf)
# ax1.set_xlabel('SLO',fontsize='16')
ax1.set_ylabel('Average Cost ($)',fontsize='26')
ax1.set_title('Bandwidth = 20Mbps',fontsize='26')
ax1.tick_params(axis='both', which='major', labelsize=24)


ax1.set_ylim(0.013,0.04)
ax4.plot(x1, slo_violation_1, '.-',linewidth=3,markersize=15,color=color_koutuchun,label='Tangram')
ax4.plot(x1, slo_violation_p1, '.-',linewidth=3,markersize=15,color=color_padding,label='Fixed Batch 10')
ax4.plot(x1, elf20_violation, '.-',linewidth=3,markersize=15,color=color_elf,label='ELF')
ax4.axhline(5, ls='--',color='k', lw=2, alpha=0.5)
ax4.tick_params(axis='both', which='major', labelsize=24)
ax4.legend(loc='upper left', fontsize='22')
ax4.set_ylabel('SLO Violation (%)',fontsize='24')
ax4.set_ylim(0,60)

##############################ax2###########################################
ax2.plot(x2, y2, '.-',linewidth=3,markersize=15,color=color_koutuchun)
ax2.fill_between(x2, y2_min, y2_max, alpha=0.5,interpolate=True,color=color_koutuchun)
ax2.plot(x2, p2, '.-',linewidth=3,markersize=15,color=color_padding)
ax2.fill_between(x2, p2_min, p2_max, alpha=0.5,interpolate=True,color=color_padding)
ax2.plot(x2, elf40, '.-',linewidth=3,markersize=15,color=color_elf)
ax2.fill_between(x2, elf40_min, elf40_max, alpha=0.5,interpolate=True,color=color_elf)
ax2.set_xlabel('SLO(s)',fontsize='26')
ax2.set_title('Bandwidth = 40Mbps',fontsize='26')
ax2.tick_params(axis='both', which='major', labelsize=24)
ax2.set_ylim(0.013,0.04)
ax5.plot(x2, slo_violation_2, '.-',linewidth=3,markersize=15,color=color_koutuchun)
ax5.plot(x2, slo_violation_p2, '.-',linewidth=3,markersize=15,color = color_padding)
ax5.plot(x2, elf40_violation, '.-',linewidth=3,markersize=15,color=color_elf)
ax5.axhline(5, ls='--',color='k', lw=2, alpha=0.5)
ax5.set_xlabel('SLO(s)',fontsize='24')
ax5.tick_params(axis='both', which='major', labelsize=24)
ax5.set_ylim(0,60)
##############################ax3###########################################
ax3.plot(x3, y3, '.-',linewidth=3,markersize=15,color=color_koutuchun)
ax3.fill_between(x3, y3_min, y3_max, alpha=0.5,interpolate=True,color=color_koutuchun)
ax3.plot(x3, p3, '.-',linewidth=3,markersize=15,color=color_padding)
ax3.fill_between(x3, p3_min, p3_max, alpha=0.5,interpolate=True,color=color_padding)
ax3.plot(x3, elf80, '.-',linewidth=3,markersize=15,color=color_elf)
ax3.fill_between(x3, elf80_min, elf80_max, alpha=0.5,interpolate=True,color=color_elf)
# ax3.set_xlabel('SLO',fontsize='16')
# ax3.set_ylabel('Average Cost ($)',fontsize='16')
ax3.set_title('Bandwidth = 80Mbps',fontsize='26')
ax3.tick_params(axis='both', which='major', labelsize=24)
ax3.set_ylim(0.013,0.04)
ax6.plot(x3, slo_violation_3, '.-',linewidth=3,markersize=15,color=color_koutuchun)
ax6.plot(x3, slo_violation_p3, '.-',linewidth=3,markersize=15,color=color_padding)
ax6.plot(x3, elf80_violation, '.-',linewidth=3,markersize=15,color=color_elf)
ax6.tick_params(axis='both', which='major', labelsize=24)
ax6.axhline(5, ls='--',color='k', lw=2, alpha=0.5)
ax6.set_ylim(0,60)
fig.tight_layout()
plt.savefig('figures/main.pdf',format='pdf')
plt.show()

