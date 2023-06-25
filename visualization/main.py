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
slo_violation_1 = [1.89,1.08,0.00,1.49,1.61]

x2 = ['0.8','0.9','1.0','1.1','1.2']
y2 = [0.016085388,0.015922392,0.015059366,0.014772899,0.014570032]
y2_max = [0.016162658,0.016067041,0.015471404,0.015103054,0.014633924]
y2_min = [0.01598764,0.015716349,0.014780704,0.014635615,0.014468143]
slo_violation_2 = [7.60,3.90,7.81,1.70,6.00]

x3 = ['0.6','0.7','0.8','0.9','1.0']
y3 = [0.017508029,0.016147701,0.015721095,0.015551972,0.014531955]
y3_max = [0.017996467,0.016800564,0.016272065,0.015716471,0.014648029]
y3_min = [0.016799645,0.015337397,0.015433856,0.015306841,0.014415261]
slo_violation_3 = [11.90,5.63,4.92,11.32,4.26]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(16,7))
ax1.plot(x1, y1, '-',linewidth=3)
ax1.fill_between(x1, y1_min, y1_max, alpha=0.2,interpolate=True,label='Koutuchuan (4x4)')
# ax1.set_xlabel('SLO',fontsize='16')
ax1.set_ylabel('Average Cost ($)',fontsize='22')
ax1.set_title('Bandwidth = 20Mbps',fontsize='22')
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.legend(loc='upper left', ncols=4, fontsize='22')
ax1.set_ylim([0.014,0.018])
ax4 = ax1.twinx()
ax4.plot(x1, slo_violation_1, '--',color='r',linewidth=3)
ax4.tick_params(axis='both', which='major', labelsize=20)
ax4.set_ylim([0,20])
ax4.grid(False)
#ax2
ax2.plot(x2, y2, '-',linewidth=3)
ax2.fill_between(x2, y2_min, y2_max, alpha=0.2,interpolate=True)
ax2.set_xlabel('SLO(s)',fontsize='18')
ax2.set_title('Bandwidth = 40Mbps',fontsize='22')
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.set_ylim([0.014,0.018])
ax5 = ax2.twinx()
ax5.plot(x2, slo_violation_2, '--',color='r',linewidth=3)
# ax5.set_ylabel('SLO Violation (%)',fontsize='16')
ax5.tick_params(axis='both', which='major', labelsize=20)
ax5.set_ylim([0,20])
ax5.grid(False)
#ax3
ax3.plot(x3, y3, '-',linewidth=3)
ax3.fill_between(x3, y3_min, y3_max, alpha=0.2,interpolate=True)
# ax3.set_xlabel('SLO',fontsize='16')
# ax3.set_ylabel('Average Cost ($)',fontsize='16')
ax3.set_title('Bandwidth = 80Mbps',fontsize='22')
ax3.tick_params(axis='both', which='major', labelsize=20)
ax3.set_ylim([0.014,0.018])
ax6 = ax3.twinx()
ax6.plot(x3, slo_violation_3, '--',color='r',linewidth=3,label='Koutuchuan')
ax6.set_ylabel('SLO Violation (%)',fontsize='22')
ax6.tick_params(axis='both', which='major', labelsize=20)
ax6.set_ylim([0,20])
ax6.legend(loc='upper right', fontsize='22')
ax6.grid(False)
fig.tight_layout()
plt.savefig('figures/main.pdf',format='pdf')
plt.show()

