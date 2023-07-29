import matplotlib.pyplot as plt
import numpy as np
import matplotlib
plt.style.use("seaborn-v0_8-darkgrid")

matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
matplotlib.rcParams['font.family'] = "sans-serif"

# make data
x = ['4K','2K','1080P','720P','480P']
y = [0.744,0.736,0.691,0.6,0.374]
y2 = [0.101,0.189,0.264,0.389,0.462]

# plot
fig, ax = plt.subplots(figsize=(6,5))

x_point = [0,1,2,3,4]
ax.plot(x, y, linewidth=2.5,marker = 'o',markersize=8)

for i in range(len(x)):
    plt.text(x[i], y[i], f'{y[i]:.3f}', ha='right', va='bottom',fontsize=20)

ax.plot(x, y2, linewidth=2.5,marker = 'o',markersize=8)


for i in range(len(x)):
    plt.text(x[i], y2[i], f'{y2[i]:.3f}', ha='right', va='bottom',fontsize=20)

ax.set_xlabel('Resolution', fontsize=22)
ax.set_ylabel('Average Precision', fontsize=22)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.legend(['Downsize','Upsize'],fontsize=20)
plt.tight_layout()
plt.savefig('figures/AP.pdf',format='pdf')
plt.show()




