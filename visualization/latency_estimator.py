import matplotlib.pyplot as plt
import numpy as np
import matplotlib
plt.style.use("seaborn-v0_8-darkgrid")

matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
matplotlib.rcParams['font.family'] = "sans-serif"


table = [0.090219428,0.151656122,0.19959718,0.280965798,0.339302976,
           0.388410434,0.4671203646,0.507089878,0.55965937,0.611210174]

# make data
x = range(1,11)
y = table

# plot
fig, ax = plt.subplots()

ax.plot(x, y, linewidth=2.0)
ax.set_ylim(0,1)
ax.set_xlim(0,10)

fig.tight_layout()
plt.savefig('figures/estimator.pdf',format='pdf')
plt.show()