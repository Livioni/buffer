import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/Users/livion/Documents/GitHub/Sources/buffer')
from baselines.cost_function import Ali_idle_cost

plt.style.use("seaborn-v0_8-darkgrid")


total_time = 1685005472.094955-1685005372.158279 
idle_time = total_time - 35.0817
idle_cost_of_SLO = Ali_idle_cost(idle_time,4)


species = ('Vanilla4K', 'Partition4K', 'KouTuChuan(1024)','KouTuChuan(1504)','KouTuChuan(2016)','SLO-aware KouTuChuan')
cost_counts = {
    'Idle Cost (CNY)': np.array([0.017165795, 0.00493000 ,0.0044300 ,0.0044300,0.00443000,idle_cost_of_SLO]),
    'Triggering Cost (CNY)': np.array([0.557568086, 0.333268663, 0.149088038,0.136327333,0.142467096,0.0844852]),
}
width = 0.6  # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots(figsize=(12, 5))
bottom = np.zeros(6)

for cost, cost_count in cost_counts.items():
    p = ax.bar(species, cost_count, width, label=cost, bottom=bottom)
    bottom += cost_count

    ax.bar_label(p, label_type='center')

ax.set_title('Total Cost of Different Algorithms')
ax.legend()

# plt.show()
plt.savefig('visualization/figures/cost.png',dpi=300)