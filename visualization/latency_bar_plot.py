import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/Users/livion/Documents/GitHub/Sources/buffer')
from baselines.cost_function import Ali_idle_cost

plt.style.use("seaborn-v0_8-darkgrid")

species = ('Vanilla4K', 'Partition4K', 'KouTuChuan(1024)','KouTuChuan(1504)','KouTuChuan(2016)','SLO-aware KouTuChuan')
cost_counts = {
    'Transmission Time (s)': np.array([337.909347, 97.139139, 86.854763 ,86.854763,86.854763,86.854763]),
    'Inference Time (s)': np.array([179.5394504, 107.2966578, 39.14897969,43.86541273,45.84291937,36.2833]),
}
width = 0.6  # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots(figsize=(12, 5))
bottom = np.zeros(6)

for cost, cost_count in cost_counts.items():
    p = ax.bar(species, cost_count, width, label=cost, bottom=bottom)
    bottom += cost_count

    ax.bar_label(p, label_type='center')

ax.set_title('Latency of Different Algorithms')
ax.legend()

# plt.show()
plt.savefig('visualization/figures/latency.png',dpi=300)