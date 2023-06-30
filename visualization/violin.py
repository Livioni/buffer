import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
plt.style.use("seaborn-v0_8-darkgrid")

matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
matplotlib.rcParams['font.family'] = "sans-serif"

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=17)

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

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

column_name = 'Latency (ms)'
data = []
for df in [df1,df2,df3]:
    temp = df[column_name].T.dropna().T.values
    data.append(temp/1000)

data.append(data)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10.5, 9))
colors = [(253/255,185/255,107/255), (254/255, 162/255, 158/255), (114/255,170/255,207/255) ,'#2AB34A','#D2ACF7']

species = (
    "20Mbps,1.0s",
    "40Mbps,1.0s",
    "80Mbps,1.0s",
)

latency1 = [21.8721136,22.1003958,21.7975106]
latency2 = [19.4760568,19.6671264,20.3856155]
latency3 = [18.9980753,19.1525231,19.3053078]

width = 0.5
bottom = np.zeros(3)
latency = [np.mean(latency1),np.mean(latency2),np.mean(latency3)]
std = [np.std(latency1),np.std(latency2),np.std(latency3)]
weight_counts = {
    "Transmission Time ": np.array([34.7689404, 17.3844702, 8.6922351]),
    "Function Execution Time ": latency,
    
}
colors = [(253/255,185/255,107/255), (254/255, 162/255, 158/255), (114/255,170/255,207/255) ,'#2AB34A']
i=0
for boolean, weight_count in weight_counts.items():
    if i > 0:
        p = axes[1][0].bar(species, weight_count, width, yerr=std[i], label=boolean, bottom=bottom,color=colors[2],edgecolor='k',capsize=4)
    else:
        p = axes[1][0].bar(species, weight_count, width, label=boolean, bottom=bottom,color=colors[0],edgecolor='k',capsize=4)
    bottom += weight_count
    i+=1

axes[1][0].set_xlabel("(c) Latency Breakdown", fontsize=17)
axes[1][0].set_ylabel("Latency (s)",fontsize=17)
axes[1][0].legend(loc="upper right",fontsize=17)
axes[1][0].tick_params(axis='both', which='major', labelsize=17)
axes[1][0].set_ylim(0, 80)

x = np.linspace(0,2,3)  # the label locations
width = 0.5  # the width of the bars

def add_label(violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))


labels = ["20Mbps,1.0s","40Mbps,1.0s","80Mbps,1.0s"]
for index,i in enumerate(x):
    parts = axes[0][0].violinplot(data[index], positions=[i],widths=width,showmeans=True,showextrema=True,points=100)
    add_label(parts, labels[index])

# axes[0][0].legend(*zip(*labels), loc='upper left',fontsize=15)
axes[0][0].set_xlabel('(a) Function Execution Latency Per-batch Distribution', fontsize=17)
axes[0][0].set_ylabel('Latency (s)', fontsize=17)
axes[0][0].set_xticks([0,1,2],labels=["20Mbps,1.0s","40Mbps,1.0s","80Mbps,1.0s"])
axes[0][0].tick_params(axis='both', which='major', labelsize=17)


hist1 = df1['Images Number'].values.tolist()
hist2 = df2['Images Number'].values.tolist()
hist3 = df3['Images Number'].values.tolist()

mu1 = np.mean(hist1)
mu2 = np.mean(hist2)
mu3 = np.mean(hist3)
sigma1 = np.std(hist1)
sigma2 = np.std(hist2)
sigma3 = np.std(hist3)
x1 = mu1 + sigma1 * np.random.randn(100)
x2 = mu2 + sigma2 * np.random.randn(100)
x3 = mu3 + sigma3 * np.random.randn(100)

n, bins1, patches = axes[0][1].hist(hist3, bins=15, linewidth=0.5, edgecolor="k",alpha=0.7,color = [185/255,209/255,188/255],orientation="horizontal",density=True,label="80Mbps,1.0s")
n, bins2, patches = axes[0][1].hist(hist2, bins=15, linewidth=0.5, edgecolor="k",alpha=0.7,color = [232/255,202/255,180/255],orientation="horizontal",density=True,label="40Mbps,1.0s")
n, bins3, patches = axes[0][1].hist(hist1, bins=15, linewidth=0.5, edgecolor="k",alpha=0.7,color = [180/255,198/255,226/255],orientation="horizontal",density=True,label="20Mbps,1.0s")

bins = np.linspace(0, 40, 100)

y1 = ((1 / (np.sqrt(2 * np.pi) * sigma1)) *
     np.exp(-0.5 * (1 / sigma1 * (bins - mu1))**2))
y2 = ((1 / (np.sqrt(2 * np.pi) * sigma2)) *
        np.exp(-0.5 * (1 / sigma2 * (bins - mu2))**2))
y3 = ((1 / (np.sqrt(2 * np.pi) * sigma3)) *
        np.exp(-0.5 * (1 / sigma3 * (bins - mu3))**2))

axes[0][1].plot(y1, bins, '--',color = '#40A9FE',linewidth=2)
axes[0][1].plot(y2, bins, '--',color = '#F98B15',linewidth=2)  
axes[0][1].plot(y3, bins, '--',color = '#7BB305',linewidth=2)


axes[0][1].set_xlabel('(b) Patches Number Per-batch Distribution ', fontsize=17)
axes[0][1].set_ylabel('Patches Number', fontsize=17)
axes[0][1].tick_params(axis='both', which='major', labelsize=16)
axes[0][1].set_yticks(np.linspace(40,0,5,endpoint=True))
axes[0][1].legend(loc="upper right",fontsize=15)
axes[0][1].xaxis.set_major_formatter(PercentFormatter(xmax=1))


vegetables = ["1", "2", "3", "4",'5','6','7','8','9']
farmers = ["1~5","6~10","11~14","15~18","19~23","24~27","28~31","32~36","36~40"]

harvest = np.array([[23,  3,   0,	0,	0,	0,	0,	0,	0],
                    [0,	6,	23,	8,	0,	0,	0,	0,	0],
                    [0,	0,	16,	12,	4,	1,	0,	0,	0],
                    [0,	0,	3,	18,	4,	4,	2,	0,	0],
                    [0,	0,	0,	8,	7,	15,	1,	1,	2],
                    [0,	0,	0,	0,	3,	13,	3,	4,	1],
                    [0,	0,	0,	0,	1,	1,	6,	6,	0],
                    [0,	0,	0,	0,	0,	3,	9,	5,	0],
                    [0,	0,	0,	0,	0,	0,	1,	1,	0]])


harvest_norm = np.round(np.array([line / line.sum(axis=0) for line in harvest]),2)

axes[1][1].set_xticks(np.arange(len(farmers)), labels=farmers)
axes[1][1].set_yticks(np.arange(len(vegetables)), labels=vegetables)
axes[1][1].imshow(harvest_norm,cmap="YlGnBu")

# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = axes[1][1].text(j, i, harvest_norm[i, j],ha="center", va="center",)

axes[1][1].set_xlabel("(d) Patches that the canvases contains.",fontsize=17)
axes[1][1].set_ylabel('Canvases Number', fontsize=17)
axes[1][1].tick_params(axis='both', which='major', labelsize=17)
# plt.setp(axes[1][1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

im, cbar = heatmap(harvest_norm, vegetables, farmers, ax=axes[1][1],
                   cmap="YlGnBu", cbarlabel="Proportion")

fig.tight_layout()
plt.savefig('figures/violin.pdf',format='pdf',bbox_inches='tight')
plt.show()