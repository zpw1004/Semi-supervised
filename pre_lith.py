import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import set_option
import torch

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
set_option("display.max_rows", 10)
pd.options.mode.chained_assignment = None

df = pd.read_excel("dataset/dc.xlsx")
training_data = df.dropna()

file_path ="output/predicted.txt"
data_list = []

with open(file_path, "r") as file:
    for line in file:
        data_list.append(int(line.strip()))

res = torch.tensor(data_list)
pre_log = res
print("pre_Log:", pre_log.shape)

facies_colors = ['#F4D03F', '#F5B041', '#DC7633', '#6E2C00', '#1B4F72', '#2E86C1', '#AED6F1', '#A569BD', '#196F3D',
                 '#FF5733', '#C70039', '#900C3F', '#581845', '#7D3C98','#56992F']

facies_labels = [
    "LGFS",  # Light Gray Fluorescent Siltstone
    "GCB",   # Gray Calcareous Breccia
    "GCS",   # Gray Calcareous Siltstone
    "GM",    # Gray Marlstone
    "GSM",   # Gray Silty Mudstone
    "GBCM",  # Gray-Brown Calcareous Mudstone
    "LGFFS", # Light Gray Fluorescent Fine Sandstone
    "GAM",   # Gray Argillaceous Marlstone
    "GCM",   # Gray Calcareous Mudstone
    "GBSM",  # Gray-Brown Silty Mudstone
    "LGFS",  # Light Gray Fine Sandstone
    "GS",    # Gray Siltstone
    "BGAS",  # Brown-Gray Argillaceous Siltstone
    "GM",    # Gray Mudstone
    "BM"     # Brown Mudstone
]
facies_color_map = {label: facies_colors[ind] for ind, label in enumerate(facies_labels)}
def make_facies_log_plot(logs, pre_log, facies_colors):
    logs = logs.sort_values(by='DEPTH')
    cmap_facies = colors.ListedColormap(facies_colors, 'indexed')
    ztop = logs.DEPTH.min()
    zbot = logs.DEPTH.max()
    cluster = np.repeat(np.expand_dims(logs['class'].values, 1), 100, 1)
    pre_cluster = np.repeat(np.expand_dims(pre_log, 1), 100, 1)
    f, ax = plt.subplots(nrows=1, ncols=7, figsize=(18, 12))  # 调整图表尺寸
    ax[0].plot(logs.GR, logs.DEPTH, '-g')
    ax[1].plot(np.log10(logs.DT24), logs.DEPTH, '-b')
    ax[2].plot(np.log10(logs.M2R3), logs.DEPTH, '-', color='yellowgreen')
    ax[3].plot(np.log10(logs.M2R9), logs.DEPTH, '-', color='r')
    ax[4].plot(logs.SP, logs.DEPTH, '-', color='black')
    im = ax[5].imshow(cluster, interpolation='none', aspect='auto', cmap=cmap_facies, vmin=0, vmax=14)
    ax[6].imshow(pre_cluster, interpolation='none', aspect='auto', cmap=cmap_facies, vmin=0, vmax=14)
    tick_positions = np.linspace(0.5, 13.5, num=len(facies_labels))
    divider = make_axes_locatable(ax[6])
    cax = divider.append_axes("right", size="30%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ticks=tick_positions, shrink=0.8)
    cbar.ax.set_yticklabels(facies_labels, rotation=90, va='center')
    cbar.ax.tick_params(size=0)
    cbar.ax.tick_params(axis='y', which='major', labelsize=10)
    for i in range(len(ax) - 2):
        ax[i].set_ylim(ztop, zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    ax[0].set_xlabel("GR", fontsize=14)
    ax[1].set_xlabel("DT24", fontsize=14)
    ax[2].set_xlabel("M2R3", fontsize=14)
    ax[3].set_xlabel("M2R9", fontsize=14)
    ax[4].set_xlabel("SP", fontsize=14)
    ax[5].set_xlabel('Facies', fontsize=14)
    ax[6].set_xlabel('Predict Facies', fontsize=14)
    ax[1].set_yticklabels([])
    ax[2].set_yticklabels([])
    ax[3].set_yticklabels([])
    ax[4].set_yticklabels([])
    ax[5].set_xticklabels([])
    ax[5].set_yticklabels([])
    ax[6].set_xticklabels([])
    ax[6].set_yticklabels([])
    plt.subplots_adjust(top=0.8)
    f.suptitle('Well Name: WELL3', fontsize=18, y=1.05)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0)
    plt.savefig("pre_lith.jpg", dpi=600, format="jpg")
    plt.show()
make_facies_log_plot(training_data, pre_log, facies_colors)
