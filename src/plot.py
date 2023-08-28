import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import matplotlib.pyplot as plt
import time
import pandas as pd

# colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], 
# [255, 0, 255],  [176, 48, 96], [46, 139, 87], [160, 32, 240], [255, 127, 80],
#  [127, 255, 212],  [218, 112, 214], [160, 82, 45], [127, 255, 0], [216, 191, 216]]

colors = [[0, 0, 1],
    [127, 255, 0],[0, 255, 0], [0, 0, 255], [46, 139, 87],[255, 0, 255], 
    [0, 255, 255],[255, 255, 255], [160, 82, 45], [160, 32, 240], [255, 127, 80],
    [218, 112, 214], [255, 0, 0], [255, 255, 0], [127, 255, 212],  [216, 191, 216]
]
def data_to_colormap2(data):
    assert len(data.shape)==2
    x_list = data.reshape((-1,))
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        y[index] = np.array(colors[item]) / 255
    return y


def data_to_colormap(data):
    assert len(data.shape)==2
    x_list = data.reshape((-1,))
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([147, 67, 46]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 100, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 123]) / 255.
        if item == 5:
            y[index] = np.array([164, 75, 155]) / 255.
        if item == 6:
            y[index] = np.array([101, 174, 255]) / 255.
        if item == 7:
            y[index] = np.array([118, 254, 172]) / 255.
        if item == 8:
            y[index] = np.array([60, 91, 112]) / 255.
        if item == 9:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 10:
            y[index] = np.array([255, 255, 125]) / 255.
        if item == 11:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 12:
            y[index] = np.array([100, 0, 255]) / 255.
        if item == 13:
            y[index] = np.array([0, 172, 254]) / 255.
        if item == 14:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 15:
            y[index] = np.array([171, 175, 80]) / 255.
        if item == 16:
            y[index] = np.array([101, 193, 60]) / 255.

    return y


def classification_map(name, map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1]*2.0/dpi, ground_truth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.set_title(name)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)
    return 0

def show_map(map, data, dpi):
    fig = plt.figure(figsize=(12,10), frameon=False)
    fig.set_size_inches(data.shape[1]*2.0/dpi, data.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    return 0