
import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from classicalStereo.datasetHandler import Dataset_Handler
from dlStereo.dataHandler_ChiTransformer import ChiTransformer_StereoDepth

def plot_odometry(infer_fp,output_plot_fp,sequence):
    classic_trajectory = np.load(f'{infer_fp}/trajectory-{sequence}_classic.npy')
    dl_trajectory = np.load(f'{infer_fp}/trajectory-{sequence}_dl.npy')
    handler = Dataset_Handler(sequence)

    # Plot Classic Stereo Odometry vs Ground Truth
    fig1 = plt.figure(figsize=(10, 10))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.view_init(elev=-40, azim=270)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    xs = handler.gt[:, 0, 3]
    ys = handler.gt[:, 1, 3]
    zs = handler.gt[:, 2, 3]
    ax1.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    ax1.plot(xs, ys, zs, label='Ground Truth', c='green')
    xs = classic_trajectory[:, 0, 3]
    ys = classic_trajectory[:, 1, 3]
    zs = classic_trajectory[:, 2, 3]
    ax1.plot(xs, ys, zs, c='orange', label='Classic Stereo Odometry')
    ax1.legend()
    ax1.set_title('Classic Stereo Odometry vs Ground Truth')
    fig1.savefig(f'{output_plot_fp}/classic_{sequence}_odometry.png', bbox_inches='tight')
    plt.close(fig1)

    # Plot Deep Learning Odometry vs Ground Truth
    fig2 = plt.figure(figsize=(10, 10))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.view_init(elev=-40, azim=270)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    xs = handler.gt[:, 0, 3]
    ys = handler.gt[:, 1, 3]
    zs = handler.gt[:, 2, 3]
    ax2.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    ax2.plot(xs, ys, zs, label='Ground Truth', c='green')
    xs = dl_trajectory[:, 0, 3]
    ys = dl_trajectory[:, 1, 3]
    zs = dl_trajectory[:, 2, 3]
    ax2.plot(xs, ys, zs, c='red', label='DL Stereo Odometry')
    ax2.legend()
    ax2.set_title('Deep Learning Odometry vs Ground Truth')
    fig2.savefig(f'{output_plot_fp}/dl_{sequence}_odometry.png', bbox_inches='tight')
    plt.close(fig2)
