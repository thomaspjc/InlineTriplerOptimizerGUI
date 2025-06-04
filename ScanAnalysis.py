#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:59:42 2025

Reading the 2D scan from the Tripler GUI

@author: thomas
"""

import h5py 
import numpy as np
import matplotlib.pyplot as plt
from cmocean import cm as cmo
from mpl_toolkits.mplot3d import axes3d 




def readScan2D(filepath):
    
    # ––– Read the datasets from the file –––––––––––––––––––––––––––––––––––––
    with h5py.File(filepath, 'r') as file:
        motor1 = file['motor1'][:] #shape (N,N)
        motor2 = file['motor2'][:] #shape (N,N)
        energy = file['energies'][:] #shape (N, N, 5)
    
    energy = np.mean(energy, axis = 2)
    return motor1, motor2, energy


def plotScan2D(filepath):
    
    motor1, motor2, energy = readScan2D(filename)
    
    # ─── “Unscramble” the snake‐scan pattern in motor1 & energy ────────────────────
    X = motor1.copy()
    Z = energy.copy()
    for i in range(X.shape[0]):
        if i % 2 == 1:
            X[i, :] = X[i, ::-1]
            Z[i, :] = Z[i, ::-1]
    # motor2 is already monotonic along each row:
    Y = motor2
    
    # ─── Compute axis‐limits for the contour offsets ───────────────────────────────
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    z_min, z_max = Z.min(), Z.max()
    
    # 2) Create a 3D surface plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        X, Y, Z,
        cmap= cmo.curl,        # you can choose any Matplotlib colormap
        edgecolor='none',
        linewidth=0, 
        antialiased=True
    )
    
    # Plot projected contours onto the walls of the box using ax.contour
    ax.contour(
        X, Y, Z,
        zdir='z',
        offset=z_min,
        cmap='coolwarm'
    )
    ax.contour(
        X, Y, Z,
        zdir='x',
        offset=x_min,
        cmap='coolwarm'
    )
    ax.contour(
        X, Y, Z,
        zdir='y',
        offset=y_max,
        cmap='coolwarm'
    )

    # 3) Labels and colorbar
    ax.set_xlabel('Motor 1 position')
    ax.set_ylabel('Motor 2 position')
    ax.set_zlabel('Average Energy')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Energy')

    plt.tight_layout()
    plt.show()
    return motor1, motor2, energy


if __name__ == "__main__":
    filename = "test_long_scan.h5"
    motor1, motor2, energy = plotScan2D(filename)

   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
