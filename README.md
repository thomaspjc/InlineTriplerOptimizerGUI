# TRIPLER GUI

**Version:** 0.1.0  
**Date:** 2025-06-04

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Launching the GUI](#launching-the-gui)  
  - [Tab 1: Inline Tripler Optimizer](#tab-1-inline-tripler-optimizer)  
  - [Tab 2: Manual Control](#tab-2-manual-control)  
- [Configuration](#configuration)  
- [Stopping/Closing](#stoppingclosing)  
- [Known Issues & Limitations](#known-issues--limitations)  
- [Future Improvements](#future-improvements)  
- [License](#license)  

---

## Overview

**TRIPLER GUI** is a two-tab Python/Tkinter application designed to:

1. Control four EPICS (or simulated) motors  
2. Perform automated optimization of SHG/THG phase-matching via a hybrid Bayesian‐Optimization + Gradient‐Ascent routine  
3. Execute both 1D and 2D “serpentine” motor scans and store results to HDF5 files  
4. Monitor and plot live energy readings from a power‐meter PV  

This initial release (v0.1.0) provides a fully functional interface for coarse/fine motor adjustments, real‐time energy monitoring, and flexible scanning workflows.

---

## Features

1. **Two‐Tab Interface**  
   - **Tab 1 “Inline Tripler Optimizer”**  
     - Schematic of the tripler assembly (loads `tripler.png` or `tripler_dark.png`)  
     - Jog arrows for coarse adjustment of Motor 1 (SHG) and Motor 3 (THG)  
     - **OPTIMIZE / STOP** buttons: launch and abort the BO + GA optimizer  
     - Live “Energy Monitor” plot updating from PV `PMTR:LR20:50:PWR` (or dummy PV in test mode)  

   - **Tab 2 “Manual Control”**  
     - Four motor panels (Motors 1–4) each with:  
       - PV entry + **Connect** button  
       - Current readback label  
       - Step‐size entry + jog ◀/▶ buttons for fine motion  
     - **Motor Scan** section: toggle between 1D and 2D modes  
       - **1D Scan:** Sweep a single motor; collect _N_ energy readings at each step; save to HDF5  
       - **2D Scan:** Serpentine pattern (zigzag) over two motors; collect _N_ readings per (x, y); save to HDF5  

2. **Background Monitoring Threads**  
   - `MotorMonitor`: polls each motor PV in its own daemon thread; updates GUI labels asynchronously  
   - `EnergyMonitor`: polls the energy PV in a daemon thread; schedules live plot updates via `tkinter.after(0, ...)`  

3. **Bayesian‐Optimization + Gradient‐Ascent**  
   - **BayesianOptimize**:  
     - Generates `n_init` random samples of \((\theta_1, \theta_2)\) within user‐defined bounds  
     - Fits a Gaussian Process (Matérn kernel + White noise)  
     - Uses Expected Improvement (EI) acquisition to propose `n_iter` successive test points  
   - **GradientAscent**:  
     - Starting from the best BO result, performs finite‐difference gradient ascent to refine the local maximum  

4. **1D & 2D Serpentine Scan Logic**  
   - **1D Scan**:  
     1. Build an array `positions = np.arange(start, stop + step≈, step)`  
     2. For each `pos`: move motor, wait, collect \(N\) energy readings, store to dataset  
   - **2D Scan** (“Serpentine”):  
     1. Build `positions1` and `positions2` arrays for two motors  
     2. Use `np.meshgrid(positions1, positions2)` and reverse every second row:  
        ```python
        motor1[1::2] = motor1[1::2, ::-1]
        ```  
     3. Outer loop: move Motor 2 to each `pos2`  
     4. Inner loop: traverse Motor 1 over its (possibly reversed) array, collect \(N\) readings  
     5. Store `motor1`, `motor2`, and `energies` datasets to HDF5  

5. **Dark / Light Mode Toggle**  
   - Toolbar button to switch theme:  
     - Updates background/foreground colors of all widgets  
     - Reloads the tripler schematic image (`tripler.png` ↔ `tripler_dark.png`)  
     - Restyles the Energy Monitor plot (axes, ticks, line color)  

6. **Thread‐Safe GUI Updates**  
   - All worker threads schedule GUI updates using `tk.after(0, callback)`  
   - Cancellation signaling via `threading.Event`—`_kill_opt` for optimizer, `_scan_stop` for scans  

---

## Requirements

- **Python 3.9+**  
- **Tkinter** (usually included with standard Python installations)  
- **NumPy**  
- **h5py**  
- **Matplotlib**  
- **SciPy** (for `scipy.optimize.minimize`, `scipy.stats.norm`)  
- **scikit-learn** (for `GaussianProcessRegressor`, `Matern`, `WhiteKernel`, `ConstantKernel`)  
- **tqdm** (for 2D scan progress bar)  
- **Pillow** (for loading and resizing schematic images)  
- **EPICS PV Access** (optional; if unavailable, runs in “test mode” with dummy PVs)  

Install via pip:
```bash
pip install numpy h5py matplotlib scipy scikit-learn tqdm pillow
