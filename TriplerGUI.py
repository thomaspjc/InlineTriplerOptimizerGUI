#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:17:02 2025

TRIPLER GUI with Optimizer

Two-tab Tkinter GUI to control 4 EPICS motors (or in testing mode):
  • Tab 1 “Inline Tripler Optimizer”: tripler diagram + big jogs for Motors 1 & 3,
    central Optimize button for delay, and bottom Energy Monitor plot.
  • Tab 2 “Manual Control”: grouped SHG/THG panels for all four motors.
Dark-mode toggle in the toolbar.

The Optimizer is based on Bayesian Optimization and refined through gradient ascent
We use the Expected Improvement Acquisition function in the BO

@author: thomas
"""


import threading
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec
import numpy as np
import tkinter as tk, tkinter.font as tkfont
import queue

from tkinter import filedialog, messagebox
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

from tqdm import tqdm
tqdm.monitor_interval = 0



text_font = None

# ——— Globals —————————————————————————————

# Colors 
STANFORD_BLUE   = "#006CB8"
STANFORD_GREEN  = "#1AECBA"

# Polling
POLL_PERIOD = 0.1  # seconds between PV.get() # needs to be greater than 0.08 for GUI to follow properly


# ——— EPICS FALLBACK / TEST MODE —————————————————————————————
try:
    from epics import PV
    EPICS_AVAILABLE = True
except ImportError:
    EPICS_AVAILABLE = False
    class PV:
        """Dummy PV class: stores values in a class-wide dict and computes dummy energy in test mode"""
        _storage = {}
        def __init__(self, name):
            self.pvname = name
            PV._storage.setdefault(name, 0.0)
        def energy(self, theta1, theta2, pert1=0.1, pert2=0.5):
            """Dummy energy: |sinc(theta1-pert1)*sinc(theta2-pert2)|"""
            return abs(np.sinc(theta1-pert1)*np.sinc(theta2-pert2)) #+ ((np.random.random(1)) * 0.001)
        def get(self):
            # In test mode, compute dummy energy for the energy PV
            if not EPICS_AVAILABLE and self.pvname == "PMTR:LR20:50:PWR":
                # Motor positions are stored under MOTOR:TEST:PV1 and MOTOR:TEST:PV3
                theta1 = PV._storage.get("MIRR:LR20:70:TRP1_MOTR_V", 1)
                theta2 = PV._storage.get("MIRR:LR20:75:TRP2_MOTR_H", 0.0)
                # Compute energy via the dummy energy() function
                return float(self.energy(theta1, theta2))
            
            if not EPICS_AVAILABLE and len(self.pvname.split(".RBV")) == 2:
                return PV._storage.get(self.pvname.split(".RBV")[0])
            # Default: return stored value
            return PV._storage.get(self.pvname, 0.0)
    
        def put(self, val):
            PV._storage[self.pvname] = float(val)
        
        def DMOV(self):
            return True
            
    def set_test_pv(name, val): PV._storage[name] = float(val)
    # preload test PVs
    for i in range(1,5): set_test_pv(f"MOTOR:TEST:PV{i}", i*10.0)
    set_test_pv("PMTR:LR20:50:PWR", 0.0)



# ——— MOTOR MONITOR THREAD —————————————————————————————————
class MotorMonitor:
    """Background thread that polls one PV and fires on_change callback."""
    def __init__(self, pv_name, data_q, poll=POLL_PERIOD):
        self.pv = PV(pv_name)
        self.q = data_q
        self.poll = poll
        self._stop = threading.Event()
        self._thr  = threading.Thread(target=self._loop, daemon=True)
        
    def start(self):
        self._stop.clear()
        self._thr.start()
        
    def stop(self):
        self._stop.set()
        self._thr.join(timeout = 5)
        #print('Motor Thread:', self._thr.is_alive())
        self._thr = None
        
    def _loop(self):
        last = None
        while not self._stop.is_set():
            val = self.pv.get()
            if val != last:                      # only push changes
                last = val
                try:                             # overwrite if queue full
                    self.q.put_nowait(val)
                except queue.Full:
                    self.q.get_nowait()
                    self.q.put_nowait(val)
            time.sleep(self.poll)
        

# ——— ENERGYMETER MONITOR THREAD —————————————————————————————————
class EnergyMonitor:
    """
    Background thread that polls an EPICS PV (or dummy) for energy readings
    and calls your on_update callback whenever the value changes.
    """
    def __init__(self, pv_name, energy_q, poll=POLL_PERIOD):
        self.pv        = PV(pv_name)
        self.q         = energy_q
        self.poll      = poll
        self._stop     = threading.Event()
        self._thr      = threading.Thread(target=self._loop, daemon=True)


    def start(self):
        self._stop.clear()
        self._thr.start()

    def stop(self):
        self._stop.set()
        self._thr.join()
        self._thr = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.q.put(self.pv.get(), block=False)
            except queue.Full:
                print('Energy Queue was full –> data point was skipped')
                pass                               # main thread will catch up
            time.sleep(self.poll)
        
        
            
            
# ——— BO/GA Optimizer —————————————————————————————————

class Optimizer:
    def __init__(self, SHG_pv, THG_pv, energy_pv, poll_period = 0.5):
        """
        Initializes the Optimizer class setting the correct PVs to control

        Parameters
        ----------
        SHG_pv : TYPE
            DESCRIPTION.
        THG_pv : TYPE
            DESCRIPTION.
        energy_pv : TYPE
            DESCRIPTION.
        poll_period : TYPE, optional
            DESCRIPTION. The default is 0.5.

        Returns
        -------
        None.

        """
        self.SHG = SHG_pv
        self.THG = THG_pv
        self.energy = energy_pv
        self.poll = poll_period
        
        self.RBVs = [None]*4
    
    def _read_energy(self, jog1, jog2, timeout = 20, tol = 1e-5):
        """
        Moves the motors to their jog positions then reads the energy

        Parameters
        ----------
        jog1 : TYPE
            DESCRIPTION.
        jog2 : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.SHG.put(jog1)
        
        rbv1 = self.RBVs[0]
        if rbv1 is None:
            name1 = self.SHG.pvname
            name1 = name1 +".RBV"
            rbv1 = PV(name1)
            
            self.RBVs[0] = rbv1

        startRBV = time.time()
        while True:
            if jog1 == rbv1.get():
                time.sleep(POLL_PERIOD/2) # could skip this and wait after the loops
                break
            # Hard Timeout that stops the process
            if time.time() - startRBV > timeout:
                raise RuntimeError(f"Motor move timed out after {timeout}s")
            if time.time() - startRBV > timeout/5:
                self.SHG.put(jog1)
                
                
        self.THG.put(jog2)
        
        rbv2 = self.RBVs[1]
        if rbv2 is None:
            name2 = self.THG.pvname
            name2 = name2 +".RBV"
            rbv2 = PV(name2)
            
            self.RBVs[1] = rbv2

        startRBV = time.time()
        while True:
            if jog2 == rbv2.get():
                time.sleep(POLL_PERIOD/2)
                break
            # Hard Timeout that stops the process
            if time.time() - startRBV > timeout:
                raise RuntimeError(f"Motor move timed out after {timeout}s")
            if time.time() - startRBV > timeout/5:
                self.THG.put(jog2)
        # Could combine the the two loops if the motors can move at the same time 
        
        return float(self.energy.get())
    
    def expected_improvement(self, X, X_sample, Y_sample, gpr, xi=0.01):
        """
        Compute Expected Improvement at candidate points X.
        - X: array of shape (n_points, 2)
        - X_sample, Y_sample: past observations
        - gpr: trained GaussianProcessRegressor
        - xi: exploration‐exploitation trade‐off parameter
        """
        # Predict mean & stddev at X
        mu, sigma = gpr.predict(X, return_std=True)
        # Best observed value so far
        mu_sample_opt = np.max(Y_sample)

        # Compute EI with numerical stability guard
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

 
    def propose_location(self, acq, X_sample, Y_sample, gpr, bounds, n_restarts=25):
        """
        Multi‐start L-BFGS-B to find x_new = argmax_x acq(x).
        - acq: acquisition function
        - bounds: array([[θ1_min, θ1_max],[θ2_min, θ2_max]])
        - n_restarts: how many random seeds to try
        """
        dim = bounds.shape[0]
        best_x = None
        best_acq = -np.inf

        # Try multiple starting points to avoid local maxima
        for start in np.random.uniform(bounds[:,0], bounds[:,1], size=(n_restarts, dim)):
            # minimize negative EI → maximize EI
            res = minimize(
                fun=lambda x: -acq(x.reshape(1, -1), X_sample, Y_sample, gpr),
                x0=start,
                bounds=bounds,
                method='L-BFGS-B'
            )
            if not res.success:
                continue
            acq_val = -res.fun
            if acq_val > best_acq:
                best_acq = acq_val
                best_x = res.x

        # Return in shape (1,2) for consistency
        return best_x.reshape(1, -1)

    
    
    def BayesianOptimize(self, bounds, n_init=30, n_iter=100, stop_event = None):
        """
        Performs 2D Bayesian optimization using a GP + EI.
        - objective: function mapping [θ1,θ2] → scalar to maximize
        - bounds: [[min1,max1],[min2,max2]]
        - n_init: # random samples to initialize GP
        - n_iter: # sequential BO acquisitions
        Returns: best_theta (1×2), best_energy
        """
        
        def objective(x):
            return self._read_energy(x[0], x[1])
        bounds = np.asarray(bounds)

        # 1) Generate initial random samples
        # Sorting the random samples in order to increase acquisition speed
        # The motor precision is set to 4 in epics pvname.precision = 4
        X_sample = np.round(np.sort(np.random.uniform(bounds[:,0], bounds[:,1],size=(n_init, bounds.shape[0])), axis = 0), 6)
        Y_sample = []
        for x in X_sample:
            if stop_event and stop_event.is_set():
                print("Initial random sampling aborted")
                break
            Y_sample.append(objective(x))
    
    
        X_sample = np.array(X_sample)
        Y_sample = np.array(Y_sample)
    
        # if you never got any samples, bail out
        if len(Y_sample)==0:
            return None, None

        # 2) Define GP surrogate with Matérn kernel + white noise
        kernel = (
            C(1.0, (1e-3, 1e3))  # constant amplitude
            * Matern(length_scale=np.ones(bounds.shape[0]), nu=2.5)
            + WhiteKernel(noise_level=1e-6)
        )
        # Disable internal hyperparam optimization to avoid warnings
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            optimizer=None,
            n_restarts_optimizer=0
        )

        # 3) BO acquisition loop
        for i in range(n_iter):
            if stop_event and stop_event.is_set():
                print("BO aborted at iteration", i)
                break
            # Fit GP to all data so far
            gpr.fit(X_sample, Y_sample)

            # Propose next best point by maximizing EI
            x_next = self.propose_location(self.expected_improvement,
                                      X_sample, Y_sample, gpr, bounds)
            # Evaluate the true objective at that point
            y_next = objective(x_next[0])

            # Augment our dataset
            X_sample = np.vstack((X_sample, x_next))
            Y_sample = np.append(Y_sample, y_next)

            print(f"BO Iter {i+1:2d}: θ = {x_next.flatten()}, E = {y_next:.6f}")

        # Return the best observed θ and its energy
        idx_best = np.argmax(Y_sample)
        return X_sample[idx_best], Y_sample[idx_best]
    
    def GradientAscent(self, jog_init, max_iters=100, alpha0=0.1,
                       decay=0.95, eps=1e-3, tol=1e-3, stop_event = None):
        """
        Starting from theta_init, perform decaying‐step finite‐difference
        gradient ascent on energy_func to polish the solution.
        """
        
        def energy_func(t1, t2):
            return self._read_energy(t1, t2)
        
        theta1, theta2 = jog_init
        E_prev = energy_func(theta1, theta2)

        for k in range(max_iters):

            if stop_event and stop_event.is_set():
                print("GA aborted at iteration", k)
                break
            # Exponentially decaying step size
            alpha = alpha0 * (decay ** k)

            # Estimate ∂E/∂θ1 via central difference
            grad1 = (
                energy_func(theta1 + eps, theta2)
                - energy_func(theta1 - eps, theta2)
            ) / (2 * eps)
            theta1 += alpha * grad1

            # Estimate ∂E/∂θ2 via central difference
            grad2 = (
                energy_func(theta1, theta2 + eps)
                - energy_func(theta1, theta2 - eps)
            ) / (2 * eps)
            theta2 += alpha * grad2

            # Check for convergence in E
            E_new = energy_func(theta1, theta2)
            if abs(E_new - E_prev) < tol:
                print(f"Refinement converged at iter {k+1}")
                break
            E_prev = E_new

        # Return the polished angles and final energy
        return np.array([theta1, theta2]), E_prev
    

# ——— MAIN APPLICATION —————————————————————————————————————————
class App(tk.Tk):
    def __init__(self):
        """
        Initialize the TRIPLER GUI

        Returns
        -------
        None.

        """
        super().__init__()
        
        # ––– Setting up the closing protocol –––––––––––––––––––––––––––––––––
        self._closing = False
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # ––– Setting up the GUI decor ––––––––––––––––––––––––––––––––––––––––
        self._load_assets()
        self.title(f"Motor Control{' (TEST MODE)' if not EPICS_AVAILABLE else ''}")
        # dark mode flag
        self.dark_mode = False
        # Tracking a 1D vs 2D scan 
        self.scan_mode = tk.StringVar(value="1D")
        
        # ––– Setting up the motors through epics –––––––––––––––––––––––––––––
        # state for 4 motors
        self.monitors     = [None]*4
        self.currents     = [0.0]*4
        # one single-slot queue per motor
        self._motor_qs = [queue.Queue(maxsize=1) for _ in range(4)]
        # widget refs
        self.pv_entries   = [None]*4
        self.RBVs         = [None]*4
        
        self.cur_labels   = [0.0]*4
        self.pv_names     = ["MIRR:LR20:70:TRP1_MOTR_V",
                             "MIRR:LR20:70:TRP1_MOTR_H",
                             "MIRR:LR20:75:TRP2_MOTR_H",
                             "MIRR:LR20:75:TRP2_MOTR_V"
                             ]
        self.step_entries = [0.0001]*4  
        self.after(0, self._drain_motor_qs)
        
        # ––– Setting up the Energy Monitor through Epics –––––––––––––––––––––
        self.ENERGY_PV = "PMTR:LR20:50:PWR"  # Power Meter PV
        self.energy_times  = []
        self.energy_values = []
        self.energy_index = 0
        # instantiate and start the monitor
        self._energy_q = queue.Queue(maxsize = 1)
        self.energy_monitor = EnergyMonitor(self.ENERGY_PV, energy_q=self._energy_q)
        self.after(0, self.energy_monitor.start)
        self._plot_pending = False
        self.after(0, self._drain_q)
        
        # event to tell optimizer/GA loops to quit early
        # FIXME: Should the following be in another method ?
        self._kill_opt = threading.Event()

        # ––– Build the GUI and start the program –––––––––––––––––––––––––––––
        self._build_ui()

    def _drain_q(self):
        if self._closing:
            return                            # we’re shutting down – stop rescheduling
        try:
            val = self._energy_q.get_nowait()
            self.energy_index += 1
            self.energy_times.append(self.energy_index)
            self.energy_values.append(val)
            if len(self.energy_values) > 100:
                self.energy_values.pop(0); self.energy_times.pop(0)
            if not self._plot_pending:        # debounce – at most one draw in queue
                self._plot_pending = True
                self.after_idle(self._update_energy_plot)
        except queue.Empty:
            #print("Queue was empty")
            pass
            
        finally:
            # Re-arm ourselves ~10 ms later
            self.after(int(POLL_PERIOD*1000), self._drain_q)
            
    def _drain_motor_qs(self):
        if self._closing:
            return                      # stop rescheduling during shutdown
    
        for idx, q in enumerate(self._motor_qs):
            try:
                val = q.get_nowait()
            except queue.Empty:
                continue
            self.currents[idx] = val
            self.cur_labels[idx].config(text=f"{val:.4g}")
    
        self.after(int(POLL_PERIOD*1000), self._drain_motor_qs)

    def _load_assets(self):
        """
        Prepare the images to be used in the GUI 

        Returns
        -------
        None.

        """
        # load the full tripler schematic
        img = Image.open("tripler.png")
        img = img.resize((1000, 200), resample=Image.LANCZOS)
        self.trip_img = ImageTk.PhotoImage(img)

    def _build_ui(self):
        """
        Build the GUI canvas

        Returns
        -------
        None.

        """
        # — toolbar with Dark-Mode toggle —
        toolbar = tk.Frame(self, pady=5)
        toolbar.pack(fill=tk.X)

        self.dark_btn = tk.Button(
            toolbar,
            text="Enable Dark Mode",
            command=self._toggle_dark
        )
        self.dark_btn.pack(side=tk.RIGHT, padx=10)
        

        # — notebook with two tabs —
        self.nb = ttk.Notebook(self)

        self.nb.bind("<Configure>", self._center_tabs)
        self.tab1 = tk.Frame(self.nb); self.tab2 = tk.Frame(self.nb)
        self.nb.add(self.tab1, text="Inline Tripler Optimizer")
        self.nb.add(self.tab2, text="Manual Control")
        self.nb.pack(fill=tk.BOTH, expand=True)

        self._build_optimize_tab(self.tab1)

        self._build_manual_tab(self.tab2)


    # ——— Tab 1: Inline Tripler Optimizer —————————————————————
    def _build_optimize_tab(self, parent):
        """
        Build the Optimizer tab (tab1)

        Parameters
        ----------
        parent : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """        
        # top: the tripler schematic
        lbl = tk.Label(parent, image=self.trip_img, bd=2, relief=tk.SUNKEN)
        lbl.grid(row=0, column=0, columnspan=3, pady=10, padx=10)
        parent.grid_columnconfigure((0,1,2), weight=1)

        # Motor 1 (SHG crystal) under left crystal
        self._make_optimizer_panel(parent, motor_idx=0, col=0, title="SHG Phase Matching")

        # Control center panel
        df = tk.LabelFrame(parent, text="Controls", padx=10, pady=10)
        df.grid(row=1, column=1, padx=20, pady=5, sticky="nsew")
        self.optimize_btn = tk.Button(df, text="OPTIMIZE", width=12, height=2,
                        command=self._on_optimize)
        self.optimize_btn.pack()#expand=True
        self.stop_btn = tk.Button(df, text="STOP", width=12, height=2,
                                  state=tk.DISABLED,
                                  command=self._stop_optimize)
        self.stop_btn.pack(pady=(5,0))
        
        # Motor 3 (THG crystal) under right crystal
        self._make_optimizer_panel(parent, motor_idx=2, col=2, title="THG Phase Matching")
        
        # bottom: Energy Monitor plot
        ef = tk.LabelFrame(parent, text="Energy Monitor", padx=5, pady=5)
        ef.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)
        parent.grid_rowconfigure(2, weight=1)

        self.fig = Figure(figsize=(6,2.5), dpi=100)
        outer = gridspec.GridSpec(2,1,
                               height_ratios = [1, 0.2],
                               width_ratios  = [1],
                               hspace = 0.1, wspace = 0.1)
        self.axE = self.fig.add_subplot(outer[0,:])
        self.xlabelE = self.axE.set_xlabel("Time")
        self.ylabelE = self.axE.set_ylabel("Energy")
        self.line_energy, = self.axE.plot([], [], marker='x', color = STANFORD_BLUE)  # placeholder
        self._energy_canvas = FigureCanvasTkAgg(self.fig, master=ef)
        self._energy_canvas.draw()
        self._energy_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self._toggle_dark()

    def _make_optimizer_panel(self, parent, motor_idx, col, title):
        """Create the SHG/THG crystal jog panels on Tab 1."""
        f = tk.LabelFrame(parent, text=title, padx=80, pady=8)
        f.grid(row=1, column=col, padx=20, pady=5, sticky="nsew")

        # PV + Connect
        tk.Label(f, text="Currently:").grid(row=0, column=0, sticky="e")
        cl = tk.Label(f, text="0.00", width=7, relief=tk.SUNKEN)
        cl.grid(row=0, column=1, columnspan=2, sticky="w", padx=1)
        self.cur_labels[motor_idx] = cl
        '''pv = tk.Entry(f, width=16)
        pv.grid(row=0, column=1, padx=5, sticky="w")'''
        
        #self.pv_entries[motor_idx] = pv

        # Jog buttons
        bf = tk.Frame(f)
        bf.grid(row=2, column=0, columnspan=3, pady=8)
        tk.Button(bf, text="<", font=(text_font,18),
                  width=4, command=lambda i=motor_idx: self._jog(i, -1)
                 ).pack(side=tk.LEFT, padx=10)
        tk.Button(bf, text=">", font=(text_font,18),
                  width=4, command=lambda i=motor_idx: self._jog(i, +1)
                 ).pack(side=tk.LEFT, padx=10)

    def _on_optimize(self):
        """
        Start the BO/GA optimizer

        Returns
        -------
        None.

        """
        
        # Intilize the optimizer
        print("Running optimizer")
        # clear any previous kill-flag
        self._kill_opt.clear()
        self.stop_btn.config(state=tk.NORMAL)
        self.optimize_btn.config(state=tk.DISABLED)
        
        self.optimizer = Optimizer(
            SHG_pv=PV(self.pv_entries[0].get()),
            THG_pv=PV(self.pv_entries[2].get()),
            energy_pv =PV(self.ENERGY_PV),
            poll_period=POLL_PERIOD
        )
        # run it in a background thread so the GUI doesn’t freeze
        self._opt_stop = threading.Event()
        self._opt_thr  = threading.Thread(target=self._run_opt, daemon=True)
        self._opt_thr.start()
        # you could read self.currents[...] and call mon.pv.put(...) here
        
    def _stop_optimize(self):
        self._kill_opt.set()
        self.stop_btn.config(state=tk.DISABLED)
        self.optimize_btn.config(state=tk.NORMAL)

        if self._opt_thr is not None: #and self._opt_thr.is_alive():
            self._opt_stop.set()
            self._opt_thr.join(timeout = 5)
            self._opt_thr = None


        
    def _run_opt(self):
        """
        Run the Optimizer
        
        Start by running a Bayesian Optimizer 
        Then run a Gradient Ascent to determine the local maximum 
        
        This hybrid method allows to determine the global maximum quickly

        Returns
        -------
        None.

        """
        # Define the search region for θ₁ and θ₂
        bounds = [[-0.3, 0.3], [-0.3, 0.3]]

        # (a) Global search via Bayesian optimization
        best_theta_bayes, best_e_bayes = self.optimizer.BayesianOptimize(bounds=bounds,
                                                          n_init=150,    # number of initial random points
                                                          n_iter=100,     # number of BO iterations
                                                          stop_event=self._kill_opt
                                                          )
        print(f"\nBayesian Opt result: θ = {best_theta_bayes}, E = {best_e_bayes:.6f}")
        if self._kill_opt.is_set():
            return
        # (b) Local polish via gradient ascent
        best_theta_refined, best_e_refined = self.optimizer.GradientAscent(jog_init=best_theta_bayes,
                                                                          max_iters=50,
                                                                          alpha0=0.1,
                                                                          decay=0.95,
                                                                          eps=1e-3,
                                                                          tol=1e-5,
                                                                          stop_event=self._kill_opt
                                                                          )
        print(f"Refined result: θ = {best_theta_refined}, E = {best_e_refined:.6f}")
        
        # once done, re-enable Optimize button
        self.after(0, self._stop_optimize)


    # ——— Tab 2: Manual Control —————————————————————————————
    def _build_manual_tab(self, parent):
        """
        Build the tab for manual control of the motors

        Parameters
        ----------
        parent : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # SHG label

        tk.Label(parent, text="SHG", font=(text_font,14)).grid( 
            row=0, column=0, columnspan=2, pady=(10,0)
        )
        # two SHG panels (motor 1 & 2)

        self._make_manual_panel(parent, motor_idx=0, row=1, col=0)
        
        self._make_manual_panel(parent, motor_idx=1, row=1, col=1)
        
        # THG label
        tk.Label(parent, text="THG", font=(text_font,14)).grid(
            row=2, column=0, columnspan=2, pady=(20,0)
        )

        # two THG panels (motor 3 & 4)
        self._make_manual_panel(parent, motor_idx=2, row=3, col=0)
        self._make_manual_panel(parent, motor_idx=3, row=3, col=1)

        for r in (1,3):
            parent.grid_rowconfigure(r, weight=1)
        for c in (0,1):
            parent.grid_columnconfigure(c, weight=1)
        
        
        # — scan UI —
        scan_f = tk.LabelFrame(parent, text="Motor Scan", padx=10, pady=10)
        scan_f.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=(20,10))
        
        '''# 0) ADD “Scan Mode” dropdown
        tk.Label(scan_f, text="Mode:").grid(row=0, column=0, sticky="e", padx=(15,0))
        # Choices: “1D” or “2D”
        mode_menu = tk.OptionMenu(scan_f, self.scan_mode, "1D", "2D")
        self.scan_mode.trace_add("write", self._on_scan_mode_change)
        mode_menu.config(width=6)  # optional: set a fixed width
        mode_menu.grid(row=0, column=1, sticky="w", padx=(2,10))
        # END “Scan Mode” dropdown'''
        self.scan_mode.set("2D")
        tk.Label(scan_f, text="Mode:").grid(row=0, column=0, sticky="e", padx=(15,0))
        self.mode_btn = tk.Button(
            scan_f,
            text="2D",
            width=6,
            command=self._toggle_scan_mode
        )
        self.mode_btn.grid(row=0, column=1, sticky="w")
        
        
        # Motor index
        tk.Label(scan_f, text="Motor # (1–4):").grid(row=1, column=0, sticky="e", padx=(15,0), pady=(5,0))
        self.scan_motor1_idx = tk.Entry(scan_f, width=4)
        self.scan_motor1_idx.insert(0, "1")
        self.scan_motor1_idx.grid(row=1, column=1, sticky="w", pady=(5,0)) 
        
        # Start/Stop
        tk.Label(scan_f, text="Start:").grid(row=1, column=2, sticky="e", padx=(15,0), pady=(5,0))
        self.scan_start1 = tk.Entry(scan_f, width=8); self.scan_start1.insert(0, "0.0")
        self.scan_start1.grid(row=1, column=3, sticky="w", pady=(5,0))
        
        tk.Label(scan_f, text="Stop:").grid(row=1, column=4, sticky="e", padx=(15,0), pady=(5,0))
        self.scan_stop1 = tk.Entry(scan_f, width=8); self.scan_stop1.insert(0, "1.0")
        self.scan_stop1.grid(row=1, column=5, sticky="w", pady=(5,0))
        
        
        
        # Step and # acquisitions
        tk.Label(scan_f, text="Step:").grid(row=1, column=6, sticky="e", padx=(15,0), pady=(5,0))
        self.scan_step1 = tk.Entry(scan_f, width=8); self.scan_step1.insert(0, "0.1")
        self.scan_step1.grid(row=1, column=7, sticky="w", pady=(5,0))
        
        tk.Label(scan_f, text="# Acqs each:").grid(row=1, column=8, sticky="e", padx=(15,0), pady=(5,0))
        self.scan_nacq = tk.Entry(scan_f, width=6); self.scan_nacq.insert(0, "5")
        self.scan_nacq.grid(row=1, column=9, sticky="w", pady=(5,0))
        

        # Motor index
        tk.Label(scan_f, text="Motor # (1–4):").grid(row=2, column=0, sticky="e", padx=(15,0), pady=(5,0))
        self.scan_motor2_idx = tk.Entry(scan_f, width=4)
        self.scan_motor2_idx.insert(0, "3")
        self.scan_motor2_idx.grid(row=2, column=1, sticky="w", pady=(5,0)) 
        
        # Start/Stop
        tk.Label(scan_f, text="Start:").grid(row=2, column=2, sticky="e", padx=(15,0), pady=(5,0))
        self.scan_start2 = tk.Entry(scan_f, width=8); self.scan_start2.insert(0, "0.0")
        self.scan_start2.grid(row=2, column=3, sticky="w", pady=(5,0))
        
        tk.Label(scan_f, text="Stop:").grid(row=2, column=4, sticky="e", padx=(15,0), pady=(5,0))
        self.scan_stop2 = tk.Entry(scan_f, width=8); self.scan_stop2.insert(0, "1.0")
        self.scan_stop2.grid(row=2, column=5, sticky="w", pady=(5,0))
        # Step and # acquisitions
        tk.Label(scan_f, text="Step:").grid(row=2, column=6, sticky="e", padx=(15,0), pady=(5,0))
        self.scan_step2 = tk.Entry(scan_f, width=8); self.scan_step2.insert(0, "0.1")
        self.scan_step2.grid(row=2, column=7, sticky="w", pady=(5,0))
        
        
        # “Save To…” button & filename label
        
        # Step and # acquisitions
        tk.Label(scan_f, text="Save to:").grid(row=5, column=0, sticky="e", padx=(15,0), pady=(5,0))
        self.scan_fname_lbl = tk.Entry(scan_f, width=42); self.scan_fname_lbl.insert(0, "No file chosen")
        self.scan_fname_lbl.grid(row=5, column=1, columnspan=6, sticky="w", pady=(5,0))
        
        
        # “Start Scan” button
        self.scan_btn = tk.Button(scan_f, text="Start Scan", command=self._on_start_scan)
        self.scan_btn.grid(row=6, column=0,  pady=(10,0), padx=(15,0))
        self.stop_scan_btn = tk.Button(scan_f, text="Stop Scan", command=self._on_stop_scan)
        self.stop_scan_btn.config(state=tk.DISABLED)
        self.stop_scan_btn.grid(row=6, column=1,  pady=(10,0), padx=(15,0))
        
    
        
    def _toggle_scan_mode(self):
        # 1) flip the scan_mode Var and update the button text
        if self.scan_mode.get() == "1D":
            self.scan_mode.set("2D")
            self.mode_btn.config(text="2D")
            new_state = "normal"
        else:
            self.scan_mode.set("1D")
            self.mode_btn.config(text="1D")
            new_state = "disabled"
    
        # 2) now enable/disable row-2 entries all at once
        for w in (self.scan_motor2_idx,
                  self.scan_start2,
                  self.scan_stop2,
                  self.scan_step2):
            w.config(state=new_state)

    def _make_manual_panel(self, parent, motor_idx, row, col):
        """Small panel with PV/connect, readback, step + jog ◀/▶."""

        f = tk.LabelFrame(parent, text=f"Motor {motor_idx+1}", padx=6, pady=6)

        f.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

        # PV + Connect
        tk.Label(f, text="PV:").grid(row=0, column=0, sticky="e")
        pv = tk.Entry(f, width=14)
        pv.insert(0, self.pv_names[motor_idx])
        pv.grid(row=0, column=1, padx=5, sticky="w")
        tk.Button(f, text="Connect",
                  command=lambda i=motor_idx: self._connect(i)
                 ).grid(row=0, column=2, padx=5)

        self.pv_entries[motor_idx] = pv

        # Current readout
        tk.Label(f, text="Cur:").grid(row=1, column=0, sticky="e")
        cl = tk.Label(f, text="0.00", width=7, relief=tk.SUNKEN)
        cl.grid(row=1, column=1, columnspan=2, sticky="w")

        self.cur_labels[motor_idx] = cl

        # Step size
        tk.Label(f, text="Step:").grid(row=2, column=0, sticky="e")
        st = tk.Entry(f, width=6); st.insert(0, "0.0001")
        st.grid(row=2, column=1, sticky="w")

        self.step_entries[motor_idx] = st

        # Jog buttons

        bf = tk.Frame(f)
        bf.grid(row=3, column=0, columnspan=3, pady=6)

        tk.Button(bf, text="<", width=2,
                  command=lambda i=motor_idx: self._jog(i, -1)
                 ).pack(side=tk.LEFT, padx=4)

        tk.Button(bf, text=">", width=2,
                  command=lambda i=motor_idx: self._jog(i, +1)
                 ).pack(side=tk.LEFT, padx=4)
            
        
    def _on_stop_scan(self):
        self.stop_scan_btn.config(state=tk.DISABLED)
        
        if hasattr(self, "_scan_stop"):
            self._scan_stop.set()

    
    def _on_start_scan(self):
        # 1) Parse & validate
        self.scan_fname = self.scan_fname_lbl.get()
        
        try:
            idx   = int(self.scan_motor1_idx.get()) - 1
            start = float(self.scan_start1.get())
            stop  = float(self.scan_stop1.get())
            step  = float(self.scan_step1.get())
            nacq  = int(self.scan_nacq.get())
            fname = str(getattr(self, "scan_fname", None))
            if not (0 <= idx < 4) or step <= 0 or nacq < 1 or not fname:
                raise ValueError
            if len(fname.split('.h5')) != 2:
                raise ValueError('The file name should have the h5 extension')
            args = [idx, start, stop, step, nacq, fname]
            if self.scan_mode.get() == '2D':
                idx   = int(self.scan_motor2_idx.get()) - 1
                start = float(self.scan_start2.get())
                stop  = float(self.scan_stop2.get())
                step  = float(self.scan_step2.get())
                if not (0 <= idx < 4) or step <= 0 or nacq < 1 or not fname:
                    raise ValueError
                args += [idx, start, stop, step]

        except ValueError:
            messagebox.showerror("Invalid scan parameters",
                                 "Please check motor #, start/stop/step, #acqs, and file.\n \
                                 The file extension should be h5")
            return
        
        # 2) Disable the button & start scan thread
        self.scan_btn.config(state=tk.DISABLED)
        self.stop_scan_btn.config(state=tk.NORMAL)
        self._scan_stop = threading.Event()
        self._scan_thr = threading.Thread(
            target=self._run_scan,
            args=args,
            daemon=True
        )
        self._scan_thr.start()
        

        
        
    def _run_scan(self, motor_idx1, start1, stop1, step1, nacq, fname,
                  motor_idx2 = None, start2 = None, stop2 = None, step2 = None, timeout = 15):
        # Save to the scan folder
        fname = "scans/" + fname
        if self.scan_mode.get() == '1D':
            # build positions array
            positions = np.arange(start1, stop1 + step1, step1)
            
            # open HDF5 file
            with h5py.File(fname, "w") as f:
                ds_p = f.create_dataset("positions", data=positions)
                ds_e = f.create_dataset("energies",
                                        shape=(len(positions), nacq),
                                        dtype="f8")
        
                for i in tqdm(range(len(positions)), ascii = True, desc = "Scanning Motors"):
                    pos = positions[i]
                    if self._scan_stop.is_set():
                        break
                    # ensure motor monitor exists
                    mon = self.monitors[motor_idx1]
                    if mon is None:
                        name = self.pv_entries[motor_idx1].get().strip()
                        mon = MotorMonitor(name, data_q = self._motor_qs[motor_idx1])
                        mon.start()
                        self.monitors[motor_idx1] = mon
                        
                    rbv = self.RBVs[motor_idx1]
                    if rbv is None:
                        name = self.pv_entries[motor_idx1].get().strip()
                        name = name +".RBV"
                        rbv = PV(name)
                        
                        self.RBVs[motor_idx1] = rbv
        
                    # move motor & settle
                    mon.pv.put(pos)
                    startRBV = time.time()
                    while True:
                        if pos == rbv.get():
                            time.sleep(POLL_PERIOD)
                            break
                        # Hard Timeout that stops the process
                        if time.time() - startRBV > timeout:
                            raise RuntimeError(f"Motor move timed out after {timeout}s")
                        if time.time() - startRBV > timeout/5:
                            mon.pv.put(pos)
                    
        
                    # take multiple acquisitions
                    for j in range(nacq):
                        if self._scan_stop.is_set():
                            break
                        e = float(PV(self.ENERGY_PV).get())
                        ds_e[i, j] = e
                        time.sleep(POLL_PERIOD)
        
                    # flush after each row
                    f.flush()
        else: 
            # build positions array
            positions1 = np.arange(start1, stop1 + step1, step1)
            positions2 = np.arange(start2, stop2 + step2, step2)
            motor1, motor2 = np.meshgrid(positions1, positions2)
            motor1[1::2] = motor1[1::2, ::-1]
            # open HDF5 file
            with h5py.File(fname, "w") as f:
                ds_p1 = f.create_dataset("motor1", data=motor1)
                ds_p2 = f.create_dataset("motor2", data=motor2)
                ds_e = f.create_dataset("energies",
                                        shape=(motor1.shape[0], motor1.shape[1], nacq),
                                        dtype="f8")

                positions1[:] = positions1[::-1]
                for i in tqdm(range(len(positions2)), ascii = True, desc = "Scanning Motors"):
                    if self._scan_stop.is_set():
                        break
                    pos2 = positions2[i]

                    # ensure motor monitor exists
                    mon2 = self.monitors[motor_idx2]

                    if mon2 is None:
                        name = self.pv_entries[motor_idx2].get().strip()
                        mon2 = MotorMonitor(name, data_q = self._motor_qs[motor_idx2])
                        mon2.start()
                        self.monitors[motor_idx2] = mon2
                        
                    rbv2 = self.RBVs[motor_idx2]
                    if rbv2 is None:
                        name2 = self.pv_entries[motor_idx2].get().strip()
                        name2 = name2 +".RBV"
                        rbv2 = PV(name2)
                        
                        self.RBVs[motor_idx2] = rbv2
                    # move motor & settle
                    mon2.pv.put(pos2)
                    positions1[:] = positions1[::-1]
                    startRBV = time.time()
                    while True:
                        if pos2 == rbv2.get():
                            time.sleep(POLL_PERIOD)
                            break
                        # Hard Timeout that stops the process
                        if time.time() - startRBV > timeout:
                            raise RuntimeError(f"Motor move timed out after {timeout}s")
                        # Soft Timeout that puts the motor position back 
                        if time.time() - startRBV > timeout/5:
                            mon2.pv.put(pos2)
                    


                    
                    for j, pos1 in enumerate(positions1):
                        if self._scan_stop.is_set():
                            break
                        # ensure motor monitor exists
                        mon1 = self.monitors[motor_idx1]
                        if mon1 is None:
                            name = self.pv_entries[motor_idx1].get().strip()
                            mon1 = MotorMonitor(name, data_q = self._motor_qs[motor_idx1])
                            mon1.start()
                            self.monitors[motor_idx1] = mon1
                        # move motor & settle
                        mon1.pv.put(pos1)
                        
                        rbv1 = self.RBVs[motor_idx1]
                        if rbv1 is None:
                            name1 = self.pv_entries[motor_idx1].get().strip()
                            name1 = name1 +".RBV"
                            rbv1 = PV(name1)
                            
                            self.RBVs[motor_idx1] = rbv1
                            
                        while True:
                            if pos1 == rbv1.get():
                                time.sleep(POLL_PERIOD)
                                break
                            # Hard Timeout that stops the process
                            if time.time() - startRBV > timeout:
                                raise RuntimeError(f"Motor move timed out after {timeout}s")
                            # Soft Timeout that puts the motor position back 
                            if time.time() - startRBV > timeout/5:
                                mon1.pv.put(pos1)
            
                        # take multiple acquisitions
                        for k in range(nacq):
                            if self._scan_stop.is_set():
                                break
                            e = float(PV(self.ENERGY_PV).get())
                            ds_e[i, j, k] = e
                            time.sleep(POLL_PERIOD)
            
                        # flush after each row
                        f.flush()
                f.close()


        # re-enable button & alert user on main thread
        '''self.after(0, lambda: self.scan_btn.config(state=tk.NORMAL))
        self.after(0, lambda: self.stop_scan_btn.config(state=tk.DISABLED))'''
        self.after(0, lambda: print("Scan complete", f"Data saved to:\n{fname}"))
        self.after(0, self._stop_scan)
        
    def _stop_scan(self):
        self.scan_btn.config(state=tk.NORMAL)
        self.stop_scan_btn.config(state=tk.DISABLED)
        self._scan_stop.set()
        self._scan_thr.join()
        self._scan_thr = None
        

    def _update_energy_plot(self):
        """Redraw the energy plot in Tab 1."""
        try:
            
            if self.winfo_exists():
                self.axE.cla()
                clr = STANFORD_BLUE if not self.dark_mode else STANFORD_GREEN
                self.line_energy, = self.axE.plot(self.energy_times, self.energy_values, '-x', color = clr)
                #self.line_energy.set_data(self.energy_times, self.energy_values)
                self.axE.set_xlabel("Time")
                self.axE.set_ylabel("Sample #")
                # make sure the plot uses the current dark/light colors
                
                if self.dark_mode:
                    self.axE.set_facecolor("#000000")
                    for spine in self.axE.spines.values():
                        spine.set_color("white")
                    self.axE.xaxis.label.set_color("white")
                    self.axE.yaxis.label.set_color("white")
                    self.axE.tick_params(colors="white")
                    self.line_energy.set_color(STANFORD_GREEN)
                else:
                    self.axE.set_facecolor("white")
                    for spine in self.axE.spines.values():
                        spine.set_color("black")
                    self.axE.xaxis.label.set_color("black")
                    self.axE.yaxis.label.set_color("black")
                    self.axE.tick_params(colors="black")
                    self.line_energy.set_color(STANFORD_BLUE)
                self._energy_canvas.draw_idle()
                
            
        except tk.TclError:
            #print("tk error worked")
            pass
        self._plot_pending = False
        
        
            

    # ——— COMMON ACTIONS —————————————————————————————————————
    def _connect(self, idx):
        """Start (or restart) polling for motor idx."""
        name = self.pv_entries[idx].get().strip()
        if not name:
            return
        if self.monitors[idx]:
            self.monitors[idx].stop()
        mon = MotorMonitor(name, data_q = self._motor_qs[idx])
        self.monitors[idx] = mon
        mon.start()


    def _jog(self, idx, direction):
        """Move motor idx by ±step."""
        try:
            step = float(self.step_entries[idx].get())
        except ValueError:
            return
        base = self.currents[idx]
        new  = base + direction*step
        mon = self.monitors[idx]
        if mon:
            mon.pv.put(new)


    def _toggle_dark(self):
        """Flip light/dark, recolor all widgets, restyle tabs & center them."""
        # 1) Flip flag & pick palette
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            fg, bg, btn_bg, btn_fg, data_color = 'white', '#000000', 'darkgrey', '#000000', STANFORD_GREEN
        else:
            fg, bg, btn_bg, btn_fg, data_color = 'black', 'white', 'lightgrey', 'black', STANFORD_BLUE


        # Change the image 
        if self.dark_mode:
            img = Image.open("tripler_dark.png")
            img = img.resize((1000, 200), resample=Image.LANCZOS)
            self.trip_img = ImageTk.PhotoImage(img)
        else:
            img = Image.open("tripler.png")
            img = img.resize((1000, 200), resample=Image.LANCZOS)
            self.trip_img = ImageTk.PhotoImage(img)
        lbl = tk.Label(self.tab1, image=self.trip_img, bd=2, relief=tk.SUNKEN)
        lbl.grid(row=0, column=0, columnspan=3, pady=10, padx=10)
        # 2) Root bg
        self.configure(bg=bg)
        
        # 3) Recolor the Energy plot 
        self.fig.patch.set_facecolor(bg)
        self.axE.set_facecolor(bg)
        self.axE.set_facecolor(bg)
        for spine in self.axE.spines.values():
            spine.set_color(fg)
        
        self.line_energy.set_color(data_color)
        self.axE.xaxis.set_tick_params(color=fg, labelcolor=fg)
        self.axE.yaxis.set_tick_params(color=fg, labelcolor=fg)
        self.axE.xaxis.label.set_color(fg)
        self.axE.yaxis.label.set_color(fg)
        self._energy_canvas.draw_idle()
        # 3) Recolor all plain-Tk widgets
        self._recolor(self, bg, fg, btn_bg, btn_fg)

        # 4) Switch to a theme that honors ttk styling
        style = ttk.Style()
        style.theme_use('default')
        
        # 5) Restyle the Notebook & its tabs
        style.configure('TNotebook',
                        background=bg,
                        borderwidth=0)

        style.configure('TNotebook.Tab',
                        background=btn_bg,
                        foreground=btn_fg,
                        padding=[10, 2])
        style.map('TNotebook.Tab',
                  background=[('selected', bg)],
                  foreground=[('selected', fg)])
        # Also recolor the page frames and labelframes
        style.configure('TFrame', background=bg)
        style.configure('TLabelframe', background=bg, foreground=fg)
        style.configure('TLabelframe.Label', background=bg, foreground=fg)


        # 6) Center the tabs within the notebook panel
        self._center_tabs()
        
        # 7) Toggle button itself
        self.dark_btn.config(
            text="Disable Dark Mode" if self.dark_mode else "Enable Dark Mode",
            bg=btn_bg, fg=btn_fg,
            activebackground=bg, activeforeground=btn_fg
        )

        
        


    def _center_tabs(self, event=None):
        """Compute total tab width and set tabmargins so tabs are centered."""
        nb = self.nb
        nb.update_idletasks()
        count = nb.index("end")
        if count <= 0:
            return

        # Sum widths of each tab
        total_w = sum(nb.bbox(i)[2] for i in range(count))
        avail   = nb.winfo_width()
        left    = max((avail - total_w)//2, 0)

        # Apply as left‐margin inside the Notebook
        style = ttk.Style()
        style.configure('TNotebook', tabmargins=[left-150, 0, 0, 0])


    def _recolor(self, widget, bg, fg, btn_bg, btn_fg):
        """Recursively apply bg/fg only where supported."""
        cls = widget.__class__.__name__
        try:
            if cls == "Frame":
                widget.config(bg=bg)
            elif cls == "LabelFrame":
                widget.config(bg=bg, fg=fg)
            elif cls == "Label":
                widget.config(bg=bg, fg=fg)
            elif cls == "Button":
                widget.config(bg=btn_bg, fg=btn_fg,
                              activebackground=bg, activeforeground=btn_fg)
            elif cls == "Entry":
                widget.config(bg=bg, fg=fg, insertbackground=fg)
            elif cls == "Canvas":
                widget.config(bg=bg)
            else:
                widget.config(bg=bg)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._recolor(child, bg, fg, btn_bg, btn_fg)


    
            
            
    def _on_close(self):
        print('Closing App')
        self._closing = True
        # stop all monitors cleanly
        for m in self.monitors:
            if m: m.stop()
                
        print('  Motor Monitors killed...')
        if self.energy_monitor:
                self.energy_monitor.stop()
        print('  Energy Monitor killed...')
        
        if getattr(self, '_scan_thr', None) and self._scan_thr.is_alive():
            self.stop_scan()

        print('  Scan Thread killed...')
        if getattr(self, '_opt_thr', None) and self._opt_thr.is_alive():
            self._stop_optimize()
        print('  Optimizer Thread killed...')
        
        print('Exiting')
        for t in threading.enumerate():
            print(t.name, t.daemon, t.is_alive())
        time.sleep(0.2)
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()

