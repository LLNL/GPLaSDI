#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''

Generates the data for all possible testing points. This allows us assess accuracy of our results.

'''

import numpy as np
from utils import *
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve2d
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.path import Path

#%% 
np.random.seed(0)

def initial_condition(a, w, x_grid):

    return a * np.exp(-x_grid ** 2 / 2 / w / w)

#%% Setting up time and space domains


time_dim = 1001
space_dim = 1001

x_min, x_max = -3, 3
t_max = 1
Dx = (x_max - x_min) / (space_dim - 1)
Dt = t_max / (time_dim - 1)
x_grid = np.linspace(x_min, x_max, space_dim)
t_grid = np.linspace(0, t_max, time_dim)
t_mesh, x_mesh = np.meshgrid(t_grid, x_grid)

#%% 
data_test = np.load('data/data_test.npy', allow_pickle = True).item()
param_test = data_test['param_test']

U0 = [initial_condition(param_test[i, 0], param_test[i, 1], x_grid) for i in range(param_test.shape[0])]
X_sim = generate_initial_data(U0, time_dim, space_dim, Dt, Dx)


data_sim = {'param_test' : param_test, 'X_sim' : X_sim, 'n_sim' : param_test.shape[0]}
os.makedirs(os.path.dirname("./data/data_sim.npy"), exist_ok=True)
np.save('data/data_sim.npy', data_sim)






