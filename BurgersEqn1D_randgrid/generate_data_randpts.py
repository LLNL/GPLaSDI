#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

#Set initial training parameters
avals = np.array([0.70, 0.75, 0.85, 0.90])
wvals = np.array([0.90, 1.00, 0.98, 1.10])

param_test = np.vstack((avals,wvals)).T

n_test = param_test.shape[0]

#Generate data (TODO: put this in another file)
U0 = [initial_condition(param_test[i, 0], param_test[i, 1], x_grid) for i in range(n_test)]
X_test = generate_initial_data(U0, time_dim, space_dim, Dt, Dx)
                     

#%%

#Choose which values will be in our initial training data
train_inds = [0,1,2,3]

#Training parameters
param_train = np.array([[avals[x],wvals[x]] for x in train_inds])
n_train = param_train.shape[0]


X_train = X_test[train_inds,:,:]

X_train_init = np.vstack(X_train)
X_test_init = X_test

#%% Choose random sampling of test parameters

# if Grid, then place grid over convex hull of training parameters
# otherwise just place random points

grid = True

if grid:
    n_a_grid = 2**5
    n_w_grid = 2**5
    a_grid = np.linspace(min(avals), max(avals), n_a_grid)
    w_grid = np.linspace(min(wvals), max(wvals), n_w_grid)
    a_grid, w_grid = np.meshgrid(a_grid, w_grid)
    possiblevals = np.hstack((a_grid.reshape(-1, 1), w_grid.reshape(-1, 1)))

param_grid = param_test

fig = plt.figure() 
ax  = fig.add_subplot(111)
hull = ConvexHull(param_grid)
#bounding box
bbox = [hull.min_bound, hull.max_bound]
#Hull path
hull_path = Path( hull.points[hull.vertices] )

plt.plot(param_grid[:,0], param_grid[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(param_grid[simplex, 0], param_grid[simplex, 1], 'k-')

if grid:
    n = len(possiblevals)
    rand_points = np.zeros((1, 2))
    for i in range(n):
        randpt = possiblevals[i,:]          
        if hull_path.contains_point(randpt):
            rand_points = np.vstack((rand_points, randpt))
    rand_points = rand_points[1:,:]
else:
    n = 500
    rand_points = np.empty((n, 2))
    for i in range(n):
        #Draw a random point in the bounding box of the convex hull
        rand_points[i] = np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])])

        #We check if the random point is inside the convex hull, otherwise we draw it again            
        while hull_path.contains_point(rand_points[i]) == False:
            rand_points[i] = np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])])

plt.scatter(param_grid[:, 0],param_grid[:, 1], marker='o',  c='blue', alpha = 1, label ='Initial points')
for simplex in hull.simplices:
        plt.plot(hull.points[simplex, 0], hull.points[simplex, 1], '-k')
plt.gca().add_patch(Rectangle((bbox[0][0], bbox[0][1]), bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1],facecolor = 'None', edgecolor = 'cyan'))
plt.scatter(rand_points[:, 0],rand_points[:, 1], s=3, marker='o',  c='red', alpha = 1, label ='Points inside hull')
plt.legend()
ax.set_xlabel('w')
ax.set_ylabel('a')
# ax. set_aspect(33, adjustable='box')
plt.savefig("grid.png", dpi = 300)

param_grid = np.vstack((param_grid,rand_points))

data_train = {'param_train' : param_train, 'X_train' : X_train, 'n_train' : n_train,'X_train_init' : X_train_init}
os.makedirs(os.path.dirname("./data/data_train.npy"), exist_ok=True)
np.save('data/data_train.npy', data_train)

# X_test = np.empty([1,nt+1,nx])

data_test = {'param_grid' : param_grid, 'X_test' : X_test,'X_test_init' : X_test_init, 'n_test' : n_test}
os.makedirs(os.path.dirname("./data/data_test.npy"), exist_ok=True)
np.save('data/data_test.npy', data_test)

data_paramvals = {'avals' : avals, 'wvals' : wvals}
os.makedirs(os.path.dirname("./data/data_paramvals.npy"), exist_ok=True)
np.save('data/data_paramvals.npy', data_paramvals)





