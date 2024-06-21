#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''

Modification of 1DBurgers example to make three changes:
    1) The parameter space does not have to be a square grid
    2) We use standard deviation of the SINDy coefficients to adaptively sample,
        instead of variation in the predictions.
    3) We assume there is no  given test data. Instead, we generate data as
        needed during training.

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
#Set initial training parameters
avals = np.array([0.70, 0.725, 0.75, 0.800, 0.85, 0.90])
wvals = np.array([0.90, 0.970, 1.00, 0.925, 0.98, 1.10])

param_train = np.vstack((avals,wvals)).T
n_train = param_train.shape[0]   

#Generate initial training data
U0 = [initial_condition(param_train[i, 0], param_train[i, 1], x_grid) for i in range(n_train)]
X_train = generate_initial_data(U0, time_dim, space_dim, Dt, Dx)

#%% Choose random sampling of test parameters

# if Grid, then place grid over convex hull of training parameters.
# otherwise just place random points

grid = True

if grid:
    n_a_grid = 2**5
    n_w_grid = 2**5
    a_grid = np.linspace(min(avals), max(avals), n_a_grid)
    w_grid = np.linspace(min(wvals), max(wvals), n_w_grid)
    a_grid, w_grid = np.meshgrid(a_grid, w_grid)
    #This array contains all parameter pairs on a square grid. 
    #We will remove values which are outside the convex hull later
    possiblevals = np.hstack((a_grid.reshape(-1, 1), w_grid.reshape(-1, 1)))

fig = plt.figure() 
ax  = fig.add_subplot(111)
hull = ConvexHull(param_train)
#bounding box
bbox = [hull.min_bound, hull.max_bound]
#Hull path
hull_path = Path( hull.points[hull.vertices] )

plt.plot(param_train[:,0], param_train[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(param_train[simplex, 0], param_train[simplex, 1], 'k-')

#Here we remove all parameters which are outside the convex hull
if grid:
    n = len(possiblevals)
    rand_points = np.zeros((1, 2))
    for i in range(n):
        randpt = possiblevals[i,:]
        #check if we are in the convex hull and not on a training point          
        if hull_path.contains_point(randpt) and (not ( randpt == hull.points).all(axis=1).any() ):
        # if hull_path.contains_point(randpt):
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

plt.scatter(param_train[:, 0],param_train[:, 1], marker='o',  c='blue', alpha = 1, label ='Initial points')
for simplex in hull.simplices:
        plt.plot(hull.points[simplex, 0], hull.points[simplex, 1], '-k')
plt.gca().add_patch(Rectangle((bbox[0][0], bbox[0][1]), bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1],facecolor = 'None', edgecolor = 'cyan'))
plt.scatter(rand_points[:, 0],rand_points[:, 1], s=3, marker='o',  c='red', alpha = 1, label ='Points inside hull')
plt.legend()
ax.set_xlabel('a')
ax.set_ylabel('w')
# ax. set_aspect(33, adjustable='box')
plt.savefig("grid.png", dpi = 300)

param_test = rand_points
n_test = param_test.shape[0]

data_train = {'param_train' : param_train, 'X_train' : X_train, 'n_train' : n_train}
os.makedirs(os.path.dirname("./data/data_train.npy"), exist_ok=True)
np.save('data/data_train.npy', data_train)


data_test = {'param_test' : param_test, 'n_test' : n_test}
os.makedirs(os.path.dirname("./data/data_test.npy"), exist_ok=True)
np.save('data/data_test.npy', data_test)




