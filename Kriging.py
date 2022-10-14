import numpy as np
import pandas as pd
from numpy.linalg import inv

# Inputs
x0 = 28
y0 = 99
nugget = 0
spherical = 1000000
rang = 1000000

# Read Data from files
df_points_xyz = pd.read_csv('Data/Input/1.DataPoints.csv')
no_of_points = df_points_xyz.shape[0]


# 1. Calculation Matrix that relate to the dataset ONLY

# 1.1 Calculation of the distance matrix
np_distance = np.zeros((no_of_points,no_of_points))
for i in range(no_of_points):
    for j in range(no_of_points):
        if i>j:
            dx = df_points_xyz.iloc[i, 0] - df_points_xyz.iloc[j, 0]
            dy = df_points_xyz.iloc[i, 1] - df_points_xyz.iloc[j, 1]
            np_distance[i,j] = (dx**2+dy**2)**0.5
np_distance = np_distance+np_distance.T

# 1.2 Calculation of the Variogram matrix
np_variogram = (1.5*np_distance/rang-0.5*(np_distance/rang)**3)*(np_distance<=rang)
np_variogram += np.ones((no_of_points,no_of_points))*(np_distance>rang)
np_variogram += (np.identity(no_of_points)==0)*nugget

# 1.3 Calculation of the Covariance and its Inverse matrix
np_covariance = nugget + spherical - np_variogram
np_covariance_inv = inv(np_covariance)

# 2 Calculate the Matrix related to the Point we want to know

# 2.1 Calculate the residual values
np_z = df_points_xyz.iloc[:,2].to_numpy()
mean = sum(df_points_xyz.iloc[:,2])/no_of_points
np_residual = np_z - mean
# 2.1 Calculate the distance matrix of point of interest
np_xy = df_points_xyz.iloc[:,0:2].to_numpy()
np_xy0 = np.array([x0,y0])
np_xy0 = np.tile(np_xy0,(no_of_points,1))
np_distance0 = (np_xy-np_xy0)**2
np_distance0 = (np_distance0[:,0]+np_distance0[:,1])**0.5

# 2.1 Calculate the Variogram matrix of point of interest
np_variogram0 = (1.5*np_distance0/rang-0.5*(np_distance0/rang)**3)*(np_distance0<=rang)
np_variogram0 += 1*(np_distance0>rang)
np_variogram0 += nugget

# 2.3 Calculate the Covariance Matrix for point of interest
np_covariance0 = nugget + spherical - np_variogram0

#2.4 Calculate the Weight Matrix for point of interest
np_weight = np.dot(np_covariance_inv,np_covariance0)
sum_weight = sum(np_weight)
mean_weight = nugget + spherical - sum_weight

#2.5 Calculate the final Kriging Estimate and the Corresponding Variance
kriging_estimate = sum(np_residual * np_weight) + mean
kriging_variance = nugget + spherical - sum(np_covariance0 * np_weight)

print(kriging_estimate,kriging_variance)