import numpy as np
import pandas as pd
from numpy.linalg import inv

class Kriging():

    def __init__(self,xyz_file_path):

        # 1. Calculation Matrix that relate to the dataset ONLY
        self.nugget = 0
        self.spherical = 10000
        self.rang = 10000

        # Read Data from files
        self.df_points_xyz = pd.read_csv(xyz_file_path)
        self.no_of_points = self.df_points_xyz.shape[0]

        # 1.1 Calculation of the distance matrix
        self.np_distance = np.zeros((self.no_of_points,self.no_of_points))
        for i in range(self.no_of_points):
            for j in range(self.no_of_points):
                if i>j:
                    dx = self.df_points_xyz.iloc[i, 0] - self.df_points_xyz.iloc[j, 0]
                    dy = self.df_points_xyz.iloc[i, 1] - self.df_points_xyz.iloc[j, 1]
                    self.np_distance[i,j] = (dx**2+dy**2)**0.5
        self.np_distance = self.np_distance+self.np_distance.T

        # 1.2 Calculation of the Variogram matrix
        self.np_variogram = (1.5*self.np_distance/self.rang-0.5*(self.np_distance/self.rang)**3)*(self.np_distance<=self.rang)
        self.np_variogram += np.ones((self.no_of_points,self.no_of_points))*(self.np_distance>self.rang)
        self.np_variogram += (np.identity(self.no_of_points)==0)*self.nugget

        # 1.3 Calculation of the Covariance and its Inverse matrix
        self.np_covariance = self.nugget + self.spherical - self.np_variogram
        self.np_covariance_inv = inv(self.np_covariance)

    def get_z(self,x0,y0):
        # 2 Calculate the Matrix related to the Point we want to know

        # 2.1 Calculate the residual values
        self.np_z = self.df_points_xyz.iloc[:,2].to_numpy()
        self.mean = sum(self.df_points_xyz.iloc[:,2])/self.no_of_points
        self.np_residual = self.np_z - self.mean
        # 2.1 Calculate the distance matrix of point of interest
        self.np_xy = self.df_points_xyz.iloc[:,0:2].to_numpy()
        self.np_xy0 = np.array([x0,y0])
        self.np_xy0 = np.tile(self.np_xy0,(self.no_of_points,1))
        self.np_distance0 = (self.np_xy-self.np_xy0)**2
        self.np_distance0 = (self.np_distance0[:,0]+self.np_distance0[:,1])**0.5

        # 2.1 Calculate the Variogram matrix of point of interest
        self.np_variogram0 = (1.5*self.np_distance0/self.rang-0.5*(self.np_distance0/self.rang)**3)*(self.np_distance0<=self.rang)
        self.np_variogram0 += 1*(self.np_distance0>self.rang)
        self.np_variogram0 += self.nugget

        # 2.3 Calculate the Covariance Matrix for point of interest
        self.np_covariance0 = self.nugget + self.spherical - self.np_variogram0

        #2.4 Calculate the Weight Matrix for point of interest
        self.np_weight = np.dot(self.np_covariance_inv,self.np_covariance0)
        self.sum_weight = sum(self.np_weight)
        mean_weight = self.nugget + self.spherical - self.sum_weight

        #2.5 Calculate the final Kriging Estimate and the Corresponding Variance
        self.kriging_estimate = sum(self.np_residual * self.np_weight) + self.mean
        self.kriging_variance = self.nugget + self.spherical - sum(self.np_covariance0 * self.np_weight)

        return self.kriging_estimate