#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:18:36 2020

@author: alex

###############################################################################
# KalmanFilter.py
#
# Revision:     1.00
# Date:         6/6/2020
# Author:       Alex
#
# Purpose:      Implement a general-purpose Kalman filter class.
#
# Inputs:
# 1. All inputs to the class must be 2D Numpy arrays, even scalars.
# 2. See the __init__ function for a full description of inputs.
#
# Notes:
# 1. Notation based on professors Greg Welch and Gary Bishop's course pack on 
#    Kalman filters from the University of North Carolina at Chapel Hill.
# 2. The terminology is based on that used in Alex Becker's tutorial located at
#    www.kalmanfilters.net.   
# 3. Running this Python file directly will plot a simple example of a Kalman
#    filter used to estimate a constant voltage.
#
##############################################################################
"""

import numpy as np
from matplotlib import pyplot as plt


class KalmanFilter:
    def __init__(self, **kwargs):
        # Initial values (time step = 0) of parameters
        self.A = kwargs['A'] # State transition matrix, size (nxn)
        self.B = kwargs['B'] # Control-input model matrix, size (nxj)
        self.H = kwargs['H'] # Observation model matrix, size (mxn)
        self.P = kwargs['P'] # Estimate error covariance matrix, size (nxn)
        self.Q = kwargs['Q'] # Process noise covariance matrix, size (nxn)
        self.R = kwargs['R'] # Measurement noise covariance matrix, size (mxm)
        self.u = kwargs['u'] # Control variable vector, size (jx1)
        self.x = kwargs['x'] # Estimated state vector, size (nx1)
        # Derived parameters
        self.j = self.u.shape[0] # Number of control dimensions
        self.m = self.H.shape[0] # Number of measurement dimensions
        self.n = self.x.shape[0] # Number of state dimensions
        j, m, n = self.j, self.m, self.n
        self.Gain = np.zeros((n,m)) # Kalman gain, size (nxm)
        self.IM = np.eye(n) # Identity matrix, size (nxn)
        # Check dimensions of parameters
        assert self.A.shape == (n,n), "A dimensions must be {}".format((n,n))
        assert self.B.shape == (n,j), "B dimensions must be {}".format((n,j))
        assert self.H.shape == (m,n), "H dimensions must be {}".format((m,n))
        assert self.P.shape == (n,n), "P dimensions must be {}".format((n,n))
        assert self.Q.shape == (n,n), "Q dimensions must be {}".format((n,n))
        assert self.R.shape == (m,m), "R dimensions must be {}".format((m,m))
        assert self.u.shape == (j,1), "u dimensions must be {}".format((j,1))
        assert self.x.shape == (n,1), "x dimensions must be {}".format((n,1))
               
        
    def predict(self):
        # Time update (Predict)
        # State extrapolation (a priori state estimate)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        # Estimate error covariance extrapolation (a priori estimate error)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
    
    
    def correct(self, z, Q=None, R=None, u=None):
        # Measurement Update (Correct)
        if Q is None: Q = self.Q
        if R is None: R = self.R
        if u is None: u = self.u
        m = self.m
        assert z.shape == (m, 1), "z dimensions must be {}".format((m,1))
        # Compute Kalman gain
        _ = np.dot(np.dot(self.H, self.P), self.H.T) + R
        self.Gain = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(_)) 
        # Update state estimate with measurement (a posteriori state estimate)
        self.x = self.x + np.dot(self.Gain, (z - np.dot(self.H, self.x)))
        # Update the estimate error covariance (a posteriori estimate error)
        self.P = np.dot((self.IM - np.dot(self.Gain, self.H)), self.P)
        # Update any changes made to measurement/process noise or control inputs
        self.Q, self.R, self.u = Q, R, u
        return self.x, self.P


def constant_voltage_example():
    # Visualize and validate the Kalman Filter class with a simple estimate of 
    # a constant voltage.  Example from section 4.3 of:
    # https://www.cs.unc.edu/~tracker/media/pdf/SIGGRAPH2001_CoursePack_08.pdf
    
    # Input initial values (time step = 0) for all parameters
    A = np.array([[1]]) # Size of (1,1).  All parameters must be 2D arrays
    B = np.array([[0]])
    H = np.array([[1]])
    P = np.array([[1]])
    Q = np.array([[1e-5]])
    R = np.array([[0.1 ** 2]])
    u = np.array([[0]])
    x = np.array([[0]])
    filter_kwargs = {
            'A' : A,
            'B' : B,
            'H' : H,
            'P' : P,
            'Q' : Q,
            'R' : R,
            'u' : u,
            'x' : x}
    # Instantiate filter class
    kf = KalmanFilter(**filter_kwargs)
    # Generate measurements
    num_steps = 50 # Number of time steps to iterate over
    meas_mean = -0.37727 # True/actual voltage value
    meas_sigma = 0.1 # True/actual measurement standard deviation
    measurements = np.random.normal(meas_mean, meas_sigma, num_steps)
    # Initialize variables to store and plot results
    true_value = np.ones((num_steps,1)) * meas_mean
    estimates = []
    estimate_uncertainty = []
    step_range = np.arange(0, num_steps)
    # Filter loop
    for k in step_range:
        kf.predict()
        z = np.array([[measurements[k]]])
        estimate, uncertainty = kf.correct(z)
        estimates.append(np.squeeze(estimate))
        estimate_uncertainty.append(np.squeeze(uncertainty))
    # Plot results
    plt.figure()
    plt.plot(step_range+1, true_value, label='True Voltage')
    plt.plot(step_range+1, measurements, marker='+', linestyle="None", label='Measurements')
    plt.plot(step_range+1, estimates, linestyle="dashed", label='Filter Estimate')
    plt.xlabel('Iteration')
    plt.ylabel('Voltage (Volts)')
    plt.title('Kalman Filter Constant Voltage Exercise')
    plt.legend()
    plt.show()
    # Plot uncertainty
    plt.figure()
    plt.plot(step_range+1, estimate_uncertainty, label='Estimate Uncertainty')
    plt.xlabel('Iteration')
    plt.ylabel('Uncertainty (Volts^2)')
    plt.title('Kalman Filter Estimate Uncertainty')
    plt.legend()
    plt.show()
    
    
if __name__ == '__main__':
    constant_voltage_example()