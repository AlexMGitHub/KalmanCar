#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 23:03:23 2020

@author: alex
###############################################################################
# KalmanCar.py
#
# Revision:     1.00
# Date:         6/20/2020
# Author:       Alex
#
# Purpose:      Move a car down a race track.  The car's true path is displayed
#               in red, the estimated path calculated by a Kalman filter is 
#               displayed in blue, and a green circle represents the estimate
#               uncertainty.
#
# Inputs:
# 1. True/actual measurement standard deviations
# 2. Assumed measurement standard deviations used in Kalman filter
#
# Outputs:
# 1. An animation of a Kalman filter estimating a car's position on a race track
# 2. A plot of the estimate uncertainty versus iteration
#
##############################################################################
"""


#%% Imports
import pygame
import numpy as np
from RaceTrack import RaceTrack
from RaceTrack import Car
from RaceTrack import KalmanEstimates
import time


#%% Functions
def make_measurements(coord_sigma):
    # Function to generate satellite coordinate measurements and car 
    # rotation/speed measurements according to true/actual standard deviations
    x_pos = int(np.round(np.random.normal(rt.path_coords[path_idx][0], coord_sigma)))
    y_pos = int(np.round(np.random.normal(rt.path_coords[path_idx][1], coord_sigma)))
    # Velocity measurements
    car_rotation = ke.kalman_rotation[path_idx]
    car_speed = car.speed
    car_direction = car_rotation / 180 * np.pi
    x_vel = int(np.round(np.random.normal(car_speed * np.cos(car_direction), vel_sigma)))
    y_vel = int(np.round(np.random.normal(car_speed * np.sin(-car_direction), vel_sigma)))
    z = np.array([[x_pos],[y_pos],[x_vel],[y_vel]])
    return z

def update_measurement_noise(coord_var, R_matrix, coord_tree_var):
    # If car is obstructed by trees return return degraded kf coord variance,
    # measurement noise covariance matrix, and degraded satellite std deviation
    if rt.tree_obstruction[path_idx] == 1: 
        new_kf_pos_var = coord_tree_var
        new_sat_sigma = pos_tree_sigma
    else:
        new_kf_pos_var = coord_var
        new_sat_sigma = pos_sigma
    R_matrix[0,0], R_matrix[1,1] = new_kf_pos_var, new_kf_pos_var
    return new_kf_pos_var, R_matrix, new_sat_sigma
    

#%% Instantiate classes
rt = RaceTrack(line_width=3) # Displays racetrack and calculates car's path
car = Car(top_speed=6, acceleration=1) # Blits car image and path traveled

# Set true/actual measurement standard deviations
pos_sigma = 20 # Std dev of satellite's X/Y position measurement
pos_tree_sigma = 1000 # Std dev of satellite measurement when obstructed by trees
vel_sigma = 2 # Std dev of car's X/Y velocity measurement (pixels/update)

# Set Kalman filter's assumed measurement variance parameters. Do not have to
# be the same as the true/actual std devs. Can be tuned for better performance.
kf_pos_var = pos_sigma ** 2 # X and Y position measurement variance
kf_pos_tree_var = pos_tree_sigma ** 2 # X and Y position obstructed variance
kf_vel_var = vel_sigma ** 2 # X and Y velocity measurement variance
kf_Q_var_pos = 0.1 # Process noise covariance for position states
kf_Q_var_vel = 0.1 # Process noise covariance for velocity states
kf_Q_var_acc = 0.1 # Process noise covariance for unknown acceleration
kf_init_uncertainty = 0 # Initial estimate error covariance

# Set Kalman filter parameters (n=4,m=4,j=1). Refer to KalmanFilter.py.
A = np.array([[1,0,1,0],                        # (nxn)
              [0,1,0,1],
              [0,0,1,0],
              [0,0,0,1]])
B = np.zeros((4,1))                             # (nxj)
H = np.eye(4)                                   # (mxn)
P = np.eye(4) * kf_init_uncertainty             # (nxn)
Q = np.eye(4)                                   # (nxn) 
Q[0,0], Q[1,1], Q[2,2], Q[3,3] = kf_Q_var_pos, kf_Q_var_pos, \
                                  kf_Q_var_vel, kf_Q_var_vel
# Q = np.array([[0.25,0,0.5,0],
#                 [0,0.25,0,0.5],
#                 [0.5,0,1,0],
#                 [0,0.5,0,1]]) * kf_Q_var_acc  # Alternate formulation of Q

R = np.eye(4)                                   # (mxm)
R[0,0], R[1,1], R[2,2], R[3,3] = kf_pos_var, kf_pos_var, \
                                 kf_vel_var, kf_vel_var
u = np.array([[0]])                             # (jx1)
x = np.array([[rt.path_coords[0][0]],           # (nx1)
              [rt.path_coords[0][1]], 
              [1], [0]])
filter_kwargs = {
        'A' : A,
        'B' : B,
        'H' : H,
        'P' : P,
        'Q' : Q,
        'R' : R,
        'u' : u,
        'x' : x}

# Class instantiates Kalman filter and blits its estimates onto screen
ke = KalmanEstimates(**filter_kwargs) 

# Blit background and foreground and update screen
rt.gameDisplay.blit(rt.background, (0,0))
rt.gameDisplay.blit(rt.foreground, (0,0))
pygame.display.update()


#%% Main program loop
# Simulation display settings
time_sleep = 0.01 # Time between screen updates
draw_gps_measurements = True # Draws GPS measurements as magenta crosses
num_measurements = 100 # Number of previous measurements to draw each iteration
# Begin loop
path_idx = 0
gps_measurements = []
while path_idx < len(rt.path_coords):
    # Time update (predict)
    ke.predict()
    car.move(path_idx)
    # Make measurements
    kf_pos_var_update, R_update, pos_sigma_update = update_measurement_noise(
        kf_pos_var, R, kf_pos_tree_var)
    z = make_measurements(pos_sigma_update)
    # Draw GPS measurements (if enabled)
    gps_measurements.append((z[0,0], z[1,0]))
    if draw_gps_measurements: 
        ke.draw_GPS_measurements(gps_measurements, num_measurements)
    # Measurement update (correct)
    ke.estimate_coord(z, R=R_update)
    # Update display and path index with car's new position
    pygame.display.update()
    path_idx += car.speed
    time.sleep(time_sleep)


#%% Plots and clean up
ke.plot_uncertainty() 
#pygame.image.save(rt.gameDisplay,"docs/images/screenshot.png")   
#time.sleep(20)
#pygame.quit()
#ke.close_plots()




        