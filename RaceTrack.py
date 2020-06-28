#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 15:13:53 2020

@author: alex
###############################################################################
# RaceTrack.py
#
# Revision:     1.00
# Date:         6/20/2020
# Author:       Alex
#
# Purpose:      Implement a parent class (RaceTrack) and two child classes
#               (Car and KalmanEstimate). Classes called by KalmanCar.py to 
#               visually represent a Kalman filter estimating the position of
#               a car driving along a race track.
#
# Inputs:
# 1. RaceTrack(): Line width of true/actual path and estimated path lines
# 2. Car(): Top speed and acceleration of car in pixels/update
# 3. KalmanEstimate(): Kalman filter input kwargs, see KalmanFilter.py
#
# Notes:
# 1. Race track art assets created by Kenney Vleugels (www.kenney.nl) and 
#   downloaded from https://opengameart.org/content/racing-pack under the 
#   CC0 1.0 universal public domain license.
#
##############################################################################
"""


#%% Imports
import numpy as np
import pygame
from matplotlib import pyplot as plt
from KalmanFilter import KalmanFilter


#%% RaceTrack class
# This class builds the background image containing the race track, the 
# foreground image containing trees, and calculates the path the car will take 
# around the race track. Also calculates the location of bends in the track, 
# location of trees obstructing the track, and the orientation of the car image 
# as it drives along the track. 
class RaceTrack():
    def __init__(self, line_width=3):
        pygame.init()
        pygame.mixer.quit() # Fixes bug with high Pygame CPU usage
        pygame.display.set_caption('KalmanCar')
        self.game_width = 1280
        self.game_height = 896
        self.gameDisplay = pygame.display.set_mode((self.game_width, 
                                                    self.game_height))
        self.line_width = line_width
        self.init_tiles() # Load and scale tiles used to draw background
        self.init_foreground() # Compute tree locations 
        self.path_calculation() # Compute car location and orientation on track
        self.background = pygame.image.load('img/GameAssets/background.png').convert()
        self.foreground = pygame.image.load('img/GameAssets/foreground.png').convert_alpha()
        
    
    def init_tiles(self):
        # Import tiles and resize
        tile_scale = (32, 32) # Dimensions of re-scaled tiles
        # Roads
        road_horz = pygame.image.load('img/Tiles/Asphalt road/road_asphalt02.png')
        road_horz = pygame.transform.scale(road_horz, tile_scale)
        road_vert = pygame.image.load('img/Tiles/Asphalt road/road_asphalt01.png')
        road_vert = pygame.transform.scale(road_vert, tile_scale)
        finish_horz = pygame.image.load('img/Tiles/Asphalt road/road_asphalt43.png')
        finish_horz = pygame.transform.scale(finish_horz, tile_scale) 
        bend_br = pygame.image.load('img/Tiles/Asphalt road/road_asphalt03.png')
        bend_br = pygame.transform.scale(bend_br, tile_scale)    
        bend_bl = pygame.image.load('img/Tiles/Asphalt road/road_asphalt05.png')
        bend_bl = pygame.transform.scale(bend_bl, tile_scale)    
        bend_tr = pygame.image.load('img/Tiles/Asphalt road/road_asphalt39.png')
        bend_tr = pygame.transform.scale(bend_tr, tile_scale)    
        bend_tl = pygame.image.load('img/Tiles/Asphalt road/road_asphalt41.png')
        bend_tl = pygame.transform.scale(bend_tl, tile_scale)    
        # Ground
        ground = pygame.image.load('img/Tiles/Sand/land_sand05.png')
        ground = pygame.transform.scale(ground, tile_scale)
        # Create a list of tiles that represents their (x,y) placement on screen
        self.tile_mapping = [
            40*[ground],
            40*[ground],
            [2*[ground], [bend_br], 2*[road_horz], [finish_horz], 31*[road_horz], [bend_bl], 2*[ground]],
            [2*[ground], [road_vert], 34*[ground], [road_vert], 2*[ground]],
            [2*[ground], [road_vert] , 34*[ground], [road_vert], 2*[ground]],
            [2*[ground], [road_vert], 34*[ground], [road_vert], 2*[ground]],
            [2*[ground], [road_vert], 34*[ground], [road_vert], 2*[ground]],
            [2*[ground], [bend_tr], 24*[road_horz], [bend_bl], 4*[ground], [bend_br], 4*[road_horz], [bend_tl], 2*[ground]],
            [27*[ground], [road_vert], 4*[ground], [road_vert], 7*[ground]],
            [27*[ground], [road_vert], 4*[ground], [road_vert], 7*[ground]],
            [27*[ground], [road_vert], 4*[ground], [road_vert], 7*[ground]],
            [27*[ground], [road_vert], 4*[ground], [road_vert], 7*[ground]],
            [2*[ground], [bend_br], 8*[road_horz], [bend_bl], 4*[ground], [bend_br], 4*[road_horz], [bend_bl], 5*[ground], [road_vert], 4*[ground], [bend_tr], 4*[road_horz], [bend_bl], 2*[ground]],
            [2*[ground], [road_vert], 8*[ground], [road_vert], 4*[ground], [road_vert], 4*[ground], [road_vert], 5*[ground], [road_vert], 9*[ground], [road_vert], 2*[ground]],
            [2*[ground], [road_vert], 8*[ground], [road_vert], 4*[ground], [road_vert], 4*[ground], [road_vert], 5*[ground], [road_vert], 9*[ground], [road_vert], 2*[ground]],
            [2*[ground], [road_vert], 8*[ground], [road_vert], 4*[ground], [road_vert], 4*[ground], [road_vert], 5*[ground], [road_vert], 9*[ground], [road_vert], 2*[ground]],
            [2*[ground], [road_vert], 8*[ground], [road_vert], 4*[ground], [road_vert], 4*[ground], [road_vert], 5*[ground], [road_vert], 9*[ground], [road_vert], 2*[ground]],
            [2*[ground], [bend_tr], 4*[road_horz], [bend_bl], 3*[ground], [road_vert], 4*[ground], [road_vert], 4*[ground], [road_vert], 5*[ground], [road_vert], 4*[ground], [bend_br], 4*[road_horz], [bend_tl], 2*[ground]],
            [7*[ground], [road_vert], 3*[ground], [road_vert], 4*[ground], [road_vert], 4*[ground], [road_vert], 5*[ground], [road_vert], 4*[ground], [road_vert], 7*[ground]],
            [7*[ground], [road_vert], 3*[ground], [road_vert], 4*[ground], [road_vert], 4*[ground], [road_vert], 5*[ground], [road_vert], 4*[ground], [road_vert], 7*[ground]],
            [7*[ground], [road_vert], 3*[ground], [road_vert], 4*[ground], [road_vert], 4*[ground], [road_vert], 5*[ground], [road_vert], 4*[ground], [road_vert], 7*[ground]],
            [7*[ground], [road_vert], 3*[ground], [road_vert], 4*[ground], [road_vert], 4*[ground], [road_vert], 5*[ground], [road_vert], 4*[ground], [road_vert], 7*[ground]],
            [2*[ground], [bend_br], 4*[road_horz], [bend_tl], 3*[ground], [bend_tr], 4*[road_horz], [bend_tl], 4*[ground], [bend_tr], 5*[road_horz], [bend_tl], 4*[ground], [bend_tr], 4*[road_horz], [bend_bl], 2*[ground]],
            [2*[ground], [road_vert], 34*[ground], [road_vert], 2*[ground]],
            [2*[ground], [road_vert], 34*[ground], [road_vert], 2*[ground]],
            [2*[ground], [bend_tr], 34*[road_horz], [bend_tl], 2*[ground]],
            40*[ground],
            40*[ground]]   
        self.tile_scale = tile_scale
        self.road_horz = road_horz
        self.road_vert = road_vert
        self.finish_horz = finish_horz
        self.bend_br = bend_br
        self.bend_bl = bend_bl
        self.bend_tr = bend_tr
        self.bend_tl = bend_tl
        self.ground = ground

    
    def init_foreground(self):
        tile_cols = int(self.game_width/self.tile_scale[0])
        tile_rows = int(self.game_height/self.tile_scale[1])
        tile_mapping = self.tile_mapping
        flat_tile_list = []
        for row in range(0,tile_rows):
            try:
                flat_list = [item for sublist in tile_mapping[row] for item in sublist]
            except:
                flat_list = tile_mapping[row]
            flat_tile_list.append(flat_list)
        
        self.flat_tile_array = np.array(flat_tile_list)
        self.tree_tiles = np.zeros((tile_rows,tile_cols))
        # Trees along bottom horizontal road
        row = 25
        tiles = range(15,36)
        for col in tiles:
            self.tree_tiles[row, col] = 1
        # Trees along left side bends
        tile_rows = range(11,23)
        tile_cols = range(0,11)
        for row in tile_rows:
            for col in tile_cols:
                if self.flat_tile_array[row,col] != self.ground:
                    self.tree_tiles[row, col] = 1
        # Trees along center bends
        tile_rows = range(7,23)
        tile_cols = range(21,28)
        for row in tile_rows:
            for col in tile_cols:
                if self.flat_tile_array[row,col] != self.ground:
                    self.tree_tiles[row, col] = 1
    

    def path_calculation(self):
        # Use the tile mapping list to calculate the car's path and store the 
        # coordinates. Also keep track of whether the car is moving along a 
        # a bend or behind trees. Orientation (rotation) of car image is also
        # computed and stored in a list.
        path_coords = [] # X/Y coordinates of car moving along race track
        self.path_bend = [] # 1 if coordinate is located in bend tile
        self.path_angles = [] # Angle car is facing at each point in path
        self.tree_obstruction = [] # 1 if tree is obstructing race track
        tile_mapping = self.tile_mapping
        tile_scale = self.tile_scale
        half_x = int(tile_scale[0]/2)
        half_y = int(tile_scale[1]/2)
        # Calculate path coordinates at start/finish line of track
        start_x = 5 * tile_scale[0] # X-coord of left side of finish line tile
        start_y = int(2.5 * tile_scale[1]) # Y-coord of middle of finish line tile
        for pixel in range(0, tile_scale[0]):
            path_coords.append((start_x + pixel, start_y))
            self.path_bend.append(0) # Not a bend tile
            self.path_angles.append(0) # Car oriented at 0 degress (right)
            self.tree_obstruction.append(0) # No trees
        start_idx_row = 2 # Row index of finish line tile
        start_idx_col = 5 # Column index of finish line tile
        rowidx = 2 # Set indices for loop. Next tile after the finish
        colidx = 6 # line is to the right.
        directionx = 1 # Car is traveling to the right
        directiony = 0 # Car is not moving vertically
        # Follow the road until it loops back to the finish line
        while not (rowidx, colidx) == (start_idx_row, start_idx_col):
            flat_list = [item for sublist in tile_mapping[rowidx] for item in sublist]
            tile_x = colidx * tile_scale[0] # Top-left corner of tile
            tile_y = rowidx * tile_scale[1]
            tree = self.tree_tiles[rowidx,colidx] # Is tile obstructed by trees
            if flat_list[colidx] == self.road_horz:
                angle = 0
                if directionx < 0: 
                    tile_x += tile_scale[0]-1 # Moving left
                    angle = 180
                for pixel in range(0, tile_scale[0]):
                    path_coords.append((tile_x + directionx*pixel, 
                                        tile_y + half_y))
                    self.path_bend.append(0)
                    self.path_angles.append(angle)
                    self.tree_obstruction.append(tree)
                directiony = 0
                colidx += directionx
            elif flat_list[colidx] == self.road_vert:
                angle = -90
                if directiony < 0: 
                    tile_y += tile_scale[1]-1 # Moving up
                    angle = 90
                for pixel in range(0, tile_scale[1]):
                    path_coords.append((tile_x + half_x, 
                                        tile_y + directiony*pixel))
                    self.path_bend.append(0)
                    self.path_angles.append(angle)
                    self.tree_obstruction.append(tree)
                directionx = 0
                rowidx += directiony
            elif flat_list[colidx] == self.bend_tr:
                arcstart = (tile_x + tile_scale[0]-1, tile_y+half_y)
                arcstop = (tile_x + half_x, tile_y)
                bend_coords = self.calc_arc(arcstart, arcstop)
                self.path_bend.extend(len(bend_coords)*[1])
                self.tree_obstruction.extend(len(bend_coords)*[tree])
                angles_array = np.linspace(0,90,num=len(bend_coords))
                if directiony == 0: # Entering from right and headed up
                    path_coords.extend(bend_coords)
                    self.path_angles.extend(list(angles_array*-1-180))
                    directionx = 0
                    directiony = -1
                    rowidx += directiony
                elif directionx == 0: # Entering from top and headed right
                    path_coords.extend(bend_coords[::-1])
                    self.path_angles.extend(list(angles_array-90))
                    directionx = 1
                    directiony = 0
                    colidx += directionx
            elif flat_list[colidx] == self.bend_tl:
                arcstart = (tile_x, tile_y+half_y)
                arcstop = (tile_x + half_x, tile_y)
                bend_coords = self.calc_arc(arcstart, arcstop)
                self.path_bend.extend(len(bend_coords)*[1])
                self.tree_obstruction.extend(len(bend_coords)*[tree])
                angles_array = np.linspace(0,90,num=len(bend_coords))
                if directiony == 0: # Entering from left and headed up
                    path_coords.extend(bend_coords)
                    self.path_angles.extend(list(angles_array))
                    directionx = 0
                    directiony = -1
                    rowidx += directiony
                elif directionx == 0: # Entering from top and headed left
                    path_coords.extend(bend_coords[::-1])
                    self.path_angles.extend(list(angles_array*-1-90))
                    directionx = -1
                    directiony = 0
                    colidx += directionx
            elif flat_list[colidx] == self.bend_br:
                arcstart = (tile_x+tile_scale[0]-1, tile_y+half_y)
                arcstop = (tile_x + half_x, tile_y+tile_scale[1]-1)
                bend_coords = self.calc_arc(arcstart, arcstop)  
                self.path_bend.extend(len(bend_coords)*[1])
                self.tree_obstruction.extend(len(bend_coords)*[tree])
                angles_array = np.linspace(0,90,num=len(bend_coords))
                if directiony == 0: # Entering from right and headed down
                    path_coords.extend(bend_coords)  
                    self.path_angles.extend(list(angles_array-180))
                    directionx = 0
                    directiony = 1
                    rowidx += directiony
                elif directionx == 0: # Entering from bottom and headed right
                    path_coords.extend(bend_coords[::-1])   
                    self.path_angles.extend(list(angles_array)[::-1])
                    directionx = 1
                    directiony = 0
                    colidx += directionx
            elif flat_list[colidx] == self.bend_bl:
                arcstart = (tile_x, tile_y+half_y)
                arcstop = (tile_x + half_x, tile_y+tile_scale[1]-1)
                bend_coords = self.calc_arc(arcstart, arcstop)         
                self.path_bend.extend(len(bend_coords)*[1])
                self.tree_obstruction.extend(len(bend_coords)*[tree])
                angles_array = np.linspace(0,90,num=len(bend_coords))
                if directiony == 0: # Entering from left and heading down
                    path_coords.extend(bend_coords) 
                    self.path_angles.extend(list(angles_array*-1))
                    directiony = 1
                    directionx = 0
                    rowidx += directiony
                elif directionx == 0: # Entering from bottom and heading left
                    path_coords.extend(bend_coords[::-1])
                    self.path_angles.extend(list(angles_array+90))
                    directiony = 0
                    directionx = -1
                    colidx += directionx
        # Hack to remove duplicate coordinates
        #self.path_coords = list(dict.fromkeys(path_coords))
        self.path_coords = path_coords
  
    
    def calc_arc(self, arcstart, arcstop):
        # Ugly function that returns one of four possible paths to traverse
        # a bend in the track.
        directionx = 1
        directiony = 1
        if arcstop[0] < arcstart[0]: directionx = -1
        if arcstop[1] < arcstart[1]: directiony = -1
        arc_coords = [arcstart]
        sym_arc1 = [(1,0),(1,1),(1,0),(1,1),(1,0),(1,1),(1,0),(1,1),(1,1),
                   (1,1),(1,1),(1,1),(0,1),(1,1),(0,1),(1,1),(0,1),(1,1)]
        sym_arc2 = [(1,0),(1,1),(1,0),(1,1),(1,0),(1,1),(1,0),(1,1),(1,1),
                   (1,1),(1,1),(1,1),(1,1),(0,1),(1,1),(0,1),(1,1),(0,1),(1,1)]
        asym_arc1 = [(1,0),(1,0),(1,0),(1,1),(1,0),(1,1),(1,0),(1,1),(1,0),
                    (1,1),(1,0),(1,1),
                    (1,1),(0,1),(1,1),(0,1),(1,1),(0,1),(1,1),(0,1),(0,1)]
        asym_arc2 = [(1,0),(1,0),(1,1),(1,0),(1,1),(1,0),(1,1),(1,0),(1,1),
                     (1,1),(1,1),(0,1),(1,1),(0,1),(1,1),(0,1),(1,1),(0,1),
                     (1,1),(0,1)]
        arc_len = np.abs(np.subtract(arcstop,arcstart))
        if arc_len[0] == arc_len[1] == 15:
            arc_pattern = sym_arc1
        elif arc_len[0] == arc_len[1] == 16:
            arc_pattern = sym_arc2
        elif arc_len[0] > arc_len[1]:
            arc_pattern = asym_arc1
        elif arc_len[0] < arc_len[1]:
            arc_pattern = asym_arc2
        # Draw arc
        for coord in arc_pattern:
            new_coord = (arc_coords[-1][0] + coord[0]*directionx, 
                         arc_coords[-1][1] + coord[1]*directiony)
            arc_coords.append(new_coord)
        arc_coords.append(arcstop)
        return arc_coords


    def background_builder(self):
        # Load, convert, and scale tiles used as background.  Assemble them 
        # into a single image and save to disk. Only needs to be run if a 
        # change to the tile mapping has bene made.
        game_width = self.game_width
        game_height = self.game_height
        background = pygame.Surface([game_width,game_height])
        # Use a list of tiles that represents their (x,y) placement on screen
        tile_scale = self.tile_scale
        horzdim = game_width // tile_scale[0] # Horizontal number of tiles
        vertdim = game_height // tile_scale[1] # Vertical number of tiles
        tile_mapping = self.tile_mapping
        # Place ground down first
        for x in range(0, horzdim):
            for y in range(0, vertdim):
                xpos = x * tile_scale[0]
                ypos = y * tile_scale[1]
                background.blit(self.ground, (xpos,ypos))
        # Iterate through the tile_mapping list and draw each tile on the screen
        for row_idx, tile_row in enumerate(tile_mapping):
            temp_idx = 0 
            for col_idx, tile_el in enumerate(tile_row):
                try: # If element is a sub-list of tiles iterate though it
                    for sub_idx, tile in enumerate(tile_el):
                        xpos = (col_idx+temp_idx+sub_idx)*tile_scale[0]
                        ypos = row_idx*tile_scale[1]
                        background.blit(tile, (xpos,ypos))
                    temp_idx += sub_idx # Keep track of sub-list indices
                except: # If just a single tile blit it
                    xpos = (col_idx+temp_idx)*tile_scale[0]
                    ypos = row_idx*tile_scale[1]
                    background.blit(tile_el, (xpos,ypos))
        # Save background to disk
        pygame.image.save(background, 'img/GameAssets/background.png')
        

    def foreground_builder(self):
        # Load, convert, and scale trees used as foreground.  Assemble them 
        # into a single transparent image and save to disk. Only needs to be 
        # run if a change to the tile mapping or tree locations has been made.
        game_width = self.game_width
        game_height = self.game_height
        tile_scale = self.tile_scale
        tile_cols = int(game_width/tile_scale[0])
        tile_rows = int(game_height/tile_scale[1])
        foreground = pygame.Surface([game_width,game_height], pygame.SRCALPHA)
        # Load and re-scale tree image                
        tree_scale = (32, 32) # Dimensions of re-scaled trees
        tree = pygame.image.load('img/Objects/tree_small.png')
        tree = pygame.transform.scale(tree, tree_scale)  
        # Place trees
        tree_offset = 10 # Number of pixels trees offset from road
        for row_idx in range(0,tile_rows):
            for col_idx in range(0, tile_cols):
                if self.tree_tiles[row_idx,col_idx] == 1:
                    xpos = self.tile_scale[0]*col_idx #+ self.tile_scale[0]/2
                    ypos = row_idx*self.tile_scale[1] #+ self.tile_scale[1]/2
                    if self.flat_tile_array[row_idx,col_idx] == self.road_horz:
                        foreground.blit(tree, (xpos,ypos-tile_scale[1]+tree_offset))
                        foreground.blit(tree, (xpos,ypos+tile_scale[1]-tree_offset))
                    if self.flat_tile_array[row_idx,col_idx] == self.road_vert:
                        foreground.blit(tree, (xpos-tile_scale[0]+tree_offset,ypos))
                        foreground.blit(tree, (xpos+tile_scale[0]-tree_offset,ypos))
                    if self.flat_tile_array[row_idx,col_idx] == self.bend_tl:
                        foreground.blit(tree, (xpos+tile_scale[0]-tree_offset,ypos))
                        foreground.blit(tree, (xpos,ypos+tile_scale[1]-tree_offset))
                    if self.flat_tile_array[row_idx,col_idx] == self.bend_tr:
                        foreground.blit(tree, (xpos-tile_scale[0]+tree_offset,ypos))
                        foreground.blit(tree, (xpos,ypos+tile_scale[1]-tree_offset))
                    if self.flat_tile_array[row_idx,col_idx] == self.bend_bl:
                        foreground.blit(tree, (xpos+tile_scale[0]-tree_offset,ypos))
                        foreground.blit(tree, (xpos,ypos-tile_scale[1]+tree_offset))
                    if self.flat_tile_array[row_idx,col_idx] == self.bend_br:
                        foreground.blit(tree, (xpos-tile_scale[0]+tree_offset,ypos))
                        foreground.blit(tree, (xpos,ypos-tile_scale[1]+tree_offset))
        # Save foreground to disk
        pygame.image.save(foreground, 'img/GameAssets/foreground.png')
        
        
#%% Car class
# Initialize the car with a user-defined top speed and acceleration.  Calculate
# the speed of the car at each point along the track, and store these values in
# a list. Blit the rotated car image onto the track, and draw the true/actual
# path of the car as a red line.
class Car(RaceTrack):
    def __init__(self,top_speed, acceleration):
        self.car_scale = (30,49)
        self.car = pygame.image.load('img/Cars/car_black_2.png').convert_alpha()
        self.car = pygame.transform.scale(self.car, self.car_scale)
        self.car = pygame.transform.rotate(self.car, -90)
        self.rotated_car = self.car.copy()
        self.car_rotation = 0 # Rotation of car image in degrees
        self.speed = 1 # Initial speed of car
        self.top_speed = int(np.clip(top_speed,1,20)) # Top speed of car in pixels/update
        self.acceleration = int(np.clip(acceleration,1,20)) # Acceleration of car in pixels/update
        self.cornering = 1 # Speed of car around corners in pixels/update
        self.speed_list = [] # Keeps track of car velocity along road
        super().__init__() # Inherit RaceTrack class variables
        self.car_coord = (self.path_coords[0][0] - int(self.car_scale[1]/2),
                          self.path_coords[0][1] - int(self.car_scale[0]/2))
        self.speed_calc() # Calculate car's velocity along road
                
        
    def move(self, coord_idx):
        # Move the car to its next location according to its speed. Delete the
        # old car image and blit the new car image with the proper orientation.
        # Center the rotated car image on the track.
        self.speed = self.speed_list[coord_idx]
        self.blit_ground('both') # Erase car's previous position
        new_coord = coord_idx+self.speed
        if new_coord >= len(self.path_coords): new_coord = 0 # Finish line
        self.car_coord = self.path_coords[new_coord] 
        self.calc_rotation(new_coord) # Rotate car image to proper orientation
        center = self.rotated_car.get_rect().center
        self.car_coord = tuple(np.subtract(self.car_coord, center))
        self.gameDisplay.blit(self.rotated_car, self.car_coord)
        self.blit_ground('foreground') # Blit foreground on top of car
        self.draw_path(coord_idx) # Draw true/actual path of car on race track
                
        
    def calc_rotation(self, coord_idx):
        # Rotate a new copy of the original car image to avoid distortion
        self.car_rotation = self.path_angles[coord_idx]    
        self.rotated_car = pygame.transform.rotate(self.car, self.car_rotation)
    
    
    def blit_ground(self, ground):
        # Blit the background AND foreground on top of the car's previous 
        # position, OR blit only the foreground on top of the car's current 
        # position.
        erase_dim = tuple(self.rotated_car.get_rect()[2:])
        coord = self.car_coord
        if ground == 'both':
            self.gameDisplay.blit(self.background, coord, 
                                  pygame.Rect(coord, erase_dim)) 
        self.gameDisplay.blit(self.foreground, coord, 
                              pygame.Rect(coord, erase_dim))


    def speed_calc(self):
        # Generates a list representing the car's speed along the road
        # Calculates the car's speed according to the car's characteristics.
        # Speed is clipped to be no less than the cornering speed and no more
        # than the car's top speed.
        speed_list = np.array(self.path_bend)*self.cornering
        speed_list = list(speed_list)
        speed = self.speed
        speed_list[0] = speed
        idx = speed
        while True:
            try:
                if self.path_bend[idx] == 0: # On a straight road, accelerate
                    speed = np.clip(speed + self.acceleration, self.cornering, 
                                    self.top_speed)
                    speed_list[idx] = speed
                    idx += speed
                else: # At a bend
                    # Lower speed coming into bend
                    dist_to_corner = self.path_bend[idx-speed:idx+1].index(1)
                    speed_list[idx-speed] = dist_to_corner
                    # Now accelerate coming out of bend
                    idx = idx - speed + dist_to_corner
                    dist_to_road = self.path_bend[idx:idx+self.tile_scale[0]].index(0)
                    idx += dist_to_road
                    speed = np.clip(self.cornering + self.acceleration, 
                                    self.cornering, self.top_speed)
                    speed_list[idx] = speed
                    idx += speed
            except:
                break # Reached end of track
        self.speed_list = speed_list


    def draw_path(self, stop_idx):
        # Draw actual path of car as a red line
        start_idx = 0
        for idx, coord in enumerate(self.path_coords[start_idx:stop_idx]):
            if start_idx + idx < len(self.path_coords)-1:
                pygame.draw.line(self.gameDisplay, (255,0,0), coord, 
                                 self.path_coords[start_idx+idx+1], self.line_width)
    

#%% KalmanEstimates class
# Initialize the class with the Kalman filter kwargs. The class then calculates
# the pixel-by-pixel orientation of the car at every pixel along the race 
# track. The orientation calculated in the RaceTrack() class is just an 
# approximation used to rotate the car image in a smooth manner, and is not an 
# accurate representation of the car's actual orientation.
#
# The KalmanEstimates class implements the KalmanFilter class predict() and 
# correct() methods, draws the estimated location of the car with a blue line,
# and draws the estimate uncertainty as a transparent green circle with a 
# radius equal to 3 times the standard deviation of the estimate uncertainty.
class KalmanEstimates(RaceTrack):
    def __init__(self, **filter_kwargs):
        super().__init__() # Inherit RaceTrack class variables
        self.kalman_rotation = self.path_angles.copy()
        self.init_rotation() # Calculate pixel-by-pixel car orientation
        self.kf = KalmanFilter(**filter_kwargs) # Instantiate Kalman filter
        # Get initial estimate and estimate uncertainty values
        self.estimate = (self.kf.x[0,0], self.kf.x[1,0])
        self.estimate_uncertainty = (self.kf.P[0,0], self.kf.P[1,1],
                                     self.kf.P[2,2], self.kf.P[3,3])
        self.estimates = [self.estimate]
        self.uncertainties = [self.estimate_uncertainty]
        # Transparent surface used to blit estimate uncertainty circle
        self.surface = pygame.Surface((self.game_width, self.game_height), 
                                      pygame.SRCALPHA)          
      
        
    def init_rotation(self):
        # Calculate the orientation of the car at every pixel along the race
        # track. There are only 8 possible orientations corresponding to one of
        # 8 neighboring pixels. Because of this, the path coordinates must 
        # never skip a pixel and no pixel should have more than two neighbors
        # (adjacent pixels).
        for idx, bend_bool in enumerate(self.path_bend):
            try:
                if bend_bool == 1 or self.path_bend[idx+1] == 1:
                    coord_diff = tuple(np.subtract(self.path_coords[idx+1], self.path_coords[idx]))
                    if coord_diff[0] != 0 and coord_diff[1] != 0:
                        if coord_diff[0] == coord_diff[1] == 1:
                            self.kalman_rotation[idx] = -45
                        if coord_diff[0] == coord_diff[1] == -1:
                            self.kalman_rotation[idx] = 135
                        if coord_diff[0] == 1 and coord_diff[1] == -1:
                            self.kalman_rotation[idx] = 45
                        if coord_diff[0] == -1 and coord_diff[1] == 1:
                            self.kalman_rotation[idx] = -135
                    else:
                        if coord_diff[0] == 1: self.kalman_rotation[idx] = 0
                        if coord_diff[0] == -1: self.kalman_rotation[idx] = 180
                        if coord_diff[1] == 1: self.kalman_rotation[idx] = -90
                        if coord_diff[1] == -1: self.kalman_rotation[idx] = 90
            except:
                pass # Reached finish line (end of coordinate list)


    def predict(self):
        # Implement the Kalman filter predict() method, and erase the previous
        # iteration's uncertainty circle.
        self.kf.predict()
        self.erase_uncertainty_circle()
  
    
    def estimate_coord(self, z, Q=None, R=None, u=None):
        # Implement the Kalman filter correct() method, store the results in a
        # list, and draw the estimated path and uncertainty circle.
        x, P = self.kf.correct(z, Q, R, u)
        x_coord = int(np.round(x[0,0]))
        y_coord = int(np.round(x[1,0]))
        x_uncertainty = P[0,0]
        y_uncertainty = P[1,1]
        xvel_uncertainty = P[2,2]
        yvel_uncertainty = P[3,3]
        self.estimate = (x_coord, y_coord)
        self.estimates.append(self.estimate)
        self.estimate_uncertainty = (x_uncertainty, y_uncertainty, 
                                      xvel_uncertainty, yvel_uncertainty)
        self.uncertainties.append(self.estimate_uncertainty)
        self.draw_estimate_path()
        self.draw_uncertainty_circle()
    
    
    def draw_estimate_path(self):
        # Draw estimates calculated by Kalman filter as a blue line
        for idx, coord in enumerate(self.estimates):
            if idx < len(self.estimates)-1:
                pygame.draw.line(self.gameDisplay, (0,0,255), coord, 
                                 self.estimates[idx+1], self.line_width)

    
    def erase_uncertainty_circle(self):    
        # Delete old uncertainty circle at the previous car coordiante. Make 
        # sure to clear transparent surface as well as blitting the background 
        # and foreground to the gameDisplay.
        idx = len(self.estimates)-1
        coord = self.estimates[idx]
        radius = 3*int(np.round(np.max(self.uncertainties[idx])**0.5))
        blit_coord = np.subtract(coord, (radius,radius))
        clear_rect = pygame.Rect(blit_coord, (2*radius,2*radius))
        self.gameDisplay.blit(self.background, blit_coord, clear_rect) 
        self.gameDisplay.blit(self.foreground, blit_coord, clear_rect)
        self.surface.fill((0,0,0,0), clear_rect)

    
    def draw_uncertainty_circle(self):
        # Draw new uncertainty circle at the current coordinate. In order to 
        # get a transparent circle must first blit the circle to a transparent 
        # surface, then blit the transparent surface to the gameDisplay.
        # Radius of circle is 3 times the standard deviation of the x-position 
        # estimate uncertainty.
        idx = len(self.estimates)-1
        coord = self.estimates[idx]
        radius = 3*int(np.round(self.uncertainties[idx][0]**0.5))
        pygame.draw.circle(self.surface, (0,255,0,128), coord, radius)
        blit_coord = np.subtract(coord, (radius,radius))
        self.gameDisplay.blit(self.surface, blit_coord, 
                              pygame.Rect(blit_coord, (2*radius,2*radius)))

        
    def draw_GPS_measurements(self, measurements, num_measurements):
        # Draw GPS measurements as magenta crosses.  The number of crosses
        # drawn each iteration can be limited to improve simulation frame rate.
        cross_dim = 7 # Cross height and width in pixels
        cross_thk = 3 # Line width of cross
        offset = (cross_dim - 1) / 2
        color = (204,0,204)
        for x, y in measurements[-num_measurements:]:
            pygame.draw.line(self.gameDisplay, color, (x-offset, y), 
                                 (x+offset, y), cross_thk)
            pygame.draw.line(self.gameDisplay, color, (x, y-offset), 
                                 (x, y+offset), cross_thk)
                        
        
    def plot_uncertainty(self):
        # Plot Kalman estimate uncertainty versus iteration in a Matplotlib
        # figure. 
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        axes = [ax1, ax1, ax2, ax2]
        colors = ['tab:red','tab:red','tab:blue','tab:blue']
        legend_labels = ['X-coord Estimate Uncertainty',
                         'Y-coord Estimate Uncertainty',
                         'X-velocity Estimate Uncertainty',
                         'Y-velocity Estimate Uncertainty']
        linestyles = ['-','--','-','--']
        plots = []
        miny, maxy = 1e6, 0
        for idx, uncertainty in enumerate(zip(*self.uncertainties)):
            lineplot = axes[idx].plot(uncertainty, color=colors[idx], 
                                      linestyle=linestyles[idx],
                                      label=legend_labels[idx])
            plots += lineplot
            if idx > 1:
                maxy = np.max([maxy, np.max(uncertainty)])
                miny = np.min([miny, np.min(uncertainty)])
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Position Uncertainty (Pixels^2)', color=colors[0])
        ax1.tick_params(axis='y', labelcolor=colors[0])
        ax2.set_ylabel('Velocity Uncertainty (Pixels^2)', color=colors[-1])
        ax2.tick_params(axis='y', labelcolor=colors[-1])
        # Auto scale because MatPlotLib doesn't handle small numbers correctly
        if maxy < 1e-10: maxy = 1e-10
        dy = (maxy - miny) * 0.1
        ax2.set_ylim(miny-dy, maxy+dy)
        plt.title('Kalman Filter Estimate Uncertainty')
        ax1.legend(plots, legend_labels, loc=0)
        fig.tight_layout()
        plt.show()
        
    def close_plots(self):
        # Close all Matplotlib plots
        plt.close('all')