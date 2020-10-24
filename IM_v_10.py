# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:57:18 2020

@author: li xiang
"""

import numpy as np
from nn import ImplicitLayer, ImplicitFunctionInf
import torch
from torch import optim
import csv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gym
import cv2
import math
import time


import sys, math
import numpy as np
import matplotlib.pyplot as plt

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
#from gym.envs.box2d.car_dynamics import Car
from kinematic_car import  Car
from gym.utils import colorize, seeding, EzPickle

import pyglet
from pyglet import gl
from math import inf
#from OpenGL.raw.GL.VERSION.GL_1_0 import GL_LINES
from simple_pid import PID
import time


# State consists of STATE_W x STATE_H pixels.
# Reward is -0.1 every frame and +1000/N for every track tile visited, where N is
# the total number of tiles visited in the track. For example, if you have finished in 732 frames,
# your reward is 1000 - 0.1*732 = 926.8 points.
# Game is solved when agent consistently gets 900+ points. Track generated is random every episode.
# Episode finishes when all tiles are visited. Car also can go outside of PLAYFIELD, that
# is far off the track, then it will get -100 and die.
# Some indicators shown at the bottom of the window and the state RGB buffer. From
# left to right: true speed, four ABS sensors, steering wheel position and gyroscope.
# To play yourself (it's rather fast for humans), type:
# python gym/envs/box2d/car_racing.py
# Remember it's powerful rear-wheel drive car, don't press accelerator and turn at the
# same time.
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

STATE_W = 96   # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE       = 6.0        # Track scale
TRACK_RAD   = 900/SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD   = 2000/SCALE # Game over boundary
FPS         = 50         # Frames per second
ZOOM        = 2.7        # Camera zoom
ZOOM_FOLLOW = True       # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21/SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40/SCALE # changed to 55 form 40 mhopk1
BORDER = 8/SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]

class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        self._contact(contact, True)
    def EndContact(self, contact):
        self._contact(contact, False)
    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj  = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj  = u1
        if not tile:
            return

        tile.color[0] = ROAD_COLOR[0]
        tile.color[1] = ROAD_COLOR[1]
        tile.color[2] = ROAD_COLOR[2]
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            # print tile.road_friction, "ADD", len(obj.tiles)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0/len(self.env.track)
                self.env.tile_visited_count += 1
        else:
            obj.tiles.remove(tile)

class CarRacing(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second' : FPS
    }

    def __init__(self, verbose=1):
        EzPickle.__init__(self)
        self.seed()
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0,0), contactListener=self.contactListener_keepref)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.fd_tile = fixtureDef(
                shape = polygonShape(vertices=
                    [(0, 0),(1, 0),(1, -1),(0, -1)]))

        self.action_space = spaces.Box( np.array([-1,0,0]), np.array([+1,+1,+1]), dtype=np.float32)  # steer, gas, brake
        self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)
        
        # PLATOONING
        # added variables
        self.track_directions = [] # list of the direction vectors at each point in the track
        self.lead_trajectory = []
        self.closest_tile = None # closest track tile to the following vehicle
        self.furthest_tile = None
        self.lead_offset = 0
        self.lead_max_offset = 5
        self.lead_max_lateral = 1 # the max lateral speed of the lead vehicle
        self.lead_tile = 0
        self.lead_steps = 2900
        self.lead_pos = self.lead_prev_pos = None
        self.lead_velocity = 0
        


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        self.car.destroy()

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpointsv
        checkpoints = []
        for c in range(CHECKPOINTS):
            alpha = 2*math.pi*c/CHECKPOINTS + self.np_random.uniform(0, 2*math.pi*1/CHECKPOINTS)
            rad = self.np_random.uniform(TRACK_RAD/3, TRACK_RAD)
            if c==0:
                alpha = 0
                rad = 1.5*TRACK_RAD
            if c==CHECKPOINTS-1:
                alpha = 2*math.pi*c/CHECKPOINTS
                self.start_alpha = 2*math.pi*(-0.5)/CHECKPOINTS
                rad = 1.5*TRACK_RAD
            checkpoints.append( (alpha, rad*math.cos(alpha), rad*math.sin(alpha)) )

        # print "\n".join(str(h) for h in checkpoints)
        # self.road_poly = [ (    # uncomment this to see checkpoints
        #    [ (tx,ty) for a,tx,ty in checkpoints ],
        #    (0.7,0.7,0.9) ) ]
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5*TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2*math.pi
            while True: # Find destination from checkpoints
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break
                if not failed:
                    break
                alpha -= 2*math.pi
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            proj = r1x*dest_dx + r1y*dest_dy  # destination vector projected on rad
            while beta - alpha >  1.5*math.pi:
                 beta -= 2*math.pi
            while beta - alpha < -1.5*math.pi:
                 beta += 2*math.pi
            prev_beta = beta
            proj *= SCALE
            if proj >  0.3:
                 beta -= min(TRACK_TURN_RATE, abs(0.001*proj))
            if proj < -0.3:
                 beta += min(TRACK_TURN_RATE, abs(0.001*proj))
            x += p1x*TRACK_DETAIL_STEP
            y += p1y*TRACK_DETAIL_STEP
            track.append( (alpha,prev_beta*0.5 + beta*0.5,x,y) )
            if laps > 4:
                 break
            no_freeze -= 1
            if no_freeze==0:
                 break
        # print "\n".join([str(t) for t in enumerate(track)])

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i==0:
                return False  # Failed
            pass_through_start = track[i][0] > self.start_alpha and track[i-1][0] <= self.start_alpha
            if pass_through_start and i2==-1:
                i2 = i
            elif pass_through_start and i1==-1:
                i1 = i
                break
        if self.verbose == 1:
            pass
            #print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2-i1))
        assert i1!=-1
        assert i2!=-1

        track = track[i1:i2-1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square( first_perp_x*(track[0][2] - track[-1][2]) ) +
            np.square( first_perp_y*(track[0][3] - track[-1][3]) ))
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False]*len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i-neg-0][1]
                beta2 = track[i-neg-1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE*0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i-neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i-1]
            road1_l = (x1 - TRACK_WIDTH*math.cos(beta1), y1 - TRACK_WIDTH*math.sin(beta1))
            road1_r = (x1 + TRACK_WIDTH*math.cos(beta1), y1 + TRACK_WIDTH*math.sin(beta1))
            road2_l = (x2 - TRACK_WIDTH*math.cos(beta2), y2 - TRACK_WIDTH*math.sin(beta2))
            road2_r = (x2 + TRACK_WIDTH*math.cos(beta2), y2 + TRACK_WIDTH*math.sin(beta2))
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01*(i%3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(( [road1_l, road1_r, road2_r, road2_l], t.color ))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (x1 + side* TRACK_WIDTH        *math.cos(beta1), y1 + side* TRACK_WIDTH        *math.sin(beta1))
                b1_r = (x1 + side*(TRACK_WIDTH+BORDER)*math.cos(beta1), y1 + side*(TRACK_WIDTH+BORDER)*math.sin(beta1))
                b2_l = (x2 + side* TRACK_WIDTH        *math.cos(beta2), y2 + side* TRACK_WIDTH        *math.sin(beta2))
                b2_r = (x2 + side*(TRACK_WIDTH+BORDER)*math.cos(beta2), y2 + side*(TRACK_WIDTH+BORDER)*math.sin(beta2))
                self.road_poly.append(( [b1_l, b1_r, b2_r, b2_l], (1,1,1) if i%2==0 else (1,0,0) ))
        self.track = track
        
        # storing the track directions
        for i in range(len(track) - 1):
            a,b,x1,y1 = track[i]
            a,b,x2,y2 = track[i+1]
            dir_vector = math.atan2(y2-y1,x2-x1) % 6.28
            self.track_directions.append(dir_vector)
        
        
        
        
        return True

    def reset(self):
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []
        
        while True:
            # mhopk1: working to create the track (INCLUDES SOME RANDOM ASPECTS)
            success = self._create_track()
            if success:
                break
            if self.verbose == 1:
                pass
                #print("retry to generate track (normal if there are not many of this messages)")
        self.car = Car(self.world, *self.track[0][1:4])
        
        # PLATOON CAR
        self.closest_tile = None
        self.lead_tile = 0
        self.lead_pos = None

        return self.step(None)[0]

    def step(self, action):
        """ controlling the vehicle through inputs
        setting the reward and determining when the 
        simulation is over
        """
        # processing the control inputs
        if action is not None:
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])

        # processing the world physics
        self.car.step(1.0/FPS)
        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.t += 1.0/FPS

        self.state = self.render("state_pixels")

        step_reward = 0
        done = False
        if action is not None: # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            # new simulation finsihed condition
            if (self.lead_tile == len(self.track) -1):
                done = True
            # dont care about tiles visited in this environment
            #if self.tile_visited_count==len(self.track):
            #    done = True
            
#            if ((self.lead_tile > self.closest_tile) and (self.lead_tile < self.furthest_tile)):
#                 self.reward += 1 * (1000)/self.tile_180


        #------------ Simulation Stopping condition ---------------------------#
#            if (self.lead_tile == len(self.track) -1 
#                or self.track_directions[min(len(self.track),self.closest_tile + 2)] > math.radians(178)
#                or self.track_directions[min(len(self.track),self.closest_tile + 2)] < 0):
#                done = True
#                print("No of tiles covered :" + str(self.closest_tile))
#                print("Track length : "+str(len(self.track)))
            if (self.closest_tile > len(self.track) - 3):
                done = True    
        #----------------------------------------------------------------------#    
            
        #----------- Inter-vehicular distance reward function------------------#

#            if ((self.lead_tile -self.closest_tile) < 7 and (self.lead_tile -self.closest_tile) > 4):
#                self.reward += 5
#            else:
#                self.reward += 0
        #----------------------------------------------------------------------#
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward = -100

        # --- PLATOON CAR ---
        # finding the closest tile to the following vehicle
        dist = inf
        prev_dist = inf
        if self.closest_tile == None:
            search_range = range(len(self.track))
        else:
            search_range = range(max(0, self.closest_tile - 10), min(len(self.track),self.closest_tile + 10))
        for i in search_range:
            a,b,x,y = self.track[i]
            dist = np.linalg.norm([x - self.car.hull.position.x, y - self.car.hull.position.y])
            if (dist > prev_dist):
                self.closest_tile = i
                break
            prev_dist = dist
        
        # finding the furthest tile
        if (self.closest_tile != None):
            max_x = self.car.hull.position[0] + STATE_W/2
            max_y = self.car.hull.position[1] + STATE_H/2
            min_x = self.car.hull.position[0] - STATE_W/2
            min_y = self.car.hull.position[1] - STATE_H/2
            for i in range(self.closest_tile,len(self.track)):
                a,b,x,y = self.track[i]
                if (((x<min_x) or (x>max_x)) or ((y<min_y) or (y>max_y))):
                    self.furthest_tile = i - 4
                    break
                
        # EDITED: calculating the lead position
        if self.lead_pos == None:
            a,b,x,y = self.track[self.lead_tile]
            self.lead_pos = [x,y]
            self.lead_prev_pos = self.lead_pos
        else: 
            self.lead_tile += len(self.track) / (self.lead_steps)
            self.lead_tile = min(len(self.track)-1,max(0,self.lead_tile))
            base_tile = min(len(self.track)-1,math.floor(self.lead_tile))
            next_tile = min(len(self.track)-1,math.ceil(self.lead_tile))
            offset_dir = self.track_directions[base_tile-1] + math.radians(90)
            a,b,x1,y1 = self.track[base_tile]
            a,b,x2,y2 = self.track[next_tile] 
            pos_diff = (x2 - self.lead_pos[0], y2 - self.lead_pos[1])
            # adding to the lead position
            self.lead_pos[0] = self.lead_pos[0] + pos_diff[0]
            self.lead_pos[1] = self.lead_pos[1] + pos_diff[1]
            # off-set
            self.lead_offset = self.lead_offset + (np.random.rand()*2 - 1) * self.lead_max_lateral
            self.lead_offset = max(-self.lead_max_offset,min(self.lead_max_offset,self.lead_offset))
            # off-set from the center lane
            self.lead_pos[0] = self.lead_pos[0] + self.lead_offset * math.cos(offset_dir)
            self.lead_pos[1] = self.lead_pos[1] + self.lead_offset * math.sin(offset_dir) 
            self.lead_velocity = (self.lead_pos[0] - self.lead_prev_pos[0], self.lead_pos[1] - self.lead_prev_pos[1])  
            self.lead_prev_pos = (self.lead_pos[0], self.lead_pos[1])     
        
        #print("closest tile:{} furthest tile:{}\n lead_pos:{} car position:{}".format(self.closest_tile, self.furthest_tile,self.lead_pos,
              #[self.car.hull.position[0],self.car.hull.position[1]]))
    
        # returning the information needed for classical control cc_vector
        # tangential/lateral error (vector projection a (distance from center lane) on b(tangential direction to road curvature) )
        a = np.array([self.car.hull.position[0] - self.track[self.closest_tile][2], self.car.hull.position[1] - self.track[self.closest_tile][3]])
        b = np.array([math.cos(self.track_directions[self.closest_tile-1] + math.radians(90)),
                       math.sin(self.track_directions[self.closest_tile-1] + math.radians(90))])
        tangent_error = np.dot(a,b)
        road_curvature = [ (x[2],x[3]) for x in self.track[self.closest_tile:self.furthest_tile] ]
        cc_vector = [self.track_directions[self.closest_tile - 1] - (self.car.hull.angle + math.radians(90)),
                     tangent_error,
                     road_curvature,
                     (self.lead_pos[0],self.lead_pos[1]),
                     self.lead_velocity,
                     (self.car.hull.position[0],self.car.hull.position[1])
                     ]
#        
#        if (abs(tangent_error) > 8):
#            self.reward -= (abs(tangent_error) - 8) * 200/ self.tile_180
        
        #print(cc_vector[0])
        return self.state, step_reward, done, {}, cc_vector

    def render(self, mode='human'):
        # draws everything that we see in the environment
        assert mode in ['human', 'state_pixels', 'rgb_array']
        if self.viewer is None:
            # creating the view
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label('0000', font_size=36,
                x=20, y=WINDOW_H*2.5/40.00, anchor_x='left', anchor_y='center',
                color=(255,255,255,255))
            self.transform = rendering.Transform()

        if "t" not in self.__dict__: return  # reset() not called yet

        zoom = 0.1*SCALE*max(1-self.t, 0) + ZOOM*SCALE*min(self.t, 1)   # Animate zoom first second
        zoom_state  = ZOOM*SCALE*STATE_W/WINDOW_W
        zoom_video  = ZOOM*SCALE*VIDEO_W/WINDOW_W
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W/2 - (scroll_x*zoom*math.cos(angle) - scroll_y*zoom*math.sin(angle)),
            WINDOW_H/4 - (scroll_x*zoom*math.sin(angle) + scroll_y*zoom*math.cos(angle)) )
        self.transform.set_rotation(angle)

        self.car.draw(self.viewer, mode!="state_pixels")

        

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode=='rgb_array':
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == 'state_pixels':
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, '_nscontext'):
                pixel_scale = win.context._nscontext.view().backingScaleFactor()  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        # the simulation view port
        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable() # have to draw after this enable
        self.render_road()
        
        # --- PLATOON CAR additions ---
        draw_track_waypoints(self.car,self.track,self.closest_tile, self.furthest_tile) #mhopk1
        draw_lead_vehicle(self.lead_pos)# mhopk1
        draw_vehicle_debug(self)
        #
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        # mhopk1: the HUD once t is disabled
        self.render_indicators(WINDOW_W, WINDOW_H)
        
        if mode == 'human':
            win.flip()
            return self.viewer.isopen

        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        try:
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            arr = arr.reshape(VP_H, VP_W, 4)
            arr = arr[::-1, :, 0:3]
        except:
            print("'ImageData' object has no attribute 'data'")
        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render_road(self):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.4, 0.8, 0.4, 1.0)
        gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)
        gl.glColor4f(0.4, 0.9, 0.4, 1.0)
        k = PLAYFIELD/20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                gl.glVertex3f(k*x + k, k*y + 0, 0)
                gl.glVertex3f(k*x + 0, k*y + 0, 0)
                gl.glVertex3f(k*x + 0, k*y + k, 0)
                gl.glVertex3f(k*x + k, k*y + k, 0)
        for poly, color in self.road_poly:
            gl.glColor4f(color[0], color[1], color[2], 1)
            for p in poly:
                gl.glVertex3f(p[0], p[1], 0)
        gl.glEnd()

    def render_indicators(self, W, H):
        gl.glBegin(gl.GL_QUADS)
        s = W/40.0
        h = H/40.0
        gl.glColor4f(0,0,0,1)
        gl.glVertex3f(W, 0, 0)
        gl.glVertex3f(W, 5*h, 0)
        gl.glVertex3f(0, 5*h, 0)
        gl.glVertex3f(0, 0, 0)
        def vertical_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place+0)*s, h + h*val, 0)
            gl.glVertex3f((place+1)*s, h + h*val, 0)
            gl.glVertex3f((place+1)*s, h, 0)
            gl.glVertex3f((place+0)*s, h, 0)
        def horiz_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place+0)*s, 4*h , 0)
            gl.glVertex3f((place+val)*s, 4*h, 0)
            gl.glVertex3f((place+val)*s, 2*h, 0)
            gl.glVertex3f((place+0)*s, 2*h, 0)
        true_speed = np.sqrt(np.square(self.car.hull.linearVelocity[0]) + np.square(self.car.hull.linearVelocity[1]))
        vertical_ind(5, 0.02*true_speed, (1,1,1))
        vertical_ind(7, 0.01*self.car.wheels[0].omega, (0.0,0,1)) # ABS sensors
        vertical_ind(8, 0.01*self.car.wheels[1].omega, (0.0,0,1))
        vertical_ind(9, 0.01*self.car.wheels[2].omega, (0.2,0,1))
        vertical_ind(10,0.01*self.car.wheels[3].omega, (0.2,0,1))
        horiz_ind(20, -10.0*self.car.wheels[0].joint.angle, (0,1,0))
        horiz_ind(30, -0.8*self.car.hull.angularVelocity, (1,0,0))
        gl.glEnd()
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()

def imm(W_H_1,W_H_2,W_I_1,W_I_2,W_O_1,W_O_2):
    zeros = np.zeros([3,3])
    A_1 = np.concatenate((zeros,W_H_2,zeros,zeros,zeros,zeros,W_I_2*W_O_1,zeros,zeros,zeros,zeros,zeros),axis=1)
    A_2 = np.concatenate((zeros,zeros,W_H_2,zeros,zeros,zeros,zeros,W_I_2*W_O_1,zeros,zeros,zeros,zeros),axis=1)
    A_3 = np.concatenate((zeros,zeros,zeros,W_H_2,zeros,zeros,zeros,zeros,W_I_2*W_O_1,zeros,zeros,zeros),axis=1)
    A_4 = np.concatenate((zeros,zeros,zeros,zeros,W_H_2,zeros,zeros,zeros,zeros,W_I_2*W_O_1,zeros,zeros),axis=1)
    A_5 = np.concatenate((zeros,zeros,zeros,zeros,zeros,W_H_2,zeros,zeros,zeros,zeros,W_I_2*W_O_1,zeros),axis=1)
    A_6 = np.concatenate((zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,W_I_2*W_O_1),axis=1)
    A_7 = np.concatenate((zeros,zeros,zeros,zeros,zeros,zeros,zeros,W_H_1,zeros,zeros,zeros,zeros),axis=1)
    A_8 = np.concatenate((zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,W_H_1,zeros,zeros,zeros),axis=1)
    A_9 = np.concatenate((zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,W_H_1,zeros,zeros),axis=1)
    A_10 = np.concatenate((zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,W_H_1,zeros),axis=1)
    A_11 = np.concatenate((zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,W_H_1),axis=1)
    A_12 = np.concatenate((zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros),axis=1)
    A = np.concatenate((A_1,A_2,A_3,A_4,A_5,A_6,A_7,A_8,A_9,A_10,A_11,A_12))
    zeros = np.zeros([3,1])
    B_1 = np.concatenate((-W_I_2,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros),axis=1)
    B_2 = np.concatenate((zeros,-W_I_2,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros),axis=1)
    B_3 = np.concatenate((zeros,zeros,-W_I_2,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros),axis=1)
    B_4 = np.concatenate((zeros,zeros,zeros,-W_I_2,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros),axis=1)
    B_5 = np.concatenate((zeros,zeros,zeros,zeros,-W_I_2,zeros,zeros,zeros,zeros,zeros,zeros,zeros),axis=1)
    B_6 = np.concatenate((zeros,zeros,zeros,zeros,zeros,-W_I_2,zeros,zeros,zeros,zeros,zeros,zeros),axis=1)
    B_7 = np.concatenate((zeros,zeros,zeros,zeros,zeros,zeros,-W_I_1,zeros,zeros,zeros,zeros,zeros),axis=1)
    B_8 = np.concatenate((zeros,zeros,zeros,zeros,zeros,zeros,zeros,-W_I_1,zeros,zeros,zeros,zeros),axis=1)
    B_9 = np.concatenate((zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,-W_I_1,zeros,zeros,zeros),axis=1)
    B_10 = np.concatenate((zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,-W_I_1,zeros,zeros),axis=1)
    B_11 = np.concatenate((zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,-W_I_1,zeros),axis=1)
    B_12 = np.concatenate((zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,-W_I_1),axis=1)
    B = np.concatenate((B_1,B_2,B_3,B_4,B_5,B_6,B_7,B_8,B_9,B_10,B_11,B_12))
    zeros = np.zeros((1,3))
    C_1 = np.concatenate((W_O_2,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros),axis=1)
    C_2 = np.concatenate((zeros,W_O_2,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros),axis=1)
    C_3 = np.concatenate((zeros,zeros,W_O_2,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros),axis=1)
    C_4 = np.concatenate((zeros,zeros,zeros,W_O_2,zeros,zeros,zeros,zeros,zeros,zeros,zeros,zeros),axis=1)
    C_5 = np.concatenate((zeros,zeros,zeros,zeros,W_O_2,zeros,zeros,zeros,zeros,zeros,zeros,zeros),axis=1)
    C_6 = np.concatenate((zeros,zeros,zeros,zeros,zeros,W_O_2,zeros,zeros,zeros,zeros,zeros,zeros),axis=1)
    C = np.concatenate((C_1,C_2,C_3,C_4,C_5,C_6))
    D = np.zeros([6,12])
    model = ImplicitLayer(36, 1, 12, 6)
    model.A.data.copy_(torch.from_numpy(A))
    model.B.data.copy_(torch.from_numpy(B))
    model.C.data.copy_(torch.from_numpy(C))
    model.D.data.copy_(torch.from_numpy(D))    
    return model

def within_border(max_x, min_x, max_y, min_y, point):
    # point is a tuple
    x = point[0]
    y = point[1]
    if (((x<min_x) or (x>max_x)) or ((y<min_y) or (y>max_y))):
        return False
    else:
        return True

def draw_vehicle_debug(my_env):
    # mhopk1: should be displayed in the right view port
    if (my_env.closest_tile != None):
        a = np.array([my_env.car.hull.position[0] - my_env.track[my_env.closest_tile][2], 
                      my_env.car.hull.position[1] - my_env.track[my_env.closest_tile][3]])
        b = np.array([math.cos(my_env.track_directions[my_env.closest_tile] + math.radians(90)),
        math.sin(my_env.track_directions[my_env.closest_tile] + math.radians(90))])
        tangent_error = np.dot(a,b)
        
        gl.glBegin(gl.GL_LINES)
        gl.glColor4f(2, 0.8, 0.79, .8)
#         gl.glVertex3f(point[0]-1, point[1], 0)
#         gl.glVertex3f(point[0], point[1]+1, 0)
#         gl.glVertex3f(point[0]+1, point[1], 0)
#         gl.glVertex3f(point[0], point[1]-1, 0)
        gl.glVertex3f(my_env.car.hull.position.x, my_env.car.hull.position.y, 0)
        gl.glVertex3f(my_env.car.hull.position.x + math.cos(my_env.car.hull.angle) * tangent_error,
                      my_env.car.hull.position.y + math.sin(my_env.car.hull.angle) * tangent_error, 0)
        gl.glEnd()
        
def draw_track_waypoints(carobj, track, closest_tile = None, furthest_tile = None, directions = None):
    # draws the track waypoints and their tangential directions
    # - closest_tile: whether or not we display the closest tile to the vehicle
    # - furthest_tile: whether or not we display the furthest tile from the vehicle
    # - directions: whether or not we display the tangential directions
    # mhopk1: should be displayed in the right view port
    max_x = carobj.hull.position[0] + STATE_W/2
    max_y = carobj.hull.position[1] + STATE_H/2
    min_x = carobj.hull.position[0] - STATE_W/2
    min_y = carobj.hull.position[1] - STATE_H/2
                
                
    if (directions != None):
        for i in range(len(track)-1):
            gl.glColor4f(0, 0, 0, 1.0)
            gl.glBegin(gl.GL_LINES)
            a,b,x,y = track[i]
            gl.glVertex3f(x,y,0)
            gl.glVertex3f(x+2*math.cos(directions[i]),y+2*math.sin(directions[i]),0)
            gl.glEnd()
        
    for i in range(len(track)):
        gl.glBegin(gl.GL_TRIANGLE_FAN)
        a,b,x,y = track[i]
        gl.glColor4f(0.1, 0.1, 0.8, 1.0)
        # draw the tile as a different color if it is the closest tile
        if (closest_tile != None):
            if (i == closest_tile):
                gl.glColor4f(1,1,1,1)
        if (furthest_tile != None):
            if (i == furthest_tile):
                gl.glColor4f(0,0,0,1)
        #gl.glVertex3f(x-1, y, 0)
        #gl.glVertex3f(x, y+1, 0)
        #gl.glVertex3f(x+1, y, 0)
        #gl.glVertex3f(x, y-1, 0)
        if ( within_border(max_x,min_x,max_y,min_y,(x,y)) ):
            for r in range(0,360,90):
                gl.glVertex3f(x + 1 * math.cos(math.radians(r)), y + 1 * math.sin(math.radians(r)), 0)
        gl.glEnd()

def draw_obstacles(obstacles):
    # mhopk1: drawing the obstacles
    pass

def draw_lead_vehicle(point):
    # mhopk1: should be displayed in the right view port
    if (point != None):
        gl.glBegin(gl.GL_TRIANGLE_FAN)
        gl.glColor4f(1, 0.1, 0.79, .8)
        for r in range(0,360,45):
            gl.glVertex3f(point[0] + 5 * math.cos(math.radians(r)), point[1] + 5 * math.sin(math.radians(r)), 0)
        gl.glEnd()
    
def draw_trajectory(traj_points,lead_pos):
    # mhopk1: draws the trajectory of the lead vehicle and its current position
    pass    

def dist_on_circle(ang,ang2):
    # angles are in degrees
    distx = math.cos(ang) - math.cos(ang2)
    disty = math.sin(ang) - math.sin(ang2)
    return math.sqrt(distx ** 2 + disty ** 2)            

    
    
if __name__=="__main__":
    
    f = open("Reward_ata.txt","a")
    f1 = open("Vehicle_angle.txt", "a")
    f2 = open("Track_angle.txt","a")
    rst = True
    rst2 = True
    reward_log = []
    for i in range(1):
        from pyglet.window import key
        a = np.array( [0.0, 0.0, 0.0] )
        def key_press(k, mod):
            global restart
            if k==0xff0d: restart = True
            if k==key.LEFT:  a[0] = -1.0
            if k==key.RIGHT: a[0] = +1.0
            if k==key.UP:    a[1] = +1.0
            if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
        def key_release(k, mod):
            if k==key.LEFT  and a[0]==-1.0: a[0] = 0
            if k==key.RIGHT and a[0]==+1.0: a[0] = 0
            if k==key.UP:    a[1] = 0
            if k==key.DOWN:  a[2] = 0
             
        primary_pid = PID(3, 0.01, 0.05, setpoint = 0) # lane keep
        secondary_pid = PID(3, 0.01, 0.05, setpoint = 0)  # angle error
        # controller plots
        fig = plt.figure()
        ax = fig.gca()    
        i=0
        prev=0
        k=3
        t1f = 0
        
        
        data_steer = []
        data_angle = []
        data_track_angle = []    
        data_throttle = []
        data_brake = []
        data_velocity = []
        data_pid_error = []
        prev_pos = [None,None]
        
        
        env = CarRacing()
        env.render()
        env.viewer.window.on_key_press = key_press
        env.viewer.window.on_key_release = key_release
        record_video = False
        if record_video:
            from gym.wrappers.monitor import Monitor
            env = Monitor(env, '/tmp/video-test', force=True)
        isopen = True
        
        # PID control variables
        control_angle = 0
        control_lat = 0
        yaw_rate = 0
        yaw_prev = 0
        
        
        loss_fn = F.mse_loss
        
        #IM
        W_H_1 = np.zeros((3,3))
        W_H_1[2,0] = -1
        W_H_1[1,1] = 1
        W_H_2 = np.zeros((3,3))
        W_H_2[2,0] = -1
        W_H_2[1,1] = 1
        W_I_1 = np.ones((3,1))
        W_I_2 = np.ones((3,1))
        W_O_1 = np.ones((1,3))
        W_O_1[0,0] = 0.35
        W_O_1[0,1] = 0.45
        W_O_1[0,2] = 4.25
        W_O_2 = np.ones((1,3))
        W_O_2[0,0] = 0.2
        W_O_2[0,1] = 0.45
        W_O_2[0,2] = 0.15
        im = imm(W_H_1,W_H_2,W_I_1,W_I_2,W_O_1,W_O_2)
        lr = 1e-2
        opti_a = optim.Adam(im.parameters(),lr=lr)
        pre_list = []
        loss_list = []
        for epoch in range(1):
            print("###### RESETTING ########")
            
            env.reset()
            total_reward = 0.0
            steps = 0
            a[0]=-1
            a[1]=0
            a[2]=0
            control_angle = 0
    #        pid = PID(2.8, 6.8,0.26, setpoint=0)
            restart = False
            #prev_aerror=0
            cnt=0
            error_list = []
            tangent_error_list = []
            dist_error_list = []
            a_list = []
            x_input_list = []
            for iteration in range(1000):
                cnt+=1
                s, r, done, info, cc = env.step(a)
                total_reward += r
                if steps == 0 or done:
                    pass
                yaw_rate = env.car.hull.angle - yaw_prev
                dist_dif = np.array([env.car.hull.position[0] - env.track[env.closest_tile][2], 
                      env.car.hull.position[1] - env.track[env.closest_tile][3]])
                tangent_dir = np.array([math.cos(env.track_directions[env.closest_tile] + math.radians(90)),
                              math.sin(env.track_directions[env.closest_tile] + math.radians(90))])
                tangent_error = np.dot(dist_dif, tangent_dir)
                error_sign = np.sign(math.sin(cc[0]))
                error_magnitude = dist_on_circle((env.car.hull.angle + math.radians(90)) % 6.28, env.track_directions[env.closest_tile - 1]) * 6.28   
                dist_error = error_sign*error_magnitude
                tangent_error_list.append(tangent_error)
                dist_error_list.append(dist_error)
                    
                yaw_rate = env.car.hull.angle - yaw_prev
                if len(tangent_error_list)>6:
                    x_input=[tangent_error_list[len(tangent_error_list)-1],tangent_error_list[len(tangent_error_list)-2],tangent_error_list[len(tangent_error_list)-3],tangent_error_list[len(tangent_error_list)-4],tangent_error_list[len(tangent_error_list)-5]]
                    #x_input=[dist_error_list[len(dist_error_list)-1],dist_error_list[len(dist_error_list)-2],dist_error_list[len(dist_error_list)-3],dist_error_list[len(dist_error_list)-4],dist_error_list[len(dist_error_list)-5]]
                    x_input.append(1)
                    x_input.extend([dist_error_list[len(dist_error_list)-1],dist_error_list[len(dist_error_list)-2],dist_error_list[len(dist_error_list)-3],dist_error_list[len(dist_error_list)-4],dist_error_list[len(dist_error_list)-5]])
                    #x_input.extend([tangent_error_list[len(tangent_error_list)-1],tangent_error_list[len(tangent_error_list)-2],tangent_error_list[len(tangent_error_list)-3],tangent_error_list[len(tangent_error_list)-4],tangent_error_list[len(tangent_error_list)-5]])
                    x_input.append(1)
                    x_input = torch.tensor(x_input)
                    x_input = torch.reshape(x_input,(1,12))
                    x_input = x_input.float()
                    x_input = x_input
                    h_0_1 = im.X0[30:33]
                    h_0_2 = im.X0[12:15]
                    B = im.B.data
                    B[33:,11] = torch.reshape(h_0_1,(1,3))
                    B[15:18,5] =  torch.reshape(h_0_2,(1,3))
                    im.B.data.copy_(B)
                    #x_input = x_input/100
                    a[0] = float(im(x_input)[0][0])

                if cnt >10 and cnt < 20:
                    a[1] = 1 #control_velocity
                else:
                    a[1] = 0
                t1 = time.monotonic()
                
                isopen = env.render()

                if done or restart or isopen == False or rst2 == False:
                    #print("--------finished scenario-------")
                    print("Reward:",total_reward)
                    reward_log.append(total_reward)
                    print("Reward mean:",np.mean(np.array(reward_log)))
                    print("Reward std:",np.std(np.array(reward_log)))
                    print("Reward max:",np.amax(np.array(reward_log)))
                    print("Reward min:",np.amin(np.array(reward_log)))
                    print("trials",len(reward_log))
                    rst = False
                    break
        env.close()