# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:57:14 2019

@author: Yue
"""

##!/usr/bin/env python 
## -*- coding: utf-8 -*- 
#import sys
#origin = sys.stdout
#f = open('multi_agent_more_layer_1000.txt', 'w')
#sys.stdout = f
## ===================================
#print('Start of program')
#
## above code put all printing in a .txt file

import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
from tkinter import colorchooser
import threading
import numpy as np
import math
import pandas as pd
import sys
import io
import os
from PIL import Image, ImageTk
import tkinter.messagebox 
import time
import random
import keras
from Experience_normal import Experience
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
from keras.layers.advanced_activations import ReLU
import datetime


class App:
    def __init__(self, master):
        self.master = master
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)        
        self.width = IntVar()
        self.width.set(1)
        self.outline = 'black'
        self.fill = None
        self.mv_beginX = self.mv_beginY = -10
        self.firstx = self.firsty = -10
        self.mv_prevx = self.mv_prevy = -10
        self.item_type = 0
        self.episode = 1000
        self.learning = False
        self.episode_end = False
        self.grid_origx = 500
        self.grid_origy = 20
        self.grid_columnNum = 16
        self.grid_rowNum = 16
        self.grid_UNIT = 50
        self.itemsNum = 0
        self.timeDelay = 1
        self.epsilon = 0.9
        self.Qtable_origx = self.grid_origx+20+(self.grid_columnNum+1)*self.grid_UNIT
        self.Qtable_origy = self.grid_origy
        self.grid_origx_center = self.grid_origx+self.grid_UNIT/2
        self.grid_origy_center = self.grid_origy+self.grid_UNIT/2
        self.Qtable_gridIndex_dict = {}
        self.show_q_table = pd.DataFrame(columns=list(range(self.n_actions)), dtype=np.float64)
                
        self.origDist = 10
        self.agentCentre = np.array([[190,180],[290,180],[390,180]])
        self.warehouseCentre = np.array([[625,195],[875,345],[1125,495]])
#        self.warehouseCentre = self.agentCentre+np.array([[0,self.grid_UNIT+self.origDist],\
#                                [0,self.grid_UNIT+self.origDist],[0,self.grid_UNIT+self.origDist]])
        self.ObstacleCentre = np.array([[725, 245], [1175, 45], [1025, 95], [1275, 445], [1275, 645], [625, 445], [1025, 595], [625, 595], [575, 795], [575, 795], [1225, 745], [975, 195], [575, 645], [925, 45], [1175, 95], [1075, 645], [1175, 745], [975, 695], [975, 645], [1175, 745], [1275, 495], [625, 95], [875, 245], [825, 395], [625, 595], [525, 195], [925, 145], [1075, 45], [1025, 245], [675, 645]])        
        self.itemOrigPosition = []    
        self.agentPosition_list = []
        self.warehousePostition_list = []
        self.ObstaclePosition_list = []        
        self.WarehouseItemIndex = []
        self.agentItemIndex = []
        self.ObstacleItemIndex = []
        self.AllItemsOrigPosition_list = []        
        self.createMark = None
        self.points = []
        self.cars_list = []
        self.selected_agent = []
        self.selected_agent_position = []
        self.selected_Obstacles_position = []        
        self.selected_Obstacles = []
        self.selected_targets = []
        self.agent = 1
        self.target = 4
        self.hell1 = 7
        self.hell2 = 8
        self.init_widgets()
        self.temp_item = None
        self.temp_items = []
        self.choose_item = None                
        self.grid_endx = self.grid_origx+self.grid_columnNum*self.grid_UNIT
        self.grid_endy = self.grid_origy+self.grid_rowNum*self.grid_UNIT        
        self.created_line = []
        self.createMarkA = None
        self.createMarkB = None
        self.createMarkC = None
        self.linesA = []
        self.linesB = []
        self.linesC = []
        self.busyA = 0
        self.busyB = 0
        self.busyC = 0        
        self.maze_size = self.grid_columnNum*self.grid_rowNum
        self.max_memory_size = 8*self.maze_size
        self.num_actions = 4
        self.epochs = 8000
        self.data_size = 32
        self.tests = 100        
        
    def resize(self,w, h, w_box, h_box, pil_image):  
      ''''' 
      resize an image
      '''  
      return pil_image.resize((w_box, h_box), Image.ANTIALIAS)

    def init_widgets(self):
        
        self.cv = Canvas(root, background='white')
        self.cv.pack(fill=BOTH, expand=True)
        self.cv.bind('<B1-Motion>', self.move)
        self.cv.bind('<ButtonRelease-1>', self.move_end)
        self.cv.bind("<Button-1>", self.leftClick_handler)
        self.cv.bind("<Button-3>", self.rightClick_handler)
        f = ttk.Frame(self.master)
        f.pack(fill=X)
        self.bns = []
        for i, lb in enumerate(('Reset', 'Start trainning', 'Close', 'Save', 'Start Running','Restart','Terminal C','Terminal B','Terminal A')):
            bn = Button(f, text=lb, command=lambda i=i: self.choose_type(i))
            if i>5:
                bn.pack(side=RIGHT, ipadx=8,ipady=5, padx=5)
            else:
                bn.pack(side=LEFT, ipadx=8,ipady=5, padx=5)
            self.bns.append(bn)
        self.bns[self.item_type]['relief'] = SUNKEN

        self.agentPosition_list = self.setItemsPositionList(self.agentCentre)
        self.warehousePostition_list = self.setItemsPositionList(self.warehouseCentre)
        self.ObstaclePosition_list = self.setItemsPositionList(self.ObstacleCentre)

        self.create_items()
        self.itemsNum = self.warehouseCentre.shape[0]+self.ObstacleCentre.shape[0]+self.agentCentre.shape[0]
 
        R = self.grid_UNIT
        self.cv.create_text(self.agentCentre[0][0]-R-20,self.agentCentre[0][1],\
                            text = "Agent:",font=('Courier',18))
        self.create_grids(self.grid_origx,self.grid_origy,self.grid_columnNum,self.grid_rowNum,self.grid_UNIT)
        self.entryCd = Entry(self.cv,text = "Please input Epsilion!")
        self.entryCd.place(x=self.agentCentre[0][0]-80, y=self.agentCentre[0][1]+500, anchor=NW)
        originX = self.grid_origx+self.grid_UNIT/2
        originY = self.grid_origy+self.grid_UNIT/2
        origin = np.array([originX, originY])        
        for i in range(0,self.grid_rowNum):
            for j in range(0, self.grid_columnNum):
                x = i*self.grid_UNIT+self.grid_origx_center
                y = j*self.grid_UNIT+self.grid_origy_center
                rowIndex = (y-self.grid_origy_center)/self.grid_UNIT
                columnIndex = (x-self.grid_origx_center)/self.grid_UNIT
                self.Qtable_gridIndex_dict[(x,y)]= int(rowIndex*self.grid_columnNum+columnIndex)
                
        print(self.Qtable_gridIndex_dict)

    def create_ObsItems(self):
        self.cv.arriveObsImage = []
        self.cv.bms_obs = []
        w_box,h_box = self.grid_UNIT,self.grid_UNIT
        
        pil_image = Image.open('obs5.jpg')
        w, h = pil_image.size
        pil_image_resized = self.resize(w, h, w_box, h_box, pil_image)  
        tk_image1 = ImageTk.PhotoImage(pil_image_resized) 
        self.cv.bms_obs.append(tk_image1)

        pil_image = Image.open('obs7.jpg')
        w, h = pil_image.size
        pil_image_resized = self.resize(w, h, w_box, h_box, pil_image)  
        tk_image2 = ImageTk.PhotoImage(pil_image_resized) 
        self.cv.bms_obs.append(tk_image2)

        pil_image = Image.open('obs8.jpg')
        w, h = pil_image.size
        pil_image_resized = self.resize(w, h, w_box, h_box, pil_image)  
        tk_image3 = ImageTk.PhotoImage(pil_image_resized) 
        self.cv.bms_obs.append(tk_image3)
        
        self.cv.bms_obs.append(tk_image1)
        self.cv.bms_obs.append(tk_image2)
        self.cv.bms_obs.append(tk_image3)
               
        for i in range(10):
            self.cv.bms_obs.append(tk_image1)
            self.cv.bms_obs.append(tk_image2)
            self.cv.bms_obs.append(tk_image3)
        print(len(self.cv.bms_obs))
        self.cv.Obstacle = []
        index = 0
        for q in self.ObstacleCentre:
            print(len(self.ObstacleCentre))
            print("q",q)
            bm = self.cv.bms_obs[index]
            t = self.cv.create_image(q[0],q[1],image=bm)
            self.cv.Obstacle.append(t)
            self.AllItemsOrigPosition_list.append(self.cv.coords(t))
            index+=1

        pil_image = Image.open('obs5_car.jpg')
        w, h = pil_image.size
        pil_image_resized = self.resize(w, h, w_box, h_box, pil_image)  
        tk_image = ImageTk.PhotoImage(pil_image_resized) 
        self.cv.arriveObsImage.append(tk_image)

            
    def create_targetItems(self):
        self.cv.arriveImage = []
        self.cv.bms_wh = []
        w_box,h_box = self.grid_UNIT,self.grid_UNIT
        
        pil_image = Image.open('warehouse4_1.jpg')
        w, h = pil_image.size
        pil_image_resized = self.resize(w, h, w_box, h_box, pil_image)  
        tk_image = ImageTk.PhotoImage(pil_image_resized) 
        self.cv.bms_wh.append(tk_image)

        pil_image = Image.open('warehouse3.jpg')
        w, h = pil_image.size
        pil_image_resized = self.resize(w, h, w_box, h_box, pil_image)  
        tk_image = ImageTk.PhotoImage(pil_image_resized) 
        self.cv.bms_wh.append(tk_image)

        pil_image = Image.open('warehouse4_2.jpg')
        w, h = pil_image.size
        pil_image_resized = self.resize(w, h, w_box, h_box, pil_image)  
        tk_image = ImageTk.PhotoImage(pil_image_resized) 
        self.cv.bms_wh.append(tk_image)
        
        self.cv.warehouse = []
        index = 0
        for q in self.warehouseCentre:
            bm = self.cv.bms_wh[index]
            t = self.cv.create_image(q[0],q[1],image=bm)
            self.cv.warehouse.append(t)
            self.AllItemsOrigPosition_list.append(self.cv.coords(t))
            index+=1
        pil_image = Image.open('warehouse3_car.jpg')
        w, h = pil_image.size
        pil_image_resized = self.resize(w, h, w_box, h_box, pil_image)  
        tk_image = ImageTk.PhotoImage(pil_image_resized) 
        self.cv.arriveImage.append(tk_image)
        
    def create_agentItems(self):
        self.cv.bms = []
        w_box,h_box = self.grid_UNIT,self.grid_UNIT
        
        pil_image = Image.open('car9.jpg')
        w, h = pil_image.size
        pil_image_resized = self.resize(w, h, w_box, h_box, pil_image)  
        tk_image = ImageTk.PhotoImage(pil_image_resized) 
        self.cv.bms.append(tk_image)

        pil_image = Image.open('car2.jpg')
        w, h = pil_image.size
        pil_image_resized = self.resize(w, h, w_box, h_box, pil_image)  
        tk_image = ImageTk.PhotoImage(pil_image_resized) 
        self.cv.bms.append(tk_image)

        pil_image = Image.open('car8.jpg')
        w, h = pil_image.size
        pil_image_resized = self.resize(w, h, w_box, h_box, pil_image)  
        tk_image = ImageTk.PhotoImage(pil_image_resized) 
        self.cv.bms.append(tk_image)
        
        self.cv.car = []
        index = 0
        for q in self.agentCentre:
            bm = self.cv.bms[index]
            t = self.cv.create_image(q[0],q[1],image=bm)
            self.cv.car.append(t)
            self.AllItemsOrigPosition_list.append(self.cv.coords(t))
            index+=1
            
    def setItemsPositionList(self,itemCentre):
        npTemp = np.hstack((itemCentre,itemCentre))
        h_u = self.grid_UNIT/2
        npHalfUnit = np.array([-h_u,-h_u,h_u,h_u])
        hs = npHalfUnit
        for diam in range(1,itemCentre.shape[0]):
            hsTemp = np.vstack((npHalfUnit,hs))
            hs = hsTemp
        return (npTemp-hs).tolist()
    
    def button_reset(self):
        time.sleep(self.timeDelay)               
        for line in self.created_line:
            self.cv.delete(line)
        print(self.selected_agent_position[0])
        self.cv.coords(self.agentA, self.selected_agent_position[0])
        self.cv.coords(self.agentB, self.selected_agent_position[1])

    def reset(self, agentIndex):
        if agentIndex == 0:
            if self.linesA!=[]:
                for line in self.linesA:
                    self.cv.delete(line)
            if self.createMarkA is not None:
                self.cv.delete(self.createMarkA)
            
        if agentIndex == 1:
            if self.linesB!=[]:
                for line in self.linesB:
                    self.cv.delete(line) 
            if self.createMarkB is not None:
                self.cv.delete(self.createMarkB) 
        
        if agentIndex == 2:
            if self.linesC!=[]:
                for line in self.linesC:
                    self.cv.delete(line)
            if self.createMarkC is not None:
                self.cv.delete(self.createMarkC)
        
        if agentIndex!=0 and agentIndex!=1 and agentIndex!=2:
            ex = Exception("agentIndex Error in reset()！")
            raise ex
        Obs_list = self.selected_Obstacles_position
        while True:
            new_loc = [random.randrange(self.grid_origx_center,self.grid_endx,self.grid_UNIT),random.randrange(self.grid_origy_center,self.grid_endy,self.grid_UNIT)]
            if new_loc not in Obs_list:
                break
#        new_loc = random.sample(rand_loc, 1)
        self.cv.coords(self.selected_agent[agentIndex],new_loc)
        coords = self.cv.coords(self.selected_agent[agentIndex])
        return coords
    
    def reward_a(self, s_, B_s_,C_s_, s, s_B,s_C):
        # reward function
        self.targetA = self.selected_targets[0]
        if s_ == self.cv.coords(self.selected_targets[0]):
            t = self.cv.create_image(s_,image=self.cv.arriveImage[0])
            self.createMarkA = t
            reward = 1           
            done = True
            s_ = 'terminal'
            
        elif s_ in self.selected_Obstacles_position:
            reward = -0.75
            done = False
            s_ = 'terminal'
 
        elif s_ == B_s_:
            reward = -0.75
            s_ = 'terminal'
            done = False
        elif s_ == C_s_:
            reward = -0.75
            s_ = 'terminal'
            done = False
        elif s_==s_B and B_s_== s:
            reward = -0.75
            s_ = 'terminal'
            done = False
        
        elif s_==s_C and C_s_== s:
            reward = -0.75
            s_ = 'terminal'
            done = False
            
        else:
            reward = -0.04
            done = False

        return reward, done
    
    def reward_b(self, s_, A_s_,C_s_, s, s_A, s_C):
        # reward function
        self.targetB = self.selected_targets[1]
        if s_ == self.cv.coords(self.selected_targets[1]):
            t = self.cv.create_image(s_,image=self.cv.arriveImage[0])
            self.createMarkB = t
            reward = 1
            done = True
            s_ = 'terminal'            
        elif s_ in self.selected_Obstacles_position:
            reward = -0.75
            done = False
            s_ = 'terminal'
        elif s_ == A_s_:
            reward = -0.75
            s_ = 'terminal'
            done = False            
        elif s_ == C_s_:
            reward = -0.75
            s_ = 'terminal'
            done = False
        elif s_==s_A and A_s_== s:
            reward = -0.75
            s_ = 'terminal'
            done = False        
        elif s_==s_C and C_s_== s:
            reward = -0.75
            s_ = 'terminal'
            done = False
        else:
            reward = -0.04
            done = False
   
        return reward, done 
    
    def reward_c(self, s_, A_s_,B_s_, s, s_A, s_B):
        # reward function
        self.targetC = self.selected_targets[2]
        if s_ == self.cv.coords(self.selected_targets[2]):
            t = self.cv.create_image(s_,image=self.cv.arriveImage[0])
            self.createMarkC = t
            reward = 1
            done = True
            s_ = 'terminal'            
        elif s_ in self.selected_Obstacles_position:
            reward = -0.75
            done = False
            s_ = 'terminal'
        elif s_ == A_s_:
            reward = -0.75
            s_ = 'terminal'
            done = False            
        elif s_ == B_s_:
            reward = -0.75
            s_ = 'terminal'
            done = False
        elif s_==s_A and A_s_== s:
            reward = -0.75
            s_ = 'terminal'
            done = False       
        elif s_==s_B and B_s_== s:
            reward = -0.75
            s_ = 'terminal'
            done = False
        else:
            reward = -0.04
            done = False   
        return reward, done
    
    def real_step(self,A_s_,B_s_,C_s_):
        self.cv.coords(self.selected_agent[0], A_s_)  
        self.cv.coords(self.selected_agent[1], B_s_) 
        self.cv.coords(self.selected_agent[2], C_s_) 
        return
    
    def build_model(self, lr=0.001):
        model = Sequential()
        model.add(Dense(3*self.maze_size,input_shape=(3*self.maze_size,)))
        model.add(PReLU())
        model.add(Dense(3*self.maze_size))
        model.add(PReLU())
        model.add(Dense(3*self.n_actions))
        model.compile(optimizer='Adam', loss='mse')
        return model
    
    # This is a small utility for printing readable time strings:
    def format_time(self,seconds):
        if seconds < 400:
            s = float(seconds)
            return "%.1f seconds" % (s,)
        elif seconds < 4000:
            m = seconds / 60.0
            return "%.2f minutes" % (m,)
        else:
            h = seconds / 3600.0
            return "%.2f hours" % (h,)
    
    def normalize_input(self,observation):
        my_array = np.ones(self.maze_size)
        index = self.Qtable_gridIndex_dict[tuple(observation)]
        my_array[int(index)] = 0.5
        return my_array

    def update(self):
        model = self.build_model()
        experience = Experience(model, max_memory=self.max_memory_size)        
        win_historyA = []   
        win_historyB = []
        win_historyC = []
        start_time = datetime.datetime.now()
        # initial observation
        observation_A = self.cv.coords(self.agentA)
        observation_B = self.cv.coords(self.agentB)
        observation_C = self.cv.coords(self.agentC)
        loss_list = []
        total_reward_listA = []
        total_reward_listB = []
        total_reward_listC = []
        avg_reward_listA= []
        avg_reward_listB= []
        avg_reward_listC= []
        episodeA = 0
        episodeB = 0
        episodeC = 0
        total_rewardA = 0
        total_rewardB = 0
        total_rewardC = 0
        visitedA = set()
        visitedB = set()
        visitedC = set()
        stepCountA = 0
        stepCountB = 0
        stepCountC = 0
        while True:
            self.labelHello = Label(self.cv, text = "episodeA: %s"%str(episodeA), font=("Helvetica", 10), width = 10, fg = "red", bg = "white")
            self.labelHello.place(x=200, y=550, anchor=NW) 
            self.labelHello = Label(self.cv, text = "episodeB: %s"%str(episodeB), font=("Helvetica", 10), width = 10, fg = "blue", bg = "white")
            self.labelHello.place(x=200, y=580, anchor=NW)
            self.labelHello = Label(self.cv, text = "episodeC: %s"%str(episodeC), font=("Helvetica", 10), width = 10, fg = "green", bg = "white")
            self.labelHello.place(x=200, y=610, anchor=NW)
            self.render()
            visitedA.add(tuple(observation_A))
            visitedB.add(tuple(observation_B))
            visitedC.add(tuple(observation_C))
            normal_observation_A = self.normalize_input(observation_A)
            normal_observation_B = self.normalize_input(observation_B)
            normal_observation_C = self.normalize_input(observation_C)
            observation = np.hstack((np.array(normal_observation_A),np.array(normal_observation_B),np.array(normal_observation_C))).reshape(1,-1)
            action_A = experience.choose_action(observation,0)
            action_B = experience.choose_action(observation,1)
            action_C = experience.choose_action(observation,2)
            A_observation_ = self.calcu_next_state(observation_A,action_A)
            B_observation_ = self.calcu_next_state(observation_B,action_B)
            C_observation_ = self.calcu_next_state(observation_C,action_C)
            normal_observation_A_ = self.normalize_input(A_observation_)
            normal_observation_B_ = self.normalize_input(B_observation_)
            normal_observation_C_ = self.normalize_input(C_observation_)
            reward_A, done_A = self.reward_a(A_observation_, B_observation_, C_observation_, observation_A, observation_B, observation_C)
            reward_B, done_B = self.reward_b(B_observation_, A_observation_, C_observation_, observation_B, observation_A, observation_C)
            reward_C, done_C = self.reward_c(C_observation_, B_observation_, A_observation_, observation_C, observation_B, observation_A)
            self.real_step(A_observation_,B_observation_,C_observation_)
            stepCountA+=1 
            stepCountB+=1
            stepCountC+=1
            if tuple(A_observation_) in visitedA:
                reward_A -= 0.25
            if tuple(B_observation_) in visitedB:
                reward_B -= 0.25
            if tuple(C_observation_) in visitedC:
                reward_C -= 0.25
            if observation_A==A_observation_:
                reward_A = reward_A - 0.8                
            if observation_B==B_observation_:
                reward_B = reward_B - 0.8
            if observation_C==C_observation_:
                reward_C = reward_C - 0.8                
            if done_A == True:
                win_historyA.append(1)
            if done_B == True:
                win_historyB.append(1)
            if done_C == True:
                win_historyC.append(1)            
            total_rewardA += reward_A 
            total_rewardB += reward_B
            total_rewardC += reward_C
            if total_rewardA < -0.5*self.maze_size:
                done_A = True
                win_historyA.append(0)
            if total_rewardB < -0.5*self.maze_size:
                done_B = True
                win_historyB.append(0)            
            if total_rewardC < -0.5*self.maze_size:
                done_C = True
                win_historyC.append(0)                    
            episode_total = [normal_observation_A, action_A, reward_A, normal_observation_A_, done_A,\
                             normal_observation_B, action_B, reward_B, normal_observation_B_, done_B,\
                         normal_observation_C, action_C, reward_C, normal_observation_C_, done_C]
            experience.remember(episode_total)                
            # Train neural network model
            inputs, targets = experience.get_data(data_size=self.data_size)
            h = model.fit(
                inputs,
                targets,
                epochs=8,
                batch_size=16,
                verbose=0,
            )
            loss = model.evaluate(inputs, targets, verbose=0)              
            # swap observation
            observation_A = A_observation_
            observation_B = B_observation_
            observation_C = C_observation_        
            # break while loop when end of this episode
            if done_A:
                if episodeA>self.episode and episodeB>self.episode and episodeC>self.episode:
                    break
                else:
                    dt = datetime.datetime.now() - start_time
                    t = self.format_time(dt.total_seconds())
                    observation_A = self.reset(0)
                    episodeA+=1
                    total_reward_listA.append(total_rewardA)
                    if len(total_reward_listA) > 100:
                        avg_rewardA = sum(total_reward_listA[-100:])/100
                        avg_reward_listA.append(avg_rewardA)
                        template = "Episode(A): {:03d}/{:d} | Loss: {:.4f} | StepCount: {:d} | Win rate: {:.3f} | Total rewards: {:.3f} | Average rewards: {:.3f} | time: {}"
                        print(template.format(episodeA, self.episode, loss, stepCountA, sum(win_historyA)/len(win_historyA), total_rewardA, avg_rewardA, t))
                    else:
                        template = "Episode(A): {:03d}/{:d} | Loss: {:.4f} | StepCount: {:d} | Win rate: {:.3f} | Total rewards: {:.3f} | time: {}"
                        print(template.format(episodeA, self.episode, loss, stepCountA, sum(win_historyA)/len(win_historyA), total_rewardA, t))
                    stepCountA = 0                 
                    total_rewardA = 0
                    visitedA = set()                    
                    done_A = 0                                       
                    loss_list.append(loss)
            # break while loop when end of this episode
            if done_B:
                if episodeA>self.episode and episodeB>self.episode and episodeC>self.episode:
                    break
                else:
                    dt = datetime.datetime.now() - start_time
                    t = self.format_time(dt.total_seconds())
                    observation_B = self.reset(1)
                    episodeB+=1
                    total_reward_listB.append(total_rewardB)
                    if len(total_reward_listB) > 100:
                        avg_rewardB = sum(total_reward_listB[-100:])/100
                        avg_reward_listB.append(avg_rewardB)
                        template = "Episode(B): {:03d}/{:d} | Loss: {:.4f} | StepCount: {:d} | Win rate: {:.3f} | Total rewards: {:.3f} | Average rewards: {:.3f} | time: {}"
                        print(template.format(episodeB, self.episode, loss, stepCountB, sum(win_historyB)/len(win_historyB), total_rewardB, avg_rewardB, t))
                    else:
                        template = "Episode(B): {:03d}/{:d} | Loss: {:.4f} | StepCount: {:d} | Win rate: {:.3f} | Total rewards: {:.3f} | time: {}"
                        print(template.format(episodeB, self.episode, loss, stepCountB, sum(win_historyB)/len(win_historyB), total_rewardB, t))            
                    stepCountB = 0                   
                    total_rewardB = 0
                    visitedB = set()                    
                    done_B = 0                                       
                    loss_list.append(loss)
         # break while loop when end of this episode
            if done_C:
                if episodeA>self.episode and episodeB>self.episode and episodeC>self.episode:
                    break
                else:
                    dt = datetime.datetime.now() - start_time
                    t = self.format_time(dt.total_seconds())
                    observation_C = self.reset(2)
                    episodeC+=1
                    total_reward_listC.append(total_rewardC)
                    if len(total_reward_listC) > 100:
                        avg_rewardC = sum(total_reward_listC[-100:])/100
                        avg_reward_listC.append(avg_rewardC)
                        template = "Episode(C): {:03d}/{:d} | Loss: {:.4f} | StepCount: {:d} | Win rate: {:.3f} | Total rewards: {:.3f} | Average rewards: {:.3f} | time: {}"
                        print(template.format(episodeC, self.episode, loss, stepCountC, sum(win_historyC)/len(win_historyC), total_rewardC, avg_rewardC, t))
                    else:
                        template = "Episode(C): {:03d}/{:d} | Loss: {:.4f} | StepCount: {:d} | Win rate: {:.3f} | Total rewards: {:.3f} | time: {}"
                        print(template.format(episodeC, self.episode, loss, stepCountC, sum(win_historyC)/len(win_historyC), total_rewardC, t))                  
                    stepCountC = 0                   
                    total_rewardC = 0
                    visitedC = set()                    
                    done_C = 0                                       
                    loss_list.append(loss)

        # end of game
        print('game over')
        self.labelHello = Label(self.cv, text = "training end!", font=("Helvetica", 10), width = 10, fg = "red", bg = "white")
        self.labelHello.place(x=200, y=750, anchor=NW)
        end_time = datetime.datetime.now()
        dt = end_time - start_time
        seconds = dt.total_seconds()
        t = self.format_time(seconds)
        print("total_time",t)
        print("total_win_historyA",sum(win_historyA)/len(win_historyA))
        print("total_win_historyB",sum(win_historyB)/len(win_historyB))
        print("total_win_historyC",sum(win_historyC)/len(win_historyC))
        self.learning = False
        self.reset(0)
        self.reset(1)
        self.reset(2)
        plt.figure()
        plt.title('Loss per Episode')
        plt.plot(loss_list)
        plt.xlabel('Episode number')
        plt.ylabel('Loss')
        plt.show()
        plt.figure()
        plt.title('Rewards per Episode')
        plt.xlabel('Episode number')
        plt.ylabel('Score')
        plt.plot(total_reward_listA,label='agentA')
        plt.plot(total_reward_listB,label='agentB')
        plt.plot(total_reward_listC,label='agentC')
        plt.legend(loc='upper right')
        plt.show()       
        plt.figure()
        plt.title('Average Rewards over 100 Episode')
        plt.xlabel('Episode number')
        plt.ylabel('Rewards')
        plt.plot(avg_reward_listA,label='agentA')
        plt.plot(avg_reward_listB,label='agentB')
        plt.plot(avg_reward_listC,label='agentC')
        plt.legend(loc='upper right')
        plt.show()
        string = str(self.episode)
        model.save("tree-agent_DQN1_%s.h5"%string)

    def new_reward_a(self, s_, B_s_, C_s_, s, s_B, s_C):
        # test reward function
        self.targetA = self.selected_targets[0]
        if s_ == self.cv.coords(self.selected_targets[0]):
            t = self.cv.create_image(s_,image=self.cv.arriveImage[0])
            self.createMarkA = t
            reward = 0                      
            s_ = 'terminal'            
        elif s_ in self.selected_Obstacles_position:
            reward = -2
            s_ = 'terminal'
        elif s_ == B_s_:
            reward = -2
            s_ = 'terminal'        
        elif s_ == C_s_:
            reward = -2
            s_ = 'terminal'
        elif s_==s_B and B_s_== s:
            reward = -2
            s_ = 'terminal'       
        elif s_==s_C and C_s_== s:
            reward = -2
            s_ = 'terminal'            
        else:
            reward = 0
        return reward
    
    def new_reward_b(self, s_, A_s_, C_s_, s, s_A, s_C):
        self.targetB = self.selected_targets[1]
        if s_ == self.cv.coords(self.selected_targets[1]):
            t = self.cv.create_image(s_,image=self.cv.arriveImage[0])
            self.createMarkB = t
            reward = 0                       
        elif s_ in self.selected_Obstacles_position:
            reward = -2
        elif s_ == A_s_:
            reward = -2        
        elif s_ == C_s_:
            reward = -2
        elif s_==s_A and A_s_== s:
            reward = -2
        elif s_==s_C and C_s_== s:
            reward = -2            
        else:
            reward = 0
        return reward
        
    def new_reward_c(self, s_, A_s_, B_s_, s, s_A, s_B):
        # test reward function
        self.targetB = self.selected_targets[1]
        if s_ == self.cv.coords(self.selected_targets[1]):
            t = self.cv.create_image(s_,image=self.cv.arriveImage[0])
            self.createMarkB = t
            reward = 0                        
        elif s_ in self.selected_Obstacles_position:
            reward = -2
        elif s_ == A_s_:
            reward = -2       
        elif s_ == B_s_:
            reward = -2                        
        elif s_==s_A and A_s_== s:
            reward = -2        
        elif s_==s_B and B_s_== s:
            reward = -2            
        else:
            reward = 0            
        return reward
    
    def choose_closer_car(self,s_A,s_B,s_C,terminal):
        dis_A = (s_A[0]-terminal[0])**2 + (s_A[1]-terminal[1])**2
        dis_B = (s_B[0]-terminal[0])**2 + (s_B[1]-terminal[1])**2
        dis_C = (s_C[0]-terminal[0])**2 + (s_C[1]-terminal[1])**2
        if dis_A < dis_B and dis_A < dis_C:
            return 0
        if dis_B < dis_A and dis_B < dis_C:
            return 1
        if dis_C < dis_A and dis_C < dis_B:
            return 2
    def run(self):
        self.run_model = load_model("tree-agent_DQN6000_1000_1000enhance.h5")
        print("model loaded!!")
        print(self.run_model.summary())
        action_B = -1
        action_A = -1
        action_C = -1
        # initial observation
        observation_A = self.cv.coords(self.agentA)
        terminal_A = 0
        observation_B = self.cv.coords(self.agentB)
        terminal_B = 0
        observation_C = self.cv.coords(self.agentC)
        terminal_C = 0
        print("observation after reset:",observation_A, observation_B)
        stepCountA = 0
        stepCountB = 0
        stepCountC = 0
        doneA = 0
        doneB = 0
        doneC = 0
        print("task_list",self.task_list)        
        while True:
            print("task_list---",self.task_list)
            time.sleep(self.timeDelay)            
            print("step----",stepCountA)
#            print("busyA",self.busyA)
            print("busyB",self.busyB)
#            print("----observeA",observation_A)
            print("----observeB",observation_B)
            stepCountA+=1
            stepCountB+=1
            stepCountC+=1
            normal_observation_A = self.normalize_input(observation_A)
            normal_observation_B = self.normalize_input(observation_B)
            normal_observation_C = self.normalize_input(observation_C)
            observation = np.hstack((np.array(normal_observation_A),np.array(normal_observation_B),np.array(normal_observation_C))).reshape(1,-1)
            if self.busyA == 0 and self.busyB == 0 and self.busyC == 0 and self.task_list!=[]:
                print("activated!!!!1!11111-------")
                terminal = self.task_list[0]
                print("------terminal",terminal)
                print(observation_A,observation_B,observation_C,terminal)
                agentIndex = self.choose_closer_car(observation_A,observation_B,observation_C,terminal)
#                self.task_list.pop(0)
                if agentIndex == 1:
                    print("----------choose_B!!!--------")
                    self.task_list.pop(0)
                    terminal_B = terminal
                    task_num_B = self.task_num_list[0]
                    self.task_num_list.pop(0)
                    time.sleep(self.timeDelay)
                    self.labelHello = Label(self.cv, text = "Task list:%s"%self.task_num_list,width = 50, fg = "black",bg = "white")
                    self.labelHello.place(x=self.agentCentre[0][0]-150, y=self.agentCentre[0][1]+450, anchor=NW)
                    self.busyB = 1
                    self.labelHello = Label(self.cv, text = "B is busy! going to %s!"%task_num_B, width = 25, fg = "blue",bg = "white")
                    self.labelHello.place(x=self.agentCentre[0][0], y=self.agentCentre[0][1]+500, anchor=NW)
                    if terminal_B == self.cv.coords(self.targetA):
                        observation = np.hstack((np.array(normal_observation_B),np.array(normal_observation_A),np.array(normal_observation_C))).reshape(1,-1)
                        action_B = np.argmax(self.run_model.predict(observation)[0][0:4])
                    elif terminal_B == self.cv.coords(self.targetB):
                        observation = np.hstack((np.array(normal_observation_A),np.array(normal_observation_B),np.array(normal_observation_C))).reshape(1,-1)
                        action_B = np.argmax(self.run_model.predict(observation)[0][4:8])
                    else:
                        observation = np.hstack((np.array(normal_observation_A),np.array(normal_observation_C),np.array(normal_observation_B))).reshape(1,-1)
                        action_B = np.argmax(self.run_model.predict(observation)[0][8:])
                    
                    B_observation_ = self.calcu_next_state(observation_B,action_B)
                elif agentIndex == 0:
                    doneA = 0
                    print("----------choose_A!!!--------")
                    self.task_list.pop(0)
                    terminal_A = terminal
                    task_num_A = self.task_num_list[0]
                    self.task_num_list.pop(0)
                    time.sleep(self.timeDelay)
                    self.labelHello = Label(self.cv, text = "Task list:%s"%self.task_num_list,width = 50, fg = "black",bg = "white")
                    self.labelHello.place(x=self.agentCentre[0][0]-150, y=self.agentCentre[0][1]+450, anchor=NW)
                    self.busyA = 1
                    self.labelHello = Label(self.cv, text = "A is busy! going to %s!"%task_num_A, width = 25, fg = "red",bg = "white")
                    self.labelHello.place(x=self.agentCentre[0][0]-180, y=self.agentCentre[0][1]+500, anchor=NW)
                    if terminal_A == self.cv.coords(self.targetA):
                        observation = np.hstack((np.array(normal_observation_A),np.array(normal_observation_B),np.array(normal_observation_C))).reshape(1,-1)
                        action_A = np.argmax(self.run_model.predict(observation)[0][0:4])
                    elif terminal_A == self.cv.coords(self.targetB):
                        observation = np.hstack((np.array(normal_observation_B),np.array(normal_observation_A),np.array(normal_observation_C))).reshape(1,-1)
                        action_A = np.argmax(self.run_model.predict(observation)[0][4:8])
                    else:
                        observation = np.hstack((np.array(normal_observation_C),np.array(normal_observation_B),np.array(normal_observation_A))).reshape(1,-1)
                        action_A = np.argmax(self.run_model.predict(observation)[0][8:])
                    
                    A_observation_ = self.calcu_next_state(observation_A,action_A)
                else:
                    doneC = 0
                    print("----------choose_C!!!--------")
                    self.task_list.pop(0)
                    terminal_C = terminal
                    task_num_C = self.task_num_list[0]
                    self.task_num_list.pop(0)
                    time.sleep(self.timeDelay)
                    self.labelHello = Label(self.cv, text = "Task list:%s"%self.task_num_list,width = 50, fg = "black",bg = "white")
                    self.labelHello.place(x=self.agentCentre[0][0]-150, y=self.agentCentre[0][1]+450, anchor=NW)
                    self.busyC = 1
                    self.labelHello = Label(self.cv, text = "C is busy! going to %s!"%task_num_C, width = 25, fg = "green",bg = "white")
                    self.labelHello.place(x=self.agentCentre[0][0]+180, y=self.agentCentre[0][1]+500, anchor=NW)
                    
                    if terminal_C == self.cv.coords(self.targetA):
                        observation = np.hstack((np.array(normal_observation_C),np.array(normal_observation_B),np.array(normal_observation_A))).reshape(1,-1)
                        action_C = np.argmax(self.run_model.predict(observation)[0][0:4])
                    elif terminal_C == self.cv.coords(self.targetB):
                        observation = np.hstack((np.array(normal_observation_A),np.array(normal_observation_C),np.array(normal_observation_B))).reshape(1,-1)
                        action_C = np.argmax(self.run_model.predict(observation)[0][4:8])
                    else:
                        observation = np.hstack((np.array(normal_observation_A),np.array(normal_observation_B),np.array(normal_observation_C))).reshape(1,-1)
                        action_C = np.argmax(self.run_model.predict(observation)[0][8:])
                    C_observation_ = self.calcu_next_state(observation_C,action_C)
            if self.busyB == 0:
                if self.task_list!=[]:
                    terminal_B = self.task_list[0]
                    self.task_list.pop(0)
                    task_num_B = self.task_num_list[0]
                    self.task_num_list.pop(0)
                    time.sleep(self.timeDelay)
                    self.labelHello = Label(self.cv, text = "Task list:%s"%self.task_num_list,width = 50, fg = "black",bg = "white")
                    self.labelHello.place(x=self.agentCentre[0][0]-150, y=self.agentCentre[0][1]+450, anchor=NW)
                    self.busyB = 1
                    doneB = 0
                    self.labelHello = Label(self.cv, text = "B is busy! going to %s!"%task_num_B, width = 25, fg = "blue",bg = "white")
                    self.labelHello.place(x=self.agentCentre[0][0], y=self.agentCentre[0][1]+500, anchor=NW)
                    if terminal_B == self.cv.coords(self.targetA):
                        observation = np.hstack((np.array(normal_observation_B),np.array(normal_observation_A),np.array(normal_observation_C))).reshape(1,-1)
                        action_B = np.argmax(self.run_model.predict(observation)[0][0:4])
                    elif terminal_B == self.cv.coords(self.targetB):
                        observation = np.hstack((np.array(normal_observation_A),np.array(normal_observation_B),np.array(normal_observation_C))).reshape(1,-1)
                        action_B = np.argmax(self.run_model.predict(observation)[0][4:8])
                    else:
                        observation = np.hstack((np.array(normal_observation_A),np.array(normal_observation_C),np.array(normal_observation_B))).reshape(1,-1)
                        action_B = np.argmax(self.run_model.predict(observation)[0][8:])
                    B_observation_ = self.calcu_next_state(observation_B,action_B)
                else:
                    print("No job for B!")
                    self.cv.delete(self.createMarkB)
                    self.labelHello = Label(self.cv, text = "No job for B!", width = 25, fg = "blue",bg = "white")
                    self.labelHello.place(x=self.agentCentre[0][0], y=self.agentCentre[0][1]+500, anchor=NW)
                    doneB = 1
                    B_observation_=[1175.0,145.0]
            if self.busyB!=0:
                if terminal_B == self.cv.coords(self.targetA):
                    observation = np.hstack((np.array(normal_observation_B),np.array(normal_observation_A),np.array(normal_observation_C))).reshape(1,-1)
                    action_B = np.argmax(self.run_model.predict(observation)[0][0:4])
                elif terminal_B == self.cv.coords(self.targetB):
                    observation = np.hstack((np.array(normal_observation_A),np.array(normal_observation_B),np.array(normal_observation_C))).reshape(1,-1)
                    action_B = np.argmax(self.run_model.predict(observation)[0][4:8])
                else:
                    observation = np.hstack((np.array(normal_observation_A),np.array(normal_observation_C),np.array(normal_observation_B))).reshape(1,-1)
                    action_B = np.argmax(self.run_model.predict(observation)[0][8:])
                B_observation_ = self.calcu_next_state(observation_B,action_B)        
            if self.busyA == 0:
                if self.task_list!=[]:
                    terminal_A = self.task_list[0]
                    self.task_list.pop(0)
                    task_num_A = self.task_num_list[0]
                    self.task_num_list.pop(0)
                    time.sleep(self.timeDelay)
                    self.labelHello = Label(self.cv, text = "Task list:%s"%self.task_num_list,width = 50, fg = "black",bg = "white")
                    self.labelHello.place(x=self.agentCentre[0][0]-150, y=self.agentCentre[0][1]+450, anchor=NW)
                    doneA = 0
                    self.busyA = 1
                    self.labelHello = Label(self.cv, text = "A is busy! going to %s!"%task_num_A, width = 25, fg = "red",bg = "white")
                    self.labelHello.place(x=self.agentCentre[0][0]-180, y=self.agentCentre[0][1]+500, anchor=NW)
                    if terminal_A == self.cv.coords(self.targetA):
                        observation = np.hstack((np.array(normal_observation_A),np.array(normal_observation_B),np.array(normal_observation_C))).reshape(1,-1)
                        action_A = np.argmax(self.run_model.predict(observation)[0][0:4])
                    elif terminal_A == self.cv.coords(self.targetB):
                        observation = np.hstack((np.array(normal_observation_B),np.array(normal_observation_A),np.array(normal_observation_C))).reshape(1,-1)
                        action_A = np.argmax(self.run_model.predict(observation)[0][4:8])
                    else:
                        observation = np.hstack((np.array(normal_observation_C),np.array(normal_observation_B),np.array(normal_observation_A))).reshape(1,-1)
                        action_A = np.argmax(self.run_model.predict(observation)[0][8:])
                    A_observation_ = self.calcu_next_state(observation_A,action_A)
                else:
                    print("No job for A!")
                    self.cv.delete(self.createMarkA)
                    self.labelHello = Label(self.cv, text = "No job for A!", width = 25, fg = "red",bg = "white")
                    self.labelHello.place(x=self.agentCentre[0][0]-180, y=self.agentCentre[0][1]+500, anchor=NW)
                    A_observation_ = observation_A
                    doneA = 1
                    A_observation_=[525.0,645.0]
            if self.busyA!=0:
                if terminal_A == self.cv.coords(self.targetA):
                    observation = np.hstack((np.array(normal_observation_A),np.array(normal_observation_B),np.array(normal_observation_C))).reshape(1,-1)
                    action_A = np.argmax(self.run_model.predict(observation)[0][0:4])
                elif terminal_A == self.cv.coords(self.targetB):
                    observation = np.hstack((np.array(normal_observation_B),np.array(normal_observation_A),np.array(normal_observation_C))).reshape(1,-1)
                    action_A = np.argmax(self.run_model.predict(observation)[0][4:8])
                else:
                    observation = np.hstack((np.array(normal_observation_C),np.array(normal_observation_B),np.array(normal_observation_A))).reshape(1,-1)
                    action_A = np.argmax(self.run_model.predict(observation)[0][8:])
                A_observation_ = self.calcu_next_state(observation_A,action_A)
            
            if self.busyC == 0:
                if self.task_list!=[]:
                    terminal_C = self.task_list[0]
                    self.task_list.pop(0)
                    task_num_C = self.task_num_list[0]
                    self.task_num_list.pop(0)
                    time.sleep(self.timeDelay)
                    self.labelHello = Label(self.cv, text = "Task list:%s"%self.task_num_list,width = 50, fg = "black",bg = "white")
                    self.labelHello.place(x=self.agentCentre[0][0]-150, y=self.agentCentre[0][1]+450, anchor=NW)
                    doneC = 0
                    self.busyC = 1
                    self.labelHello = Label(self.cv, text = "C is busy! going to %s!"%task_num_C, width = 25, fg = "green",bg = "white")
                    self.labelHello.place(x=self.agentCentre[0][0]+180, y=self.agentCentre[0][1]+500, anchor=NW)
                    if terminal_C == self.cv.coords(self.targetA):
                        observation = np.hstack((np.array(normal_observation_C),np.array(normal_observation_B),np.array(normal_observation_A))).reshape(1,-1)
                        action_C = np.argmax(self.run_model.predict(observation)[0][0:4])
                    elif terminal_C == self.cv.coords(self.targetB):
                        observation = np.hstack((np.array(normal_observation_A),np.array(normal_observation_C),np.array(normal_observation_B))).reshape(1,-1)
                        action_C = np.argmax(self.run_model.predict(observation)[0][4:8])
                    else:
                        observation = np.hstack((np.array(normal_observation_A),np.array(normal_observation_B),np.array(normal_observation_C))).reshape(1,-1)
                        action_C = np.argmax(self.run_model.predict(observation)[0][8:])
                    C_observation_ = self.calcu_next_state(observation_C,action_C)
                else:
                    print("No job for C!")
                    self.cv.delete(self.createMarkC)
                    self.labelHello = Label(self.cv, text = "No job for C!", width = 25, fg = "green",bg = "white")
                    self.labelHello.place(x=self.agentCentre[0][0]+180, y=self.agentCentre[0][1]+500, anchor=NW)
                    C_observation_ = observation_C
                    doneC = 1
                    C_observation_=[525.0,95.0]
            if self.busyC!=0:
                if terminal_C == self.cv.coords(self.targetA):
                    observation = np.hstack((np.array(normal_observation_C),np.array(normal_observation_B),np.array(normal_observation_A))).reshape(1,-1)
                    action_C = np.argmax(self.run_model.predict(observation)[0][0:4])
                elif terminal_C == self.cv.coords(self.targetB):
                    observation = np.hstack((np.array(normal_observation_A),np.array(normal_observation_C),np.array(normal_observation_B))).reshape(1,-1)
                    action_C = np.argmax(self.run_model.predict(observation)[0][4:8])
                else:
                    observation = np.hstack((np.array(normal_observation_A),np.array(normal_observation_B),np.array(normal_observation_C))).reshape(1,-1)
                    action_C = np.argmax(self.run_model.predict(observation)[0][8:])
                C_observation_ = self.calcu_next_state(observation_C,action_C)
            
            print("agentB---------",self.cv.coords(self.agentB))
            print("terminal_B",terminal_B)
            print("agentA---------",self.cv.coords(self.agentA))
            print("terminal_A",terminal_A)
            print("agentC---------",self.cv.coords(self.agentC))
            print("terminal_C",terminal_C)
            if self.cv.coords(self.agentB) == terminal_B:
                doneB = 1
                
                t = self.cv.create_image(terminal_B,image=self.cv.arriveImage[0])
                self.createMarkB = t
                self.labelHello = Label(self.cv, text = "B done!", width = 25, fg = "blue",bg = "white")
                self.labelHello.place(x=self.agentCentre[0][0], y=self.agentCentre[0][1]+500, anchor=NW)
                time.sleep(self.timeDelay)
                self.busyB = 0
                self.labelHello = Label(self.cv, text = "B is free!", width = 25, fg = "blue",bg = "white")
                self.labelHello.place(x=self.agentCentre[0][0], y=self.agentCentre[0][1]+500, anchor=NW)
                B_observation_ = self.reset(1)
                for line in self.linesB:
                    self.cv.delete(line)
            if self.cv.coords(self.agentA) == terminal_A:
                doneA = 1
                t = self.cv.create_image(terminal_A,image=self.cv.arriveImage[0])
                self.createMarkA = t
                self.labelHello = Label(self.cv, text = "A done!", width = 25, fg = "red",bg = "white")
                self.labelHello.place(x=self.agentCentre[0][0]-180, y=self.agentCentre[0][1]+500, anchor=NW)
                time.sleep(self.timeDelay)
                self.busyA = 0
                self.labelHello = Label(self.cv, text = "A is free!", width = 25, fg = "red",bg = "white")
                self.labelHello.place(x=self.agentCentre[0][0]-180, y=self.agentCentre[0][1]+500, anchor=NW)
                A_observation_ = self.reset(0)
                for line in self.linesA:
                    self.cv.delete(line)                
                print("---A DONE")

            if self.cv.coords(self.agentC) == terminal_C:
                doneC = 1
                t = self.cv.create_image(terminal_C,image=self.cv.arriveImage[0])
                self.createMarkA = t
                self.labelHello = Label(self.cv, text = "C done!", width = 25, fg = "green",bg = "white")
                self.labelHello.place(x=self.agentCentre[0][0]+180, y=self.agentCentre[0][1]+500, anchor=NW)
                time.sleep(self.timeDelay)
                self.busyC = 0
                self.labelHello = Label(self.cv, text = "C is free!", width = 25, fg = "green",bg = "white")
                self.labelHello.place(x=self.agentCentre[0][0]+180, y=self.agentCentre[0][1]+500, anchor=NW)
                C_observation_ = self.reset(2)
                for line in self.linesC:
                    self.cv.delete(line)
            self.real_step(A_observation_,B_observation_,C_observation_)
            if self.busyA != 0:
                lineA = self.cv.create_line(observation_A[0], observation_A[1],
                      A_observation_[0], A_observation_[1],
                      fill='red',
                      arrow=LAST,
                      arrowshape=(10,20,8),
                      dash=(4, 4)  
                      )
                self.linesA.append(lineA)
            if self.busyB != 0:
                lineB = self.cv.create_line(observation_B[0], observation_B[1],
                      B_observation_[0], B_observation_[1],
                      fill='blue',
                      arrow=LAST,
                      arrowshape=(10,20,8),
                      dash=(4, 4)  
                      )
                self.linesB.append(lineB)
            if self.busyC != 0:
                lineC = self.cv.create_line(observation_C[0], observation_C[1],
                      C_observation_[0], C_observation_[1],
                      fill='green',
                      arrow=LAST,
                      arrowshape=(10,20,8),
                      dash=(4, 4)  
                      )
                self.linesC.append(lineC)                    
            observation_A = A_observation_
            observation_B = B_observation_
            observation_C = C_observation_
            print("take step")

    def start_learning(self):
        print(self.agentItemIndex)
        print(self.WarehouseItemIndex)
        print(self.ObstacleItemIndex)
        print(self.itemsNum) 
        self.selected_agent = []
        self.selected_targets = []
        self.selected_Obstacles = []
        self.selected_agent_position = []
        self.selected_Obstacles_position = []
        
        for item in range(1,self.itemsNum+1):
            print(item)
            p = self.cv.coords(item)
            print("self.grid_origx and self.grid_origy = ",self.grid_origx,self.grid_origy)
            print("p=",p)
            if p[0]>=self.grid_origx and p[1]>=self.grid_origy \
                and p[0]<=self.grid_endx and p[1]<=self.grid_endy:
                if item in range(self.agentItemIndex[0],self.agentItemIndex[1]+1):
                    self.selected_agent.append(item)
                    self.selected_agent_position.append(p)
                elif item in range(self.WarehouseItemIndex[0],self.WarehouseItemIndex[1]+1):
                    self.selected_targets.append(item)
                elif item in range(self.ObstacleItemIndex[0],self.ObstacleItemIndex[1]+1):
                    self.selected_Obstacles.append(item)
                    self.selected_Obstacles_position.append(p)

        print(self.selected_agent) 
        print(self.selected_targets)  
        print("111------------",self.selected_Obstacles_position) 
        self.targetA = self.selected_targets[0]
        self.targetB = self.selected_targets[1]
        self.targetC = self.selected_targets[2]
        self.agentA = self.selected_agent[0]
        self.agentB = self.selected_agent[1]
        self.agentC = self.selected_agent[2]
        if len(self.selected_agent)==0 or len(self.selected_agent)>3:
            tkinter.messagebox.showinfo("Info：","Please put 3 agents on grid map!")
#        elif len(self.selected_targets)==0 or len(self.selected_targets)>3:
#            tkinter.messagebox.showinfo("Info：","请选择一个或者两个Target-->warehouse！")
#        elif len(self.selected_agent)!=len(self.selected_targets):
#            tkinter.messagebox.showinfo("Info：","请选择相同数目的agent和Target！")
        else:
            self.t = threading.Timer(self.timeDelay, self.update)
            self.t.start()
            self.learning = True
            
            
            
    def restart(self):
        print("restart")
        self.cv.coords(self.agentA,self.agentCentre[0])
        self.cv.coords(self.agentB,self.agentCentre[1])
        
    def calcu_next_state(self,loc,action):
         
        UNIT = self.grid_UNIT
        ss = loc
        np_s = np.array(ss)
        dissS = np.array([self.grid_origx,self.grid_origy]) 
        s = (np_s-dissS).tolist()
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (self.grid_rowNum - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (self.grid_columnNum - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        s_=[]
        s_ = [ss[0]+base_action[0],ss[1]+base_action[1]]
        return s_    
    
    def render(self):
        time.sleep(self.timeDelay)
        
    def create_items(self):
        self.AllItemsOrigPosition_list.append([0,0,0,0])       
        self.create_agentItems()
        self.agentItemIndex = [1,len(self.agentPosition_list)]
        self.create_targetItems()
        self.WarehouseItemIndex = [self.agentItemIndex[1]+1,self.agentItemIndex[1]+len(self.warehousePostition_list)]
        self.create_ObsItems()
        self.ObstacleItemIndex = [self.WarehouseItemIndex[1]+1,self.WarehouseItemIndex[1]+len(self.ObstaclePosition_list)]

    def create_grids(self,origx,origy,column,row,UNIT):
        # create grids
        for c in range(origx, origx+(column+1)*UNIT, UNIT):
            x0, y0, x1, y1 = c, origy, c, origy+row*UNIT
            self.cv.create_line(x0, y0, x1, y1,width=2)
        for r in range(origy, origy+(row+1)*UNIT, UNIT):
            x0, y0, x1, y1 = origx, r, origx+row*UNIT, r
            self.cv.create_line(x0, y0, x1, y1,width=2)        
    def choose_type(self, i):
        for b in self.bns: b['relief'] = RAISED        
        self.bns[i]['relief'] = SUNKEN
        self.item_type = i        
        if self.item_type==1:
            self.start_learning()
            self.bns[i]['relief'] = RAISED
        elif self.item_type==2:
            os._exit(0)
        elif self.item_type==3:            
            temp_s1=str(self.cv.coords(self.targetA))+str(self.selected_Obstacles_position)+'both avoiding'
            temp_s2=str(self.cv.coords(self.targetB))+str(self.selected_Obstacles_position)+'both avoiding'
            self.RL_A.q_table.to_csv("table terminal%s.csv"%temp_s1, index_label="index_label")
            self.RL_B.q_table.to_csv("table terminal%s.csv"%temp_s2, index_label="index_label")
            print("SAVED!!!")
            self.labelHello = Label(self.cv, text = "table saved!!", font=("Helvetica", 10), width = 10, fg = "red", bg = "white")
            self.labelHello.place(x=250, y=750, anchor=NW)
        elif self.item_type==0:
            self.button_reset()
        elif self.item_type==4:
            self.start_running()
        elif self.item_type==5:            
            self.restart()        
        elif self.item_type==8:
            print("A----Pressed")
            self.task_list.append(self.cv.coords(self.targetA))
            self.task_num_list.append('A')
            self.labelHello = Label(self.cv, text = "Task list:%s"%self.task_num_list,width = 50, fg = "black",bg = "white")
            self.labelHello.place(x=self.agentCentre[0][0]-150, y=self.agentCentre[0][1]+450, anchor=NW)
            self.bns[i]['relief'] = RAISED
        elif self.item_type==7:
            print("B------pressed")
            self.task_list.append(self.cv.coords(self.targetB))
            self.task_num_list.append('B')
            self.labelHello = Label(self.cv, text = "Task list:%s"%self.task_num_list, width = 50, fg = "black",bg = "white")
            self.labelHello.place(x=self.agentCentre[0][0]-150, y=self.agentCentre[0][1]+450, anchor=NW)
            self.bns[i]['relief'] = RAISED
        elif self.item_type==6:
            print("C------pressed")
            self.task_list.append(self.cv.coords(self.targetC))
            self.task_num_list.append('C')
            self.labelHello = Label(self.cv, text = "Task list:%s"%self.task_num_list, width = 50, fg = "black",bg = "white")
            self.labelHello.place(x=self.agentCentre[0][0]-150, y=self.agentCentre[0][1]+450, anchor=NW)
            self.bns[i]['relief'] = RAISED

    def start_running(self):
        print(self.agentItemIndex)
        print(self.WarehouseItemIndex)
        print(self.ObstacleItemIndex)
        print(self.itemsNum)
        self.selected_agent = []
        self.selected_targets = []
        self.selected_Obstacles = []
        self.selected_agent_position = []
        self.selected_Obstacles_position = []    
        self.task_list = []
        self.task_num_list = []
        for item in range(1,self.itemsNum+1):
            print(item)
            p = self.cv.coords(item)
            print("self.grid_origx and self.grid_origy = ",self.grid_origx,self.grid_origy)
            print("p=",p)
            if p[0]>=self.grid_origx and p[1]>=self.grid_origy:
                if item in range(self.agentItemIndex[0],self.agentItemIndex[1]+1):
                    self.selected_agent.append(item)
                    self.selected_agent_position.append(p)
                elif item in range(self.WarehouseItemIndex[0],self.WarehouseItemIndex[1]+1):
                    self.selected_targets.append(item)
                elif item in range(self.ObstacleItemIndex[0],self.ObstacleItemIndex[1]+1):
                    self.selected_Obstacles.append(item)
                    
                    self.selected_Obstacles_position.append(p)
                    print("self.selected_Obstacles_position",self.selected_Obstacles_position)
                    print("select---AGENT",self.selected_agent_position)
        print(self.selected_agent) 
        print("selected_targets:",self.cv.coords(self.selected_targets[0]))  
        print(self.selected_Obstacles) 
        
        if len(self.selected_agent)<=1 or len(self.selected_agent)>3:
            tkinter.messagebox.showinfo("Please place TWO agent on map!")
        elif len(self.selected_targets)==0 or len(self.selected_targets)>3:
            tkinter.messagebox.showinfo("Please choose ONE terminal!")
        else:
            print("self.selected_agent[0]=",self.selected_agent[0])
            self.agentA = self.selected_agent[0]
            self.agentB = self.selected_agent[1]
            self.agentC = self.selected_agent[2]
            self.targetA = self.selected_targets[0]
            self.targetB = self.selected_targets[1]
            self.targetC = self.selected_targets[2]
            print("------agentA:------",self.cv.coords(self.agentA))
            print("------agentB:------",self.cv.coords(self.agentB))
            print("------agentC:------",self.cv.coords(self.agentC))
            print("------A:------",self.cv.coords(self.targetA))
            print("------B:------",self.cv.coords(self.targetB))
            print("------C:------",self.cv.coords(self.targetC))
            self.labelHello = Label(self.cv, text = "start running!!", font=("Helvetica", 10), width = 10, fg = "red", bg = "white")
            self.labelHello.place(x=250, y=750, anchor=NW)       
            self.t = threading.Timer(self.timeDelay, self.run)
            self.t.start()
            self.learning = True
                 
    def rightClick_handler(self, event):
        print("rightClick_handler:event.x,event.y = ",event.x,event.y)
        self.start_learning()
        
    def leftClick_handler(self, event):
        self.entryCd.place_forget()       
        if self.learning:
            print("Learing on going!")
        else:
            for item in range(1,self.itemsNum+1):
                position = self.cv.coords(item)
                R = self.grid_UNIT/2
                p = [position[0]-R,position[1]-R,position[0]+R,position[1]+R]
                if event.x>=p[0] and event.x<=p[2] and \
                    event.y>=p[1] and event.y<=p[3]:
                    print("find you!", item)
                    t = item
                    self.choose_item_handler(event,t)
                    
    def choose_item_handler(self, event, t):
        self.choose_item = t
        self.itemOrigPosition = self.cv.coords(t)
 
    def move(self, event):
        if self.choose_item is not None:
            t = self.choose_item
            self.cv.coords(t,event.x,event.y)
   
    def adjust_items_into_grids(self,event):
        if self.choose_item is not None:
            t = self.choose_item
            position = self.cv.coords(t)
            centerX = position[0]
            centerY = position[1]
            Grids_X0 = self.grid_origx
            Grids_X1 = self.grid_origx+(self.grid_columnNum+1)*self.grid_UNIT
            Grids_Y0 = self.grid_origy
            Grids_Y1 = self.grid_origy+(self.grid_rowNum+1)*self.grid_UNIT            
            if (centerX in range(Grids_X0,Grids_X1)) and (centerY in range(Grids_Y0,Grids_Y1)):
                columnIndex = math.floor((centerX- Grids_X0)/self.grid_UNIT)
                rowIndex = math.floor((centerY- Grids_Y0)/self.grid_UNIT)
                adjustedX0 = Grids_X0+columnIndex*self.grid_UNIT+self.grid_UNIT/2
                adjustedY0 = Grids_Y0+rowIndex*self.grid_UNIT+self.grid_UNIT/2
                self.cv.coords(t,adjustedX0,adjustedY0)
            else:
                self.cv.coords(t,self.AllItemsOrigPosition_list[t])
                self.itemOrigPosition = []
                
    def move_end(self, event):
        if self.choose_item is not None:
            t = self.choose_item
            print("move_end: item on going is:",t)
            self.adjust_items_into_grids(event)
            self.choose_item = None
            
            
        print("self.cv.coords(1) = ",self.cv.coords(1))
        print("self.cv.coords(2) = ",self.cv.coords(2))
        print("self.cv.coords(3) = ",self.cv.coords(3))
        print("self.cv.coords(4) = ",self.cv.coords(4))
        print("self.cv.coords(5) = ",self.cv.coords(5))
        print("self.cv.coords(6) = ",self.cv.coords(6))
              
       
    def delete_item(self, event):
        if self.choose_item is not None:
            self.cv.delete(self.choose_item)
            
root = Tk()
root.title("Multi-agent DQN")
root.attributes("-fullscreen", False)
app = App(root)
root.bind('<Delete>', app.delete_item)
root.mainloop()


#print('End of program')
## ===================================
#sys.stdout = origin
#f.close()
