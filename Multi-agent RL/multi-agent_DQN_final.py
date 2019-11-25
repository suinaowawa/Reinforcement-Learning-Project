# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:13:43 2019

@author: Yue
"""
"""uncomment these lines to get the printed output in a .txt file"""
## =================================== 
## -*- coding: utf-8 -*- 
#import sys
#origin = sys.stdout
#f = open('single_dqn_train_log.txt', 'w')
#sys.stdout = f
## ===================================
#print('Start of program')

from tkinter import *
from tkinter import ttk
import threading
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
from PIL import Image, ImageTk
import tkinter.messagebox 
import time
import random

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import json
from Experience_normal_two import Experience

class App:
    def __init__(self, master):
        self.master = master

#        grid map setting
        self.grid_origx = 500
        self.grid_origy = 20
        self.grid_columnNum = 8
        self.grid_rowNum = 8
        self.grid_UNIT = 90
        self.maze_size = self.grid_columnNum*self.grid_rowNum
#        define total training episodes
        self.episode = 10000
#        define number of tests to run
        self.tests = 100
#        set a small amount of delay (second) to make sure tkinter works properly
#        if want to have a slower visulazation for testing, set the delay to larger values
        self.timeDelay = 0.0005
        
        
#       other initialization   
        self.max_memory_size = 8*self.maze_size
        self.data_size = 32
        self.n_actions = 4         
        self.outline = 'black'
        self.fill = None
        self.item_type = 0
        self.learning = False
        self.itemsNum = 0        
        self.epsilon = 0.9
        self.Qtable_origx = self.grid_origx+20+(self.grid_columnNum+1)*self.grid_UNIT
        self.Qtable_origy = self.grid_origy
        self.grid_origx_center = self.grid_origx+self.grid_UNIT/2
        self.grid_origy_center = self.grid_origy+self.grid_UNIT/2
        self.Qtable_gridIndex_dict = {}       
        self.show_q_table = pd.DataFrame(columns=list(range(self.n_actions)), dtype=np.float64)                
        self.origDist = 10
        self.agentCentre = np.array([[190,180],[290,180],[390,180]])
        self.warehouseCentre = self.agentCentre+np.array([[0,self.grid_UNIT+self.origDist],\
                                [0,self.grid_UNIT+self.origDist],[0,self.grid_UNIT+self.origDist]])
        self.ObstacleCentre1 = np.array([[725,515],[725,335],[635,695]])

        self.ObstacleCentre2 = np.array([[905,245],[545,245],[995,605]])
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
        self.created_line = []  
        self.lines = []        
        self.grid_endx = self.grid_origx+self.grid_columnNum*self.grid_UNIT
        self.grid_endy = self.grid_origy+self.grid_rowNum*self.grid_UNIT 
        
        self.createMarkA = None
        self.createMarkB = None
        self.linesA = []
        self.linesB = []
#        self.busyA = 0
#        self.busyB = 0
#        
#        self.maze_size = self.grid_columnNum*self.grid_rowNum
#        self.max_memory_size = 8*self.maze_size
#        self.num_actions = 4
#        self.epochs = 8000
#        self.data_size = 32
#        self.tests = 100
        
        
    def resize(self,w, h, w_box, h_box, pil_image):  
      ''''' 
      resize a pil_image 
      '''  
      return pil_image.resize((w_box, h_box), Image.ANTIALIAS)


    def init_widgets(self):
        
        self.cv = Canvas(root, background='white')
        self.cv.pack(fill=BOTH, expand=True)
        # bind events of dragging with mouse
        self.cv.bind('<B1-Motion>', self.move)
        self.cv.bind('<ButtonRelease-1>', self.move_end)
        self.cv.bind("<Button-1>", self.leftClick_handler)
        
        # bind events of double-left-click
        self.cv.bind("<Button-3>", self.rightClick_handler)
        f = ttk.Frame(self.master)
        f.pack(fill=X)
        self.bns = []

        # initialize buttons
        for i, lb in enumerate(('Reset', 'Start trainning', 'Close', 'Save', 'Start Running')):
            bn = Button(f, text=lb, command=lambda i=i: self.choose_type(i))
            bn.pack(side=LEFT, ipadx=8,ipady=5, padx=5)
            self.bns.append(bn)
        self.bns[self.item_type]['relief'] = SUNKEN

        #initialize agent, warehouses and obstacles positions
        self.agentPosition_list = self.setItemsPositionList(self.agentCentre)
        self.warehousePostition_list = self.setItemsPositionList(self.warehouseCentre)
        self.ObstaclePosition_list1 = self.setItemsPositionList(self.ObstacleCentre1)
        self.ObstaclePosition_list2 = self.setItemsPositionList(self.ObstacleCentre1)
        self.ObstaclePosition_list = self.ObstaclePosition_list1 + self.ObstaclePosition_list2
        self.create_items()
        self.itemsNum = self.warehouseCentre.shape[0]+self.ObstacleCentre1.shape[0]+self.ObstacleCentre2.shape[0]+self.agentCentre.shape[0]
        R = self.grid_UNIT
        self.cv.create_text(self.agentCentre[0][0]-R-20,self.agentCentre[0][1],\
                            text = "Agent:",font=('Courier',18))
        self.cv.create_text(self.warehouseCentre[0][0]-R-20,self.warehouseCentre[0][1],\
                            text = "Warehouse:",font=('Couried',18))
        self.cv.create_text(self.grid_origx+250,self.grid_origy-50, text = "Single agent Q-Learning Simulation",\
                            font=('Times',38),fill = 'red')
        self.cv.create_text(self.grid_origx+252,self.grid_origy-52, text = "Single agent Q-Learning Simulation",\
                            font=('Times',38),fill = 'green')        
        
        #draw grids
        self.create_grids(self.grid_origx,self.grid_origy,self.grid_columnNum,self.grid_rowNum,self.grid_UNIT)
        
        for i in range(0,self.grid_rowNum):
            for j in range(0, self.grid_columnNum):
                x = i*self.grid_UNIT+self.grid_origx_center
                y = j*self.grid_UNIT+self.grid_origy_center
                rowIndex = (y-self.grid_origy_center)/self.grid_UNIT
                columnIndex = (x-self.grid_origx_center)/self.grid_UNIT
                self.Qtable_gridIndex_dict[(x,y)]= rowIndex*self.grid_columnNum+columnIndex
                
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
        self.cv.Obstacle = []
        index = 0
        for q in self.ObstacleCentre1:
            bm = self.cv.bms_obs[index]
            t = self.cv.create_image(q[0],q[1],image=bm)
            self.cv.Obstacle.append(t)
            self.AllItemsOrigPosition_list.append(self.cv.coords(t))
            index+=1
        for q in self.ObstacleCentre2:
            bm = self.cv.bms_obs[index]
            t = self.cv.create_image(q[0],q[1],image=bm)
            self.cv.Obstacle.append(t)
            self.AllItemsOrigPosition_list.append(self.cv.coords(t))
            index+=1

        #arriving picture
        pil_image = Image.open('obs5_car.jpg')
        w, h = pil_image.size
        pil_image_resized = self.resize(w, h, w_box, h_box, pil_image)  
        tk_image = ImageTk.PhotoImage(pil_image_resized) 
        self.cv.arriveObsImage.append(tk_image)
        #t = self.cv.create_image(q[0],q[1],image=self.cv.arriveImage[0])              
            
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
        
        #arriving picture
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
        self.cv.coords(self.agentA, self.selected_agent_position[0])
        self.cv.coords(self.agentB, self.selected_agent_position[1])
  
    def reset(self, agentIndex):
        """
        reset the agent to a random valid location
        """
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
        
        if agentIndex!=0 and agentIndex!=1:
            ex = Exception("agentIndex Error in reset()！")
            raise ex
        Obs_list = self.selected_Obstacles_position
        while True:
            new_loc = [random.randrange(self.grid_origx_center,self.grid_endx,self.grid_UNIT),random.randrange(self.grid_origy_center,self.grid_endy,self.grid_UNIT)]
            if new_loc not in Obs_list:
                break
        self.cv.coords(self.selected_agent[agentIndex],new_loc)
        coords = self.cv.coords(self.selected_agent[agentIndex])
        return coords
    
    def reward_a(self, s_, B_s_, s, s_B):
        """
        rewarding scheme
        """
        self.targetA = self.selected_targets[0]
        if s_ == self.cv.coords(self.selected_targets[0]):
            t = self.cv.create_image(s_,image=self.cv.arriveImage[0])
            self.createMarkA = t
            reward = 1           
            done = True
            
        elif s_ in self.selected_Obstacles_position:
            reward = -0.75
            done = False
        
        elif s_ == B_s_:
            reward = -0.75
            done = False

        elif s_==s_B and B_s_== s:
            reward = -0.75
            done = False
            
        else:
            reward = -0.04
            done = False

        return reward, done
    
    def reward_b(self, s_, A_s_, s, s_A):
        """
        rewarding scheme
        """
        self.targetB = self.selected_targets[1]
        if s_ == self.cv.coords(self.selected_targets[1]):
            t = self.cv.create_image(s_,image=self.cv.arriveImage[0])
            self.createMarkB = t
            reward = 1
            done = True
       
        elif s_ in self.selected_Obstacles_position:
            reward = -0.75
            done = False
        elif s_ == A_s_:
            reward = -0.75
            done = False
            
        elif s_==s_A and A_s_== s:
            reward = -0.75
            done = False
        else:
            reward = -0.04
            done = False
   
        return reward, done 
    
    def real_step(self,A_s_,B_s_):
        self.cv.coords(self.selected_agent[0], A_s_)  # move agent
        self.cv.coords(self.selected_agent[1], B_s_)  # move agent
        return
    
    def build_model(self, lr=0.001):
        model = Sequential()
        model.add(Dense(2*self.maze_size,input_shape=(2*self.maze_size,)))
        model.add(PReLU())
        model.add(Dense(2*self.maze_size))
        model.add(PReLU())
        model.add(Dense(2*self.n_actions))
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
        """
        normalize for DQN inputs
        """
        my_array = np.ones(self.maze_size)
        index = self.Qtable_gridIndex_dict[tuple(observation)]
        my_array[int(index)] = 0.5
        return my_array

    def update(self):
        """
        main function for training
        """
        model = self.build_model()
        experience = Experience(model, max_memory=self.max_memory_size)        
        win_historyA = []   # history of win/lose game
        win_historyB = []   # history of win/lose game
        win_rateA = 0.0
        win_rateB = 0.0
        hsize = self.maze_size//2
        start_time = datetime.datetime.now()
        observation_A = self.cv.coords(self.agentA)
        observation_B = self.cv.coords(self.agentB)
        loss_list = []
        total_reward_listA = []
        total_reward_listB = []
        avg_reward_listA= []
        avg_reward_listB= []
        episodeA = 0
        episodeB = 0
        total_rewardA = 0
        total_rewardB = 0
        visitedA = set()
        visitedB = set()
        stepCountA = 0
        stepCountB = 0
        while True:
            self.labelHello = Label(self.cv, text = "episodeA: %s"%str(episodeA), font=("Helvetica", 10), width = 10, fg = "red", bg = "white")
            self.labelHello.place(x=200, y=550, anchor=NW) 
            self.labelHello = Label(self.cv, text = "episodeB: %s"%str(episodeB), font=("Helvetica", 10), width = 10, fg = "blue", bg = "white")
            self.labelHello.place(x=200, y=580, anchor=NW)   
            self.render()
            normal_observation_A = self.normalize_input(observation_A)
            normal_observation_B = self.normalize_input(observation_B)
            observation = np.hstack((np.array(normal_observation_A),np.array(normal_observation_B))).reshape(1,-1)
            action_A = experience.choose_action(observation,0)
            action_B = experience.choose_action(observation,1)
            A_observation_ = self.calcu_next_state(observation_A,action_A)
            B_observation_ = self.calcu_next_state(observation_B,action_B)
            normal_observation_A_ = self.normalize_input(A_observation_)
            normal_observation_B_ = self.normalize_input(B_observation_)
            reward_A, done_A = self.reward_a(A_observation_, B_observation_, observation_A, observation_B)
            reward_B, done_B = self.reward_b(B_observation_, A_observation_, observation_B, observation_A)
            self.real_step(A_observation_,B_observation_)
            stepCountA+=1 
            stepCountB+=1

            if tuple(A_observation_) in visitedA:
                reward_A -= 0.25
            if tuple(B_observation_) in visitedB:
                reward_B -= 0.25
            if observation_A==A_observation_:
                reward_A = reward_A - 0.8                
            if observation_B==B_observation_:
                reward_B = reward_B - 0.8
            if done_A == True:
                win_historyA.append(1)
            if done_B == True:
                win_historyB.append(1)            
            total_rewardA += reward_A 
            total_rewardB += reward_B
            if total_rewardA < -0.5*self.maze_size:
                done_A = True
                win_historyA.append(0)
            if total_rewardB < -0.5*self.maze_size:
                done_B = True
                win_historyB.append(0)                    
            episode_total = [normal_observation_A, action_A, reward_A, normal_observation_A_, done_A,\
                         normal_observation_B, action_B, reward_B, normal_observation_B_, done_B]
            experience.remember(episode_total)
                
            inputs, targets = experience.get_data(data_size=self.data_size)
            h = model.fit(
                inputs,
                targets,
                epochs=8,
                batch_size=16,
                verbose=0,
            )
            loss = model.evaluate(inputs, targets, verbose=0)              
            observation_A = A_observation_
            observation_B = B_observation_
            # break while loop when end of this episode
            if done_A:
                if episodeA>self.episode and episodeB>self.episode:
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
                if episodeA>self.episode and episodeB>self.episode:
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
        print("average rewards per episode_A",sum(total_reward_listA)/len(total_reward_listA))        
        print("average rewards per episode_B",sum(total_reward_listB)/len(total_reward_listB))
        self.learning = False
        self.reset(0)
        self.reset(1)
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
        plt.legend(loc='upper right')
        plt.show()
        
        plt.figure()
        plt.title('Average Rewards over 100 Episode')
        plt.xlabel('Episode number')
        plt.ylabel('Rewards')
        plt.plot(avg_reward_listA,label='agentA')
        plt.plot(avg_reward_listB,label='agentB')
        plt.legend(loc='upper right')
        plt.show()
        model.save("multi-agent_DQN.h5"%string)

        
    def new_reward_a(self, s_, B_s_, s, s_B):
        """
        rewarding scheme for running tests
        """  
        self.targetA = self.selected_targets[0]
        if s_ == self.cv.coords(self.selected_targets[0]):
            t = self.cv.create_image(s_,image=self.cv.arriveImage[0])
            self.createMarkA = t
            reward = 0           
            
        elif s_ in self.selected_Obstacles_position:
            reward = -2
        
        elif s_ == B_s_:
            reward = -2
            
        elif s_==s_B and B_s_== s:
            reward = -2
          
        else:
            reward = 0
        return reward
    
    def new_reward_b(self, s_, A_s_, s, s_A):
        """
        rewarding scheme for running tests
        """  
        self.targetB = self.selected_targets[1]
        if s_ == self.cv.coords(self.selected_targets[1]):
            t = self.cv.create_image(s_,image=self.cv.arriveImage[0])
            self.createMarkB = t
            reward = 0
            
        elif s_ in self.selected_Obstacles_position:
            reward = -2

        elif s_ == A_s_:
            reward = -2

        elif s_==s_A and A_s_== s:
            reward = -2            
        else:
            reward = 0            
        return reward
    
    
    def run(self):
        """
        main function for running tests
        """
        test = 0
        rewardsA = []
        rewardsB = []
        string = str(self.episode)
        self.run_model = load_model("multi-agent_DQN_example.h5")
        print(self.run_model.summary())       
        action_B = -1
        action_A = -1
        action_list_A = []
        action_list_B = []
        observation_A = self.cv.coords(self.agentA)        
        observation_B = self.cv.coords(self.agentB)        
        doneA = 0
        doneB = 0
        total_rewardA = 0
        total_rewardB = 0
        visitedA = [observation_A]
        visitedB = [observation_B]
        win_countA = 0
        win_countB = 0
        win_listA = []
        win_listB = []
        while True:
            self.labelHello = Label(self.cv, text = "Test:%s"%str(test),font=("Helvetica", 10), width = 10, fg = "blue", bg = "white")
            self.labelHello.place(x=self.agentCentre[0][0]-150, y=self.agentCentre[0][1]+500, anchor=NW)
            if self.cv.coords(self.agentA) == self.cv.coords(self.targetA):
                doneA = 1
            if self.cv.coords(self.agentB) == self.cv.coords(self.targetB):
                doneB = 1
            time.sleep(self.timeDelay)
            normal_observation_A = self.normalize_input(observation_A)
            normal_observation_B = self.normalize_input(observation_B)
            observation = np.hstack((np.array(normal_observation_A),np.array(normal_observation_B))).reshape(1,-1)
            if doneA != 1:
                action_A = np.argmax(self.run_model.predict(observation)[0][0:4])
                action_list_A.append(action_A)
                A_observation_ = self.calcu_next_state(observation_A,action_A)
                if A_observation_ == self.cv.coords(self.targetA):
                    win_listA.append(1)

            if doneB != 1:
                action_B = np.argmax(self.run_model.predict(observation)[0][4:8])
                action_list_B.append(action_B)
                B_observation_ = self.calcu_next_state(observation_B,action_B)
                if B_observation_ == self.cv.coords(self.targetB):
                    win_listB.append(1)

            reward_A = self.new_reward_a(A_observation_, B_observation_, observation_A, observation_B)
            reward_B = self.new_reward_b(B_observation_, A_observation_, observation_B, observation_A)            
            if B_observation_ in visitedB:
                reward_B -= 0.5
            else:
                visitedB.append(B_observation_)
            
            if A_observation_ in visitedA:
                reward_A -= 0.5
            else:
                visitedA.append(A_observation_)            
            if doneA:
                A_observation_ = self.cv.coords(self.targetA)
            if doneB:
                B_observation_ = self.cv.coords(self.targetB)
            self.real_step(A_observation_,B_observation_)
            total_rewardA += reward_A
            total_rewardB += reward_B
            if total_rewardA < -1:
                doneA = 1
            if total_rewardB < -1:
                doneB = 1
            if doneA != 1:
                lineA = self.cv.create_line(observation_A[0], observation_A[1],
                      A_observation_[0], A_observation_[1],
                      fill='red',
                      arrow=LAST,
                      arrowshape=(10,20,8),
                      dash=(4, 4)  
                      )
                self.linesA.append(lineA)
                visitedA.append(A_observation_)

            if doneB != 1:
                lineB = self.cv.create_line(observation_B[0], observation_B[1],
                      B_observation_[0], B_observation_[1],
                      fill='blue',
                      arrow=LAST,
                      arrowshape=(10,20,8),
                      dash=(4, 4) 
                      )
                self.linesB.append(lineB)
                visitedB.append(B_observation_)
            observation_A = A_observation_
            observation_B = B_observation_
                        
            if doneA:
                action_A = -1
                visitedA = []
            if doneB:
                action_B = -1
                visitedB = []
            
            if doneA and doneB:
                total_rewardA += 1
                total_rewardB += 1
                rewardsA.append(total_rewardA)  
                rewardsB.append(total_rewardB)
                if total_rewardA == 1:
                    win_countA += 1
                if total_rewardB == 1:
                    win_countB += 1                
                total_rewardA = 0
                total_rewardB = 0
                reward_A = 0
                reward_B = 0
                self.reset(0)
                self.reset(1)
                doneA = 0
                doneB = 0
                observation_A = self.cv.coords(self.agentA)        
                observation_B = self.cv.coords(self.agentB)
                visitedA = []
                visitedB = []
                test += 1
            if test > self.tests:
                self.labelHello = Label(self.cv, text = "running end!!", font=("Helvetica", 10), width = 10, fg = "red", bg = "white")
                self.labelHello.place(x=250, y=750, anchor=NW)
                break
        
        print("win_countA",sum(win_listA))
        print("win_countB",sum(win_listB))
        plt.figure()
        plt.title('Score per Episode')
        plt.xlabel('Episode number')
        plt.ylabel('Score')
        plt.plot(rewardsA,label='agentA')
        plt.plot(rewardsB,label='agentB')
        plt.legend(loc='upper right')
        plt.show()
    
    def start_learning(self):
        self.selected_agent = []
        self.selected_targets = []
        self.selected_Obstacles = []
        self.selected_agent_position = []
        self.selected_Obstacles_position = []
        
        for item in range(1,self.itemsNum+1):
            p = self.cv.coords(item)
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

        self.targetA = self.selected_targets[0]
        self.targetB = self.selected_targets[1]
        self.agentA = self.selected_agent[0]
        self.agentB = self.selected_agent[1]
        if len(self.selected_agent)==1 or len(self.selected_agent)>2:
            tkinter.messagebox.showinfo("INFO","Please choose TWO agents for trainning！")
        elif len(self.selected_targets)==1 or len(self.selected_targets)>2:
            tkinter.messagebox.showinfo("INFO","Please choose TWO targets for trainning！")
        else:
            self.t = threading.Timer(self.timeDelay, self.update)
            self.t.start()
            self.learning = True
            
            
            
    def restart(self):
        self.cv.coords(self.agentA,self.agentCentre[0])
        self.cv.coords(self.agentB,self.agentCentre[1])
        
    def calcu_next_state(self,loc,action):
        """
        calculate next state based on location and action
        """
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
        for c in range(origx, origx+(column+1)*UNIT, UNIT):
            x0, y0, x1, y1 = c, origy, c, origy+row*UNIT
            self.cv.create_line(x0, y0, x1, y1,width=2)
        for r in range(origy, origy+(row+1)*UNIT, UNIT):
            x0, y0, x1, y1 = origx, r, origx+row*UNIT, r
            self.cv.create_line(x0, y0, x1, y1,width=2)
    
    def choose_type(self, i):
        """
        function of clicking different button
        """
        for b in self.bns: b['relief'] = RAISED
        self.bns[i]['relief'] = SUNKEN
        self.item_type = i        
        if self.item_type==1:
        #            start training
            self.start_learning()            
            self.bns[i]['relief'] = RAISED
        elif self.item_type==2:
        #            close simulation tool
            os._exit(0)
        elif self.item_type==3:
        #           save q_table
            temp_s=str(self.cv.coords(self.target))+str(self.selected_Obstacles_position)
            self.RL.q_table.to_csv("single_qtable_%s.csv"%temp_s, index_label="index_label")
            print("SAVED!!!")
            self.labelHello = Label(self.cv, text = "table saved!!", font=("Helvetica", 10), width = 10, fg = "red", bg = "white")
            self.labelHello.place(x=350, y=750, anchor=NW)
        elif self.item_type==0:
            self.button_reset()
        elif self.item_type==4:
        #            start running tests
            self.start_running()
        elif self.item_type==5:
            self.restart()
                
    def start_running(self):
        self.selected_agent = []
        self.selected_targets = []
        self.selected_Obstacles = []
        self.selected_agent_position = []
        self.selected_Obstacles_position = []   
        self.task_list = []
        self.task_num_list = []
        for item in range(1,self.itemsNum+1):
            p = self.cv.coords(item)
            if p[0]>=self.grid_origx and p[1]>=self.grid_origy:
                if item in range(self.agentItemIndex[0],self.agentItemIndex[1]+1):
                    self.selected_agent.append(item)
                    self.selected_agent_position.append(p)
                elif item in range(self.WarehouseItemIndex[0],self.WarehouseItemIndex[1]+1):
                    self.selected_targets.append(item)
                elif item in range(self.ObstacleItemIndex[0],self.ObstacleItemIndex[1]+1):
                    self.selected_Obstacles.append(item)                    
                    self.selected_Obstacles_position.append(p)

        if len(self.selected_agent)<=1 or len(self.selected_agent)>2:
            tkinter.messagebox.showinfo("INFO","Please place TWO agents on map!")
        elif len(self.selected_targets)==0 or len(self.selected_targets)>2:
            tkinter.messagebox.showinfo("INFO","Please choose TWO terminals!")
        else:
            self.agentA = self.selected_agent[0]
            self.agentB = self.selected_agent[1]
            self.targetA = self.selected_targets[0]
            self.targetB = self.selected_targets[1]
            self.labelHello = Label(self.cv, text = "start running!!", font=("Helvetica", 10), width = 10, fg = "red", bg = "white")
            self.labelHello.place(x=250, y=750, anchor=NW)       
            self.t = threading.Timer(self.timeDelay, self.run)
            self.t.start()
            self.learning = True
                 
    def rightClick_handler(self, event):
        self.start_learning()
        
    def leftClick_handler(self, event):
        if self.learning:
            print("Learing on going!")
        else:
            for item in range(1,self.itemsNum+1):
                position = self.cv.coords(item)
                R = self.grid_UNIT/2
                p = [position[0]-R,position[1]-R,position[0]+R,position[1]+R]
                if event.x>=p[0] and event.x<=p[2] and \
                    event.y>=p[1] and event.y<=p[3]:
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
            self.adjust_items_into_grids(event)
            self.choose_item = None              
       
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
