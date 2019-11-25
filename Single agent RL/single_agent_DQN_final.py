# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 15:29:43 2019

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
from Experience1 import Experience
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import json

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
        self.episode = 1000
#        define number of tests to run
        self.tests = 100
#        set a small amount of delay (second) to make sure tkinter works properly
#        if want to have a slower visulazation for testing, set the delay to larger values
        self.timeDelay = 0.005
        
        
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
#        print("npTemp=",npTemp)
        h_u = self.grid_UNIT/2
        npHalfUnit = np.array([-h_u,-h_u,h_u,h_u])
        hs = npHalfUnit
        for diam in range(1,itemCentre.shape[0]):
            hsTemp = np.vstack((npHalfUnit,hs))
            hs = hsTemp
#            print("hs=",hs)
        return (npTemp-hs).tolist()
    def button_reset(self):
        time.sleep(self.timeDelay)
        if self.createMark is not None:
            self.cv.delete(self.createMark)
        for line in self.created_line:
            self.cv.delete(line)
        self.cv.coords(self.agent, self.selected_agent_position)

        coords = self.cv.coords(self.agent)
        return coords
    
    def reset(self):
        """
        reset the agent to a random valid location
        """
        if self.lines!=[]:
            for line in self.lines:
                self.cv.delete(line)
        Obs_list = self.ObstaclePosition_list
        while True:
            new_loc = [random.randrange(self.grid_origx_center,self.grid_rowNum*self.grid_UNIT+self.grid_origx_center,self.grid_UNIT),random.randrange(self.grid_origy_center,self.grid_columnNum*self.grid_UNIT+self.grid_origy_center,self.grid_UNIT)]
            if new_loc not in Obs_list:
                break
        self.cv.coords(self.selected_agent[0],new_loc )
        coords = self.cv.coords(self.selected_agent[0])
        return coords
    
    def reward(self, s_, s):
        """
        rewarding scheme
        """
        self.target = self.selected_targets[0]
        if s_ == self.cv.coords(self.selected_targets[0]):
            t = self.cv.create_image(s_,image=self.cv.arriveImage[0])
            self.createMarkA = t
            reward = 1
            done = True

            
        elif s_ in self.selected_Obstacles_position:
            reward = -0.75
            done = False
          
        else:
            reward = -0.04
            done = False

        return reward, done
    
    
    def real_step(self,s_):
        self.cv.coords(self.selected_agent[0], s_)  # move agent       
        return
    
    def build_model(self, lr=0.001):
        model = Sequential()
        model.add(Dense(64,input_shape=(64,)))
        model.add(PReLU())
        model.add(Dense(32))
        model.add(PReLU())
        model.add(Dense(self.n_actions))
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
        win_history = [] 
        episode = 0
        loss = 0.0
        start_time = datetime.datetime.now()
        # initial observation
        observation = self.cv.coords(self.agent)
        loss_list = []
        total_reward_list = []
        avg_reward_list= []
        action = -1
        visited = set()
        total_reward = 0
        start_time = datetime.datetime.now()
        self.labelHello = Label(self.cv, text = "start training!", font=("Helvetica", 10), width = 10, fg = "red", bg = "white")
        self.labelHello.place(x=200, y=750, anchor=NW)
        stepCount = 0
        while True:
            self.labelHello = Label(self.cv, text = "episode: %s"%str(episode), font=("Helvetica", 10), width = 10, fg = "blue", bg = "white")
            self.labelHello.place(x=200, y=550, anchor=NW)             
            self.render()
            visited.add(tuple(observation))
            stepCount+=1    
            normal_observation = self.normalize_input(observation)
            normal_observation = normal_observation.reshape(1,-1)
            action = experience.choose_action(normal_observation)
            observation_ = self.calcu_next_state(observation,action)                
            normal_observation_ = self.normalize_input(observation_)                
            reward, done = self.reward(observation_,observation)
            self.real_step(observation_)
            if tuple(observation_) in visited:
                reward -= 0.25
            if observation==observation_:
                reward = reward - 0.8
            if done == True:
                win_history.append(1)                
            total_reward += reward 
            if total_reward < -0.5*self.maze_size:
                done = True
                win_history.append(0)               
            episode_total = [normal_observation, action, reward, normal_observation_, done]
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
            observation = observation_            
            if done:
                if episode > self.episode:
                    break
                else:
                    observation = self.reset()
                    dt = datetime.datetime.now() - start_time
                    t = self.format_time(dt.total_seconds())
                    total_reward_list.append(total_reward)
                    loss_list.append(loss)
                    
                    if len(total_reward_list) > 100:
                        avg_reward = sum(total_reward_list[-100:])/100
                        avg_reward_list.append(avg_reward)
                        template = "Episode: {:03d}/{:d} | Loss: {:.4f} | StepCount: {:d} | Win rate: {:.3f} | Total rewards: {:.3f} | Average rewards: {:.3f} | time: {}"
                        print(template.format(episode, self.episode, loss, stepCount, sum(win_history)/len(win_history), total_reward, avg_reward, t))
                    else:
                        template = "Episode: {:03d}/{:d} | Loss: {:.4f} | StepCount: {:d} | Win rate: {:.3f} | Total rewards: {:.3f} | time: {}"
                        print(template.format(episode, self.episode, loss, stepCount, sum(win_history)/len(win_history), total_reward, t))                    
                    episode+=1
                    stepCount = 0
                    total_reward = 0
                    visited = set()
                    done = 0
            
        # end of training
        print('training over!')
        self.labelHello = Label(self.cv, text = "training end!", font=("Helvetica", 10), width = 10, fg = "red", bg = "white")
        self.labelHello.place(x=200, y=750, anchor=NW)
        print("total_win_rate",sum(win_history)/len(win_history))
        print("total_time",t)
        print("average rewards per episode",sum(total_reward_list)/len(total_reward_list))
        self.learning = False
        self.reset()
        plt.figure()
        plt.title('Loss per Episode')
        plt.xlabel('Episode number')
        plt.ylabel('Loss')
        plt.plot(loss_list)
        plt.show()
        plt.figure()
        plt.title('Rewards per Episode')
        plt.xlabel('Episode number')
        plt.ylabel('Rewards')
        plt.plot(total_reward_list)
        plt.show()
        
        plt.figure()
        plt.title('Average Rewards over 100 Episode')
        plt.xlabel('Episode number')
        plt.ylabel('Rewards')
        plt.plot(avg_reward_list)
        plt.show()
        
        h5file = "test" + ".h5"
        json_file = "test" + ".json"
        model.save_weights(h5file, overwrite=True)
        model.save('single_agent_DQN.h5')
        with open(json_file, "w") as outfile:
            json.dump(model.to_json(), outfile)
        end_time = datetime.datetime.now()
        dt = end_time - start_time
        seconds = dt.total_seconds()
        t = self.format_time(seconds)

        return seconds
    
                
    def new_reward(self, s_, s):
        """
        rewarding scheme for running tests
        """       
        if s_ == self.cv.coords(self.selected_targets[0]):
            t = self.cv.create_image(s_,image=self.cv.arriveImage[0])
            self.createMark = t
            reward = 0            
        elif s_ in self.selected_Obstacles_position:
            reward = -2
        else:
            reward = 0            
        return reward            
                
                
    def run(self):
        """
        main function for running tests
        """
        self.run_model = load_model('test_model_single_916_2000.h5')
        print("model loaded!!")
        print(self.run_model.summary())        
        action = -1        
        action_list = []
        observation = self.cv.coords(self.agent)        
        done = 0
        test = 0
        total_reward = 0
        visited = [observation]
        rewards = []
        win_count = 0
        while True:
            self.labelHello = Label(self.cv, text = "Test:%s"%str(test),font=("Helvetica", 10), width = 10, fg = "blue", bg = "white")
            self.labelHello.place(x=self.agentCentre[0][0]-150, y=self.agentCentre[0][1]+500, anchor=NW)
            time.sleep(self.timeDelay)
            normal_observation = self.normalize_input(observation)
            normal_observation = np.array(normal_observation).reshape(1,-1)
            action = np.argmax(self.run_model.predict(normal_observation)[0])
            action_list.append(action)            
            observation_ = self.calcu_next_state(observation,action)
            reward = self.new_reward(observation_, observation)            
            if observation_ in visited:
                reward -= 0.5
            else:
                visited.append(observation_)            
            if done:
                observation_ = self.cv.coords(self.target)           
            self.real_step(observation_)
            total_reward += reward
            if total_reward < -1:
                done = 1
            if done != 1:
                line = self.cv.create_line(observation[0], observation[1],
                      observation_[0], observation_[1],
                      fill='red',
                      arrow=LAST,
                      arrowshape=(10,20,8),
                      dash=(4, 4)
                      )
                self.lines.append(line)
            observation = observation_    
            if self.cv.coords(self.agent) == self.cv.coords(self.target):
                done = 1            
            if done:
                action = -1
                visited = []        
                total_reward += 1
                if total_reward == 1:
                    win_count += 1
                rewards.append(total_reward)                  
                total_reward = 0                
                self.reset()                
                done = 0                
                observation = self.cv.coords(self.agent)        
                test += 1
            if test > self.tests:
                self.labelHello = Label(self.cv, text = "running end!!", font=("Helvetica", 10), width = 10, fg = "red", bg = "white")
                self.labelHello.place(x=250, y=750, anchor=NW)
                break
        print("win_count",win_count)
        plt.figure()
        plt.title('Score per Episode')
        plt.xlabel('Episode number')
        plt.ylabel('Score')
        plt.plot(rewards)       
        plt.show()    

    def start_learning(self):
        """
        initialization for training process
        """
        self.selected_agent = []
        self.selected_targets = []
        self.selected_Obstacles = []
        self.selected_agent_position = []
        self.selected_Obstacles_position = []
        
        for item in range(1,self.itemsNum+1):
           
            p = self.cv.coords(item)
            
            if p[0]>=self.grid_origx and p[1]>=self.grid_origy:
                if item in range(self.agentItemIndex[0],self.agentItemIndex[1]+1):
                    self.selected_agent.append(item)
                    self.selected_agent_position = p
                elif item in range(self.WarehouseItemIndex[0],self.WarehouseItemIndex[1]+1):
                    self.selected_targets.append(item)
                elif item in range(self.ObstacleItemIndex[0],self.ObstacleItemIndex[1]+1):
                    self.selected_Obstacles.append(item)
                    self.selected_Obstacles_position.append(p)
        
        if len(self.selected_agent)==0 or len(self.selected_agent)>1:
            tkinter.messagebox.showinfo("INFO","Please choose ONE agent for trainning！")
        elif len(self.selected_targets)==0 or len(self.selected_targets)>1:
            tkinter.messagebox.showinfo("INFO","Please choose ONE target for trainning！")
        else:
            self.agent = self.selected_agent[0]
            self.target = self.selected_targets[0]
            
            self.t = threading.Timer(self.timeDelay, self.update)
            self.t.start()
            self.learning = True
    
        
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
        # create grids
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
        """
        initialization for testing
        """
        self.selected_agent = []
        self.selected_targets = []
        self.selected_Obstacles = []
        self.selected_agent_position = []
        self.selected_Obstacles_position = []
        self.selected_targets_position = []
       
        for item in range(1,self.itemsNum+1):
            p = self.cv.coords(item)
            if p[0]>=self.grid_origx and p[1]>=self.grid_origy:
                if item in range(self.agentItemIndex[0],self.agentItemIndex[1]+1):
                    self.selected_agent.append(item)
                    self.selected_agent_position = p
                elif item in range(self.WarehouseItemIndex[0],self.WarehouseItemIndex[1]+1):
                    self.selected_targets.append(item)
                    self.selected_targets_position = p
                elif item in range(self.ObstacleItemIndex[0],self.ObstacleItemIndex[1]+1):
                    self.selected_Obstacles.append(item)                    
                    self.selected_Obstacles_position.append(p)                      
        
        if len(self.selected_agent)<=0 or len(self.selected_agent)>1:
            tkinter.messagebox.showinfo("INFO","Please place ONE agent on map!")
        elif len(self.selected_targets)==0 or len(self.selected_targets)>1:
            tkinter.messagebox.showinfo("INFO","Please choose ONE terminal!")
        else:
            self.agent = self.selected_agent[0]
            self.target = self.selected_targets[0]           
            self.t = threading.Timer(self.timeDelay, self.run)
            self.t.start()
            self.learning = True   
            

                 
    def rightClick_handler(self, event):
        self.start_learning()
        
    def leftClick_handler(self, event):
        """
        bind events of choosing warehouse
        """

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
                #return to original position if not drag near grids
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
root.title("single agent DQN")
root.attributes("-fullscreen", False)
app = App(root)
root.bind('<Delete>', app.delete_item)
root.mainloop()


#print('End of program')
## ===================================
#sys.stdout = origin
#f.close()
