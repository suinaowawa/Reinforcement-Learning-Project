# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 13:34:07 2019

@author: Yue
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 01:17:35 2019

@author: Yue
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 20:47:55 2019

@author: Yue
"""


import numpy as np




class Experience(object):
    def __init__(self, model, max_memory=100, discount=0.95, epsilon=0.9):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]
        self.epsilon = epsilon
        self.actions = [0, 1, 2, 3]
    def remember(self, episode):
        # episode = [envstate, action, reward, envstate_next, game_over]
        # memory[i] = episode
        # envstate == flattened 1d maze cells info, including rat cell (see method: observe)
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, envstate):
#        print("Q_sa_total==============",self.model.predict(envstate)[0])
        return self.model.predict(envstate)[0]

    def get_data(self, data_size=10):
#        env_size = 4   # envstate 1d size (1st element of episode)
        env_size = 64
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            
            
#            print("envshape",envstate.shape)
#            print("@@@@In get_data: envstate_A = "+str(envstate))
#            print("@@@@In get_data: envstate_B = "+str(envstate_B))
#            print("@@@@In get_data: envstate_next_A = "+str(envstate_next))
#            print("@@@@In get_data: envstate_next_B = "+str(envstate_next_B))
           
            inputs[i] = envstate
            # There should be no target values for actions not taken.
            targets[i] = self.predict(envstate.reshape(1,-1))
#            targets = targets[i]
        
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
#            envstate_next = np.hstack((envstate_next_A,envstate_next_B))
            Q_sa_total = self.predict(envstate_next.reshape(1,-1))
#            print("Q_sa_total==============",Q_sa_total)
            Q_sa = np.max(Q_sa_total)
#            Q_sa_B = np.max(Q_sa_total[4:])
            if game_over:
                targets[i][action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i][action] = reward + self.discount * Q_sa
                

                
#            targets[i] = targets
#            print("@@@@In get_data: targets[i] = "+str(targets[i]))              
        return inputs, targets
    
    def choose_action(self, observation):
        
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            
            action = np.argmax(self.predict(observation))
           
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action