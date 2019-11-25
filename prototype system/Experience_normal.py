# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:57:57 2019

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
        env_size = 16*16*3
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate_A, action_A, reward_A, envstate_next_A, game_over_A, \
            envstate_B, action_B, reward_B, envstate_next_B, game_over_B, \
            envstate_C, action_C, reward_C, envstate_next_C, game_over_C = self.memory[j]
            
            envstate = np.hstack((envstate_A,envstate_B,envstate_C))
#            print("envshape",envstate.shape)
#            print("@@@@In get_data: envstate_A = "+str(envstate_A))
#            print("@@@@In get_data: envstate_B = "+str(envstate_B))
#            print("@@@@In get_data: envstate_next_A = "+str(envstate_next_A))
#            print("@@@@In get_data: envstate_next_B = "+str(envstate_next_B))
#           
            inputs[i] = envstate
            # There should be no target values for actions not taken.
            targets[i] = self.predict(envstate.reshape(1,-1))
            targetsA = targets[i][0:4]
            targetsB = targets[i][4:8]
            targetsC = targets[i][8:]
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            envstate_next = np.hstack((envstate_next_A,envstate_next_B,envstate_next_C))
            Q_sa_total = self.predict(envstate_next.reshape(1,-1))
#            print("Q_sa_total==============",Q_sa_total)
            Q_sa_A = np.max(Q_sa_total[0:4])
            Q_sa_B = np.max(Q_sa_total[4:8])
            Q_sa_C = np.max(Q_sa_total[8:])
            if game_over_A:
                targetsA[action_A] = reward_A
            else:
                # reward + gamma * max_a' Q(s', a')
                targetsA[action_A] = reward_A + self.discount * Q_sa_A
                
            if game_over_B:
                targetsB[action_B] = reward_B
            else:
                # reward + gamma * max_a' Q(s', a')
                targetsB[action_B] = reward_B + self.discount * Q_sa_B
            if game_over_C:
                targetsC[action_C] = reward_C
            else:
                # reward + gamma * max_a' Q(s', a')
                targetsC[action_C] = reward_C + self.discount * Q_sa_C
                
            targets[i] = np.hstack((targetsA,targetsB,targetsC))
#            print("@@@@In get_data: targets[i] = "+str(targets[i]))              
        return inputs, targets
    
    def choose_action(self, observation ,agentIndex):
        
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            if agentIndex == 0:
                action = np.argmax(self.predict(observation)[0:4])
            if agentIndex == 1:
                action = np.argmax(self.predict(observation)[4:8])
            if agentIndex == 2:
                action = np.argmax(self.predict(observation)[8:])
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action