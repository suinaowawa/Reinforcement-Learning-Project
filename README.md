# Reinforcement-Learning-Project
Visualization application for multi-agent reinforcement learning algorithms

## Project Name: Real Time IoT for Distributed Machine Learning
![Alt text](https://github.com/suinaowawa/Reinforcement-Learning-Project/blob/master/figures/design%20interface.png)
## Project Introduction:
This project design and implement a self-tracking multi-agent system, which can be used for goods-delivering vehicles in the low-cost intelligent warehouse distribution system. Different reinforcement learning algorithms(e.g. Multi-agent Q-learning, DQN) are compared and tested in simulation and hardware grid world environment. 
## Project Demo:
#### _**To quickly understand how this program works, watch these video clips!**_
- Single agent Q-learning training process: https://youtu.be/brZHhvpx5PA
- Single agent Q-learning testing: https://youtu.be/rxgr_9u862Q
- Multi-agent DQN training process: https://youtu.be/kuo40hc72-Y
- Multi-agent DQN testing: https://youtu.be/M5OzCxlyQDc
- Prototype system based on multi-agent DQN: https://youtu.be/Bc_-8lmLLPA
![Alt text](https://github.com/suinaowawa/Reinforcement-Learning-Project/blob/master/figures/2.PNG)
## Main packages been used for software:
-	Tkinter (8.6.8) a graphical user interface (GUI)tool used to build the simulation software
-	Pandas (0.23.4) a data analysis tool used for Q-table
-	Keras (2.2.4) is a high-level neural networks API (Application Programming Interface) used for building Deep Q network.
-	Numpy (1.15.4) used for scientific computing
-	Matplotlib (3.0.2) used to plot testing results’ curves
-	Threading used to do multitasking between visualization and computation process.
-	Pillow (5.3.0) used to display images of cars, warehouses etc. on simulation environment.

## How to run the code
#### Simulation code
In the ‘multi-agent RL’ folder, there are 4  .py files. ‘RL_brain’ is the file that manage the Q tables storing. ‘Experience_normal_two’ is the experience replay code. The two files have been modified to fit in a two agents’ environment.
1.	Run file ‘multi-agent_Qlearning_final’ and ‘multi-agent_DQN_final’ to use the simulation tool. 
2.	After running the script, we can see the user interface, drag any two warehouses on left to the desired spot on grid map, then drag any two cars to the grid map.
3.	Click on ‘start training’ button to visualize the training process of cars, adjust the updating speed by tuning parameter ‘self.timeDelay’ in line 57.
4.	After trained for pre-defined episodes, save the model of Qlearn by clicking ‘Save’ button, DQN model will be saved automatically.
5.	Visualize the testing process on a trained model by drag the warehouses to the exact same places on grid map and place two cars at arbitrary locations. Click ‘start running’ button.
6.	In the folder, we save the pre-trained model for map where the first warehouse is located on the bottom-left corner, and the second warehouse on the up-right corner. Use this map setting, you can directly see the testing of the cars.
##### Note: If you want to close the simulation tool, click ‘close’ button. When there is an error occur on the ‘Console’ at the first time of closing the program, ‘restart the kernel’ to prevent the same error from happening again.
Similarly, in the ‘single agent RL’ folder, you can find the scripts for single agent Q-learning and DQN simulation. Run the script in the same way above, except for this time, only drag one warehouse and one car on the grid map. The pre-trained model corresponds to the map when there is a warehouse at the bottom-left corner. 



#### Simulation documents
In the ‘simulation_documents’ folder, training log corresponds to each simulation experiment in this report can be found. 

#### Prototype system code
In the ‘prototype system’ folder, there is two .py file, the ‘Experience normal.py’ is modified based on the samyzaf [44]. 
1.	Run the file ‘prototype_system.py’.
2.	Drag three cars at arbitrary locations on grid map.
3.	Click ‘start running’ button and click ‘Terminal A’, ‘Terminal B’, ‘Terminal C’ as input tasks at any time.
4.	This prototype system uses the final trained model that wrote in the report, if you want to train another model, click ‘start training’ after drag all three cars on map, and the model will be saved automatically.
5.	Parameter ‘self.timeDelay’  is tunable in line 67, to adjust the speed of cars’ movement.
##### Note: If you want to close the simulation tool, click ‘close’ button. When there is an error occur on the ‘Console’ at the first time of closing the program, ‘restart the kernel’ to prevent the same error from happening again.
