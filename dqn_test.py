import gymnasium as gym
import pygame
import PIL
from torchvision.transforms.functional import to_pil_image
from input_testing import prepare_state_into_tensor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
import torch
from torchvision import io, transforms, models
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import gc
from torch.cuda.amp import autocast, GradScaler
from collections import deque
import copy
from gymnasium.envs.box2d.car_racing import CarRacing as _BaseCarRacing

def make_action(action_num):
    # steering [-1, -0.5, 0, 0.5, 1]
    # gas [0, 0.5, 1]
    # brake [0, 0.5, 1]

    action_space = [
        # Steering = -1.0
        [-1.0, 0.0, 0.0], [-1.0, 0.5, 0.0], [-1.0, 1.0, 0.0],  # Gas
        [-1.0, 0.0, 0.5], [-1.0, 0.0, 1.0],  # Brake

        # Steering = -0.5
        [-0.5, 0.0, 0.0], [-0.5, 0.5, 0.0], [-0.5, 1.0, 0.0],  # Gas
        [-0.5, 0.0, 0.5], [-0.5, 0.0, 1.0],  # Brake

        # Steering = 0.0
        [0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0],  # Gas
        [0.0, 0.0, 0.5], [0.0, 0.0, 1.0],  # Brake

        # Steering = 0.5
        [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 1.0, 0.0],  # Gas
        [0.5, 0.0, 0.5], [0.5, 0.0, 1.0],  # Brake

        # Steering = 1.0
        [1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, 1.0, 0.0],  # Gas
        [1.0, 0.0, 0.5], [1.0, 0.0, 1.0],  # Brake
    ]

    return np.array(action_space[action_num])







class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.conv1 = nn.Conv2d(1,2,kernel_size=3, stride=2, padding=1) #(1,80,80) -> (2,40,40)
        self.conv2 = nn.Conv2d(2,4,kernel_size=3, stride=2, padding=1) #(4,20,20)
        self.conv3 = nn.Conv2d(4,8,kernel_size=3, stride=2, padding=1) #(8,10,10)
        self.conv4 = nn.Conv2d(8,16,kernel_size=3, stride=2, padding=1) #(16,5,5)

        self.flatten = nn.Flatten(start_dim=1) 

        self.fc1 = nn.Linear(16*5*5 + 5, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 25)

        self.relu = F.relu
        


    
    def forward(self, x, aux):

        #print(x.size(), aux.size())
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        x = self.flatten(x)
        #print(x.size(), aux.size())
        x = torch.cat((x,aux), dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        #NO SOFTMAX

        return x

    def save(self):
        torch.save(self.state_dict(), "car_racing_model.pth")

    def get_weights(self):
        return self.state_dict()
    
    def set_weights(self, weights):
        self.load_state_dict(weights)

def is_off_road(state_image):
    center_x = 40
    center_y = 60
    region = state_image[0, center_y-2:center_y+3, center_x-2:center_x+3]
    avg_value = torch.mean(region)
    #print(avg_value)
    return (avg_value > 110).item()

def pick_action(policy, state, aux):
    state_tensor = prepare_state_into_tensor(state).unsqueeze(0)
    
    #print(state_tensor.size())
    q_values = policy(state_tensor,aux.unsqueeze(0).to(torch.float32))
    arg_max = torch.argmax(q_values).item()
    #print(arg_max)
    return arg_max




policy = Policy()
policy.set_weights(torch.load(f"carracing_4000.pth", map_location='cuda'))

seed = 12345
env = gym.make('CarRacing-v3', render_mode='human',lap_complete_percent=0.95, domain_randomize=False, continuous=True)





def car_speed():
    return  np.linalg.norm(env.unwrapped.car.hull.linearVelocity)

def car_heading():
    return ((env.unwrapped.car.hull.angle + np.pi) / (2*np.pi) ) % 1

def car_steering():
    return env.unwrapped.car.wheels[0].joint.angle + 0.5

def car_angle_vel():
    return env.unwrapped.car.hull.angularVelocity



state, _ = env.reset(seed=seed)


for i in range(983):
    # Render the game window
    
    env.render()

    state_speed = car_speed()
    state_angle_vel = car_angle_vel()
    state_heading = car_heading()
    state_steering = car_steering()
    state_off_road = is_off_road(prepare_state_into_tensor(state))
    auxilary = torch.tensor([state_speed, state_angle_vel, state_heading, state_steering, int(state_off_road)])
    action = pick_action(policy, state, auxilary)


    #if i < 40:
    #   action = 10
    state, reward, done, truncated, info = env.step(make_action(action))
    print(i)
    if done:
        break


