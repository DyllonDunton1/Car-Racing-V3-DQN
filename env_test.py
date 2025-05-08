import gymnasium as gym
import pygame
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
from input_testing import prepare_state_into_tensor


def is_off_road(state_image):
    center_x = 40
    center_y = 60
    region = state_image[0, 0, center_y-2:center_y+3, center_x-2:center_x+3]
    avg_value = torch.mean(region)
    print(avg_value)
    return (avg_value > 110).item()
    


env = gym.make('CarRacing-v3', render_mode='human',lap_complete_percent=0.95, domain_randomize=False, continuous=True)
state = env.reset()
#print(state)
tensor_state = torch.tensor(state[0], dtype=torch.float32)
#print(tensor_state)
#print(env.action_space)

def car_speed():
    return  np.linalg.norm(env.unwrapped.car.hull.linearVelocity)

def car_heading():
    return ((env.unwrapped.car.hull.angle + np.pi) / (2*np.pi) ) % 1

def car_steering():
    return env.unwrapped.car.wheels[0].joint.angle + 0.5

def car_angle_vel():
    return env.unwrapped.car.hull.angularVelocity

iter = -1
while True:
    # Render the game window
    env.render()
    iter += 1

    if iter < 100:
        action = [-1.0, 0.1, 0]
    else:
        action = [0, 0.1, 0]
    state, reward, done, truncated, info = env.step(np.array(action))
    #state_tensor = prepare_state_into_tensor(state).unsqueeze(0)
    #print(tensor_state)
    if done:
        state, info = env.reset()
        break
    
    print(car_steering())

    
# Close the environment
env.close()

