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
import time


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

def car_speed(env):
    return  np.linalg.norm(env.unwrapped.car.hull.linearVelocity)

def car_heading(env):
    return ((env.unwrapped.car.hull.angle + np.pi) / (2*np.pi) ) % 1

def car_steering(env):
    return env.unwrapped.car.wheels[0].joint.angle + 0.5

def car_angle_vel(env):
    return env.unwrapped.car.hull.angularVelocity

def pick_action(policy, state, aux):
    state_tensor = prepare_state_into_tensor(state).unsqueeze(0)
    aux = aux.unsqueeze(0).to(torch.float32)
    #print(state_tensor.size())
    q_values = policy(state_tensor, aux)
    arg_max = torch.argmax(q_values).item()
    #print(arg_max)
    return arg_max

class Rollout_Policy():
    def __init__(self, base_policy, seed, rollout_render_mode='state_pixels', render_mode='human'):
        super(Rollout_Policy, self).__init__()

        self.seed = seed
        self.base_policy = base_policy
        self.action_list = []
        self.action_size = 25
        self.truncation_limit = 20
        self.frames_to_run = 0

        self.main_env = gym.make('CarRacing-v3', render_mode=None,lap_complete_percent=0.95, domain_randomize=False, continuous=True)
        self.current_state, _ = self.main_env.reset(seed=self.seed)

        

    def generate_new_env_and_catch_up(self):
        rollout_env = gym.make('CarRacing-v3', render_mode=None,lap_complete_percent=0.95, domain_randomize=False, continuous=True)
        state, _ = rollout_env.reset(seed=self.seed)

        for action in self.action_list:
            state, reward, done, truncated, info = rollout_env.step(make_action(action))


        return rollout_env, state

    def find_rollout_best_action(self):

        reward_sums = []

        #run base policy for each action and store reward sums
        for action in range(self.action_size):
            print(f"ACTION: {action}")
            off_road_count = 0
            test_env, state = self.generate_new_env_and_catch_up()

            state, reward, done, truncated, info = test_env.step(make_action(action))
            state_speed = car_speed(test_env)
            state_angle_vel = car_angle_vel(test_env)
            state_heading = car_heading(test_env)
            state_steering = car_steering(test_env)
            state_off_road = is_off_road(prepare_state_into_tensor(state))
            auxilary = torch.tensor([state_speed, state_angle_vel, state_heading, state_steering, int(state_off_road)])
            

            if state_off_road:
                off_road_count += 1
                reward -= 5
                if off_road_count > 20:
                    reward = -100
            else:
                off_road_count = 0
                reward += 6*min(1,state_speed/70)


            #run a base policy to 3500 max
            reward_sum = reward
            for frame in range(self.truncation_limit):
                action_picked = pick_action(self.base_policy, state, auxilary)

                state, reward, done, truncated, info = test_env.step(make_action(action_picked))
                state_speed = car_speed(test_env)
                state_angle_vel = car_angle_vel(test_env)
                state_heading = car_heading(test_env)
                state_steering = car_steering(test_env)
                state_off_road = is_off_road(prepare_state_into_tensor(state))
                auxilary = torch.tensor([state_speed, state_angle_vel, state_heading, state_steering, int(state_off_road)])

                if state_off_road:
                    off_road_count += 1
                    reward -= 5
                    if off_road_count > 20:
                        reward = -100
                else:
                    off_road_count = 0
                    reward += 6*min(1,state_speed/70)

                reward_sum += reward

                if done or truncated:
                    break
            
            reward_sums.append(reward_sum)
        
        #Best action is the one with best reward sum
        print(np.array(reward_sums))
        best_action = np.argmax(np.array(reward_sums))
        return best_action

    def rollout_run(self):

        for i in range(self.frames_to_run):
            print(f"ROLLOUT: {i}")
            action = self.find_rollout_best_action()
            self.action_list.append(action)
            self.current_state, reward, done, truncated, info = self.main_env.step(make_action(action))
            np.savetxt('rollout.txt', np.array(self.action_list))

            if done:
                break

    def render_actions(self):
        
        render_env = gym.make('CarRacing-v3', render_mode="human",lap_complete_percent=0.95, domain_randomize=False, continuous=True)
        current_state, _ = render_env.reset(seed=self.seed)
        for i, action in enumerate(self.action_list):
            render_env.render()
            current_state, reward, done, truncated, info = render_env.step(make_action(action))
            print(i)
            time.sleep(0.05)

            if done or truncated:
                break



policy = Policy()
policy.set_weights(torch.load(f"carracing_4000.pth", map_location='cuda'))

seed = 12345

rollout_policy = Rollout_Policy(policy, seed)
actions_to_start =  np.loadtxt('rollout.txt')
print(f"Adding {len(actions_to_start)} actions from previous rollout!")
for action in actions_to_start:
    rollout_policy.action_list.append(int(action))
rollout_policy.rollout_run()
rollout_policy.render_actions()


