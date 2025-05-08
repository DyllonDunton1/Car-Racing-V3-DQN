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





class DQNLearner():
    def __init__(self, env, state_shape=(1, 80, 80), out_shape=(1,25), replay_buffer_size=300, start_episode=0, sampling_batch_size=100, update_timer_thresh=1000, num_episodes=100, epsilon=1.0, gamma=0.99, lr=0.0001, save_name='carracing', pick_up=False, episode_stop=3500, epochs_per_fit=1):
        self.env = env
        self.state_shape=state_shape
        self.out_shape=out_shape
        self.replay_buffer_size=replay_buffer_size
        self.sampling_batch_size=sampling_batch_size
        self.update_timer_thresh=update_timer_thresh
        self.num_episodes=num_episodes
        self.epsilon=epsilon
        self.epsilon_moving = epsilon
        self.gamma=gamma
        self.lr=lr
        self.save_name=save_name
        self.episode_stop = episode_stop
        self.epochs = epochs_per_fit
        self.starting_episode = start_episode


        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        self.update_timer = 0
        self.reward_sums_per_episode = []
        self.actions_list = []
        self.epsilon_per_episode = []
        
        self.online_net = Policy()
        self.target_net = Policy()

        self.optimizer=optim.Adam(self.online_net.parameters(), lr = self.lr)

        if pick_up:
            self.online_net.set_weights(torch.load(f"carracing_3700.pth", map_location='cuda'))

        self.target_net.set_weights(self.online_net.get_weights())

    def is_off_road(self,state_image):
        center_x = 40
        center_y = 60
        region = state_image[0, center_y-2:center_y+3, center_x-2:center_x+3]
        avg_value = torch.mean(region)
        #print(avg_value)
        return (avg_value > 110).item()
    
    def car_speed(self):
        return  np.linalg.norm(self.env.unwrapped.car.hull.linearVelocity)

    def car_heading(self):
        return ((self.env.unwrapped.car.hull.angle + np.pi) / (2*np.pi) ) % 1

    def car_steering(self):
        return self.env.unwrapped.car.wheels[0].joint.angle + 0.5

    def car_angle_vel(self):
        return self.env.unwrapped.car.hull.angularVelocity

    def loss_function(self,predictions, truth):
        loss = 0

        for i, action in enumerate(self.actions_list):
            error = truth[i][action] - predictions[i][action]
            loss += error*error
        
        loss /= self.sampling_batch_size
        return loss

    def fit(self, state_batch, state_aux_batch, output_batch):
        for epoch in range(self.epochs):
            self.online_net.train()
            train_loss = 0

            self.optimizer.zero_grad()
            predicted_outputs = self.online_net(state_batch, state_aux_batch)
            train_loss = self.loss_function(predicted_outputs, output_batch)
            train_loss.backward()
            self.optimizer.step()
            
            #print(f"Epoch training loss: {train_loss.item()}")


    def episodic_train(self):
        
        for episode in range(self.starting_episode, self.starting_episode+self.num_episodes):
            if len(self.replay_buffer) < self.replay_buffer_size:
                print(f"Replay buffer current size: {len(self.replay_buffer)}")
            rewards_list = []
            print(f"Starting episode {episode+1}")

            #if episode > 200 and len(self.replay_buffer)  >= self.replay_buffer_size:
            self.epsilon_moving = max(self.epsilon * (0.995**(episode - 200)), 0.1)
            
            print(f"EPSILON: {self.epsilon_moving}")
            state,_ = self.env.reset()

            iter = 0
            off_road_count = 0
            is_terminal = False
            state_speed = self.car_speed()
            state_angle_vel = self.car_angle_vel()
            state_heading = self.car_heading()
            state_steering = self.car_steering()
            state_off_road = self.is_off_road(prepare_state_into_tensor(state))
            auxilary = torch.tensor([state_speed, state_angle_vel, state_heading, state_steering, int(state_off_road)])

            while not is_terminal and iter <= self.episode_stop:
                
                #self.env.render()
                action = self.pick_action(state, auxilary, episode)
                next_state, reward, is_terminal, _, _ = self.env.step(make_action(action))
                next_state_tensor = prepare_state_into_tensor(next_state)

                next_off_road = self.is_off_road(next_state_tensor)
                next_state_speed = self.car_speed()
                next_state_angle_vel = self.car_angle_vel()
                next_state_heading = self.car_heading()
                next_state_steering = self.car_steering()

                next_auxilary = torch.tensor([next_state_speed, next_state_angle_vel, next_state_heading, next_state_steering, int(next_off_road)])
                
                # Tailor reward
                
                
                if next_off_road:
                    off_road_count += 1
                    reward -= 5
                    if off_road_count > 20:
                        reward = -100
                        is_terminal = True
                        print("OFF ROAD EXIT")
                else:
                    off_road_count = 0
                    reward += 3*min(1,next_state_speed/70)


                
                


                rewards_list.append(reward)
                
                self.replay_buffer.append((prepare_state_into_tensor(state), auxilary,action,reward,next_state_tensor,next_auxilary,is_terminal))

                #print(f"Replay_buffer length: {len(self.replay_buffer)}")
                if len(self.replay_buffer) >= self.replay_buffer_size:
                    #if iter == 0:
                    #    print(f"\n0/{self.episode_stop}")
                    #elif (iter+1) % 500 == 0:
                    #    print(f"{iter+1}/{self.episode_stop}")
                    #else:
                        #print(".", end="", flush=True)

                    self.train_net()
                    iter += 1

                state = next_state
                auxilary = next_auxilary



            reward_sum = np.sum(rewards_list)
            print(f"Sum or Rewards for Episode: {reward_sum}")
            self.reward_sums_per_episode.append(reward_sum)
            self.epsilon_per_episode.append(self.epsilon_moving)
            torch.save(self.online_net.get_weights(), f'{self.save_name}.pth')

            if episode%100 == 0:
                torch.save(self.online_net.get_weights(), f'{self.save_name}_{episode}.pth')
                with open('info.txt', 'w') as file:
                    file.write(f"{self.reward_sums_per_episode, self.epsilon_per_episode}")

    def pick_action(self, state, aux, episode):

        #if episode == 0:
        #    return np.random.randint(25)
        
        random_num = np.random.random()

        
        if random_num < self.epsilon_moving:
            return np.random.randint(25)
        self.online_net.eval()
        #print(state_tensor.size())
        state_tensor = prepare_state_into_tensor(state).unsqueeze(0)
        q_values = self.online_net(state_tensor,aux.unsqueeze(0).to(torch.float32))
        arg_max = torch.argmax(q_values).item()
        #print(arg_max)
        return arg_max

    def train_net(self):
        #print(self.replay_buffer)
        
        random_indeces = np.random.choice(list(range(self.replay_buffer_size)), self.sampling_batch_size, replace=False)
        #print(random_indeces)
        random_sampling = [self.replay_buffer[i] for i in random_indeces]

        state_batch = torch.zeros((self.sampling_batch_size, 1, 80, 80))
        state_aux_batch = torch.zeros((self.sampling_batch_size, 5))

        next_state_batch = torch.zeros((self.sampling_batch_size, 1, 80, 80))
        next_state_aux_batch = torch.zeros((self.sampling_batch_size, 5))

        for i, (s_img, s_aux, a, r, ns_img, ns_aux, done) in enumerate(random_sampling):
            state_batch[i] = s_img
            state_aux_batch[i] = s_aux
            next_state_batch[i] = ns_img
            next_state_aux_batch[i] = ns_aux

        state_batch = state_batch.to(torch.float32)
        state_aux_batch = state_aux_batch.to(torch.float32)
        next_state_batch = next_state_batch.to(torch.float32)
        next_state_aux_batch = next_state_aux_batch.to(torch.float32)
        #print(state_batch.size(), next_state_batch.size())

        q_values_for_next_state_using_target_net = self.target_net(next_state_batch, next_state_aux_batch)
        q_values_for_state_using_online_net = self.online_net(state_batch, state_aux_batch)

        output_batch = torch.zeros(size=(self.sampling_batch_size,25))
        #print(output_batch.size())

        self.actions_list = []

        for state_index, (state, s_aux, action, reward, next_state, n_aux, is_terminal) in enumerate(random_sampling):

            y = reward
            if not is_terminal:
                #print(q_values_for_next_state_using_target_net[state_index])
                y += self.gamma*(torch.max(q_values_for_next_state_using_target_net[state_index]))
            
            self.actions_list.append(action)

            output_batch[state_index] = q_values_for_state_using_online_net[state_index]
            output_batch[state_index][action] = y

        #print("Fitting Online Net")
        self.fit(state_batch, state_aux_batch, output_batch.to(torch.float32))

        self.update_timer += 1
        if self.update_timer >= self.update_timer_thresh:
            self.update_timer = 0
            #print("Copying weights to target network")
            self.target_net.set_weights(self.online_net.get_weights())
            torch.save(self.online_net.get_weights(), f'{self.save_name}.pth')



episodes_to_train = 2000
sample_size = 128
replay_buffer_size = 10000
episode_stop_limit = 3500
epochs_per_fit = 1
start_episode = 3700
pickup = True


render_mode = 'state_pixels'
#render_mode = 'human'
env = gym.make('CarRacing-v3', render_mode=render_mode,lap_complete_percent=0.95, domain_randomize=False, continuous=True)
dqn_learner = DQNLearner(env,num_episodes = episodes_to_train,pick_up=pickup, sampling_batch_size=sample_size, episode_stop=episode_stop_limit, replay_buffer_size=replay_buffer_size, start_episode=start_episode)
dqn_learner.episodic_train()
print(f"Reward List: {dqn_learner.reward_sums_per_episode}")
print(f"Epsilon List: {dqn_learner.epsilon_per_episode}")

