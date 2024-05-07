import torch
import gymnasium as gym
import einops as ein
import numpy as np
from tqdm import tqdm
from typing import List
from .buffer import Buffer, Transition





class ReplayTrainer:
    def __init__(self, env: gym.Env, agent, target_update_interval=16, batch_size=128, buffer_cap=8192):
        self.env = env
        self.agent = agent
        
        self.target_update_interval = target_update_interval

        self.batch_size = batch_size
        self.replay_buffer = Buffer(buffer_cap)

    def _sample_transitions(self):
        done = False
        count = 0
        observation, info = self.env.reset()

        while not done:
            
            cur_observation = observation

            action = self.agent.take_action(observation).item()
            observation, reward, terminated, truncated, info = self.env.step(action)
            
            ################# current ######################### next 
            t = Transition(cur_observation, action, reward, observation, terminated or truncated)
            self.replay_buffer.push(t)

            done = terminated or truncated
            count += 1
        # end while
        # print(f'replay buffer size = {len(self.replay_buffer)}')

    
    def train(self, num_updates): # batch_size = number of transitions
        
        for iupdate in tqdm(range(num_updates)):
            self._sample_transitions() # at least sample once
            while len(self.replay_buffer) < self.batch_size * 8:
                self._sample_transitions()

            bt = self.replay_buffer.draw(self.batch_size)
            bt.to(self.agent.device)

            self.agent.update_transitions(bt)

            if iupdate % (self.target_update_interval - 1) == 0:
                self.agent.sync_target()
        
        # print(len(self.replay_buffer))



        
        
        












