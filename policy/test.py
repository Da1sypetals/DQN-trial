from .dqn import DQN
import torch
import gymnasium as gym
import time


class Test:
    def __init__(self, env: gym.Env, agent):
        self.env = env
        self.agent = agent


    def run(self, interval=False):
        done = False
        observation, info = self.env.reset()

        while not done:

            action = self.agent.take_action(observation).item()
            observation, reward, terminated, truncated, info = self.env.step(action)

            print(f'action: {action}')

            done = terminated or truncated

            if interval:
                time.sleep(interval)







