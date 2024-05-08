import torch
import gymnasium as gym
from policy.net import make_nets
from kan.kan_nets import make_kan_nets
from policy.dqn import DQN
from policy.train import ReplayTrainer
from policy.test import Test


device = torch.device('cuda')


# env = gym.make("LunarLander-v2")
env = gym.make("CartPole-v1")
# test_env = gym.make("LunarLander-v2", render_mode='human')
test_env = gym.make("CartPole-v1", render_mode='human')

# q_net, target_net = make_nets(4, 2, device)
q_net, target_net = make_kan_nets(4, 2, device)


q_optim = torch.optim.AdamW(q_net.parameters(), lr=0.002)

agent = DQN(q_net, target_net, q_optim, device=device, epsilon=.02)

trainer = ReplayTrainer(env, agent, target_update_interval=16, batch_size=128)
test = Test(test_env, agent)

for turn in range(100):
    print(f'> turn {turn}')
    trainer.train(num_updates=128)

    test.run()




