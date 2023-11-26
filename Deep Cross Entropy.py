import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import gym


def get_trajectory(env, agent, max_iter=100, plot_game=True):
    res = {'states': [],
           'rewards': [],
           'actions': []}
    state = env.reset()
    for i in range(max_iter):
        action = agent.get_action(state)
        state1, reward, is_end, prob = env.step(action)
        res['states'].append(state)
        res['rewards'].append(reward)
        res['actions'].append(action)
        state = state1

        if is_end:
            break

        if plot_game == True:
            env.render()
            time.sleep(0.01)
    return res


n_dim = 4


class RandomAgent():
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def get_action(self, state):
        return np.random.randint(self.n_actions)


class DeepAgent(torch.nn.Module):
    def __init__(self, dim_state, n_act):
        super().__init__()
        self.dim_state = dim_state
        self.n_act = n_act

        self.model = nn.Sequential(nn.Linear(self.dim_state, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, self.n_act)
                                   )
        self.softmax = nn.Softmax()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_):
        return self.model(input_)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.forward(state)
        res = self.softmax(logits).detach().numpy()
        action = np.random.choice(self.n_act, p=res)
        return action

    def fit(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        for traj in elite_trajectories:
            for state, action in zip(traj['states'], traj['actions']):
                elite_states.append(state)
                elite_actions.append(action)
        elite_states = torch.FloatTensor(elite_states)
        elite_actions = torch.LongTensor(elite_actions)
        pred = self.forward(elite_states)
        loss = self.loss(pred, elite_actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


env = gym.make('CartPole-v1')
n_action = 2
n_dim = 4
q_param = 0.8
agent = DeepAgent(n_act=n_action, dim_state=n_dim)
n_iterations = 50
n_trajectories = 20
trajectory_len = 1000

for j in range(n_iterations):
    trajectories = []
    print(f'iteration num : {j}')
    for k in range(n_trajectories):
        trajectories.append(get_trajectory(env, agent, max_iter=trajectory_len, plot_game=False))

    print('traj_count:', len(trajectories))
    total_rewards = [np.sum(i['rewards']) for i in trajectories]
    quant = np.quantile(total_rewards, q_param)
    print('quantile:', quant)
    print(f"mean total reward : {np.mean(total_rewards)}")
    best_trajectories = []
    for i in trajectories:
        if np.sum(i['rewards']) > quant:
            best_trajectories.append(i)

    if len(best_trajectories) > 0:
        agent.fit(best_trajectories)

get_trajectory(env, agent, max_iter=trajectory_len, plot_game=True)

