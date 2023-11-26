import gym
import numpy as np
import time




n_actions = 5



class Agent:

    def __init__(self, n_actions, n_states):
        self.n_states = n_states
        self.n_actions = n_actions
        self.model = np.ones((self.n_states, self.n_actions))/ self.n_actions

    def get_action(self, state):
        return np.random.choice(list(range(self.n_actions)), p=self.model[state])

    def fit(self, best_trajectorie, typ = 'default', lambd = None):
# possible types of fitting:
# "default" - default cross entropy methon
# "laplas" - laplase smoothing method, needs lambda as param
# "smoothong" - policy smoothing method needs lambda as param


        new_model = np.zeros((self.n_states, self.n_actions))
        for i in best_trajectories:
            for state, action in zip(i['states'], i['actions']):
                new_model[state,action] += 1
        for i in range(self.n_states):
            if typ in ['default','smoothing']:
                if np.sum(new_model[i]) > 0:
                    new_model[i] = new_model[i]/np.sum(new_model[i])
                else:
                    new_model[i] = self.model[i]
            elif typ == 'laplas':
                new_model[i] = (new_model[i] + lambd) / (np.sum(new_model[i]) + lambd * self.n_actions)

        if typ in ['default', 'laplas']:
            self.model = new_model
        elif typ == 'smoothing':
            self.model = lambd * new_model + (1 - lambd)*self.model
        return None




def get_trajectory(env, agent, max_iter = 10000, plot_game = True):
    res = {'states': [],
           'rewards': [],
           'actions': []}
    state = env.reset()
    for i in range(max_iter):
        action = agent.get_action(state)
        state, reward, is_end, prob = env.step(action)
        res['states'].append(state)
        res['rewards'].append(reward)
        res['actions'].append(action)
        if is_end == True:
            break

        if plot_game == True:
            env.render()
            time.sleep(0.01)
    return res

agent = Agent(6,500)
n_iterations = 3
n_trajectories = 1000
env = gym.make('Taxi-v3')
trajectories = []
loss = -np.inf
for j in range(n_iterations):
    for i in range(n_trajectories):
        trajectories.append(get_trajectory(env, agent, plot_game=False))


    total_rewards = [np.sum(i['rewards']) for i in trajectories]
    quant = np.quantile(total_rewards, 0.99)
    print(quant)
    best_trajectories = []
    for i in trajectories:
        if np.sum(i['rewards']) > quant:
            best_trajectories.append(i)
    agent.fit(best_trajectories, lambd=2, typ='laplas')
get_trajectory(env=env, agent=agent)
print(agent.model)