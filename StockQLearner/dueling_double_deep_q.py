import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from StockQLearner.Environment import StockEnv
from dqn_agent import Agent

env = StockEnv()
env.seed(0)
print('State shape: ', env.n_o)
print('Number of actions: ', env.n_a)

agent = Agent(state_size=env.n_o, action_size=env.n_a, seed=0)


def train(n_episodes=3300, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    # check_dict = torch.load('checkpoint_Dueling_DDQN.pth')
    # agent.qnetwork_local.load_state_dict(check_dict)
    # #
    # check_dict = torch.load('checkpoint_Deul_DDQ.pth')
    # agent.qnetwork_target.load_state_dict(check_dict)

    for i_episode in range(1, n_episodes + 1):
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_Dueling_DDQN.pth')
    torch.save(agent.qnetwork_target.state_dict(), 'checkpoint_Deul_DDQ.pth')
    return scores


scores = train()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

