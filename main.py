from agent import Agent
from environment import Environment
from collections import deque
import torch
import numpy as np

env = Environment()
agent = Agent(action_size=env.number_actions)

number_episodes = 2000
maximum_timesteps_per_episode = 10000
epsilon_starting_value = 1.0
episilon_ending_value = 0.01
episilon_decay_value = 0.995
epsilon = epsilon_starting_value
scores_on_100_episodes = deque(maxlen=100)

for episodes in range(1, number_episodes + 1):
    state, _ = env.reset()
    score = 0
    for timesteps in range(0, maximum_timesteps_per_episode):
        action = agent.action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done) #type: ignore
        state = next_state
        score += reward #type: ignore
        if done:
            break
    scores_on_100_episodes.append(score)
    epsilon = max(episilon_ending_value, episilon_decay_value * epsilon)
    print(f'\rEpisode: {episodes}\tAverage Score: {np.mean(scores_on_100_episodes):.2f}', end='')
    if episodes % 100 == 0:
        print(f'\rEpisode: {episodes}\tAverage Score: {np.mean(scores_on_100_episodes):.2f}')
    if np.mean(scores_on_100_episodes) >= 500.0: #type: ignore
        print(f'\nEnvironment Solved in {episodes:d} episodes!\tAverage Score: {np.mean(scores_on_100_episodes):.2f}')
        torch.save(agent.local_qnet.state_dict(), 'checkpoint.pth')
        break