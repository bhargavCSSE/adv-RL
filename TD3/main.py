import gym
import numpy as np
from td3_torch import Agent
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=0.001, beta=0.001, input_dims=env.observation_space.shape, 
            tau=0.005, env=env, batch_size=100, layer1_size=400, layer2_size=300, 
            n_actions=env.action_space.shape[0])
    n_games = 1000

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
    
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            
            if not load_checkpoint:
                agent.learn()
            
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        
        print('episode', i, 'score %.1f'%score, 'average score %.1f'%avg_score)
    
    x = [i+1 for i in range(n_games)]
    plt.plot(x, score_history)
    plt.show()