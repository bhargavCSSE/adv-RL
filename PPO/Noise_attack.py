import gym
import numpy as np
import pandas as pd
from tqdm import tqdm
from ppo_torch import Agent
from matplotlib import pyplot as plt

def noise_attack(observation, epsilon):
    noise = np.sign(np.random.normal(0, 1, len(observation)))
    perturbed_observation = observation + epsilon*noise
    return perturbed_observation

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')    
    load_checkpoint = True
    render = False
    n_trials = 10
    n_games = 100
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    best_score = env.reward_range[0]
    perturbation = 0.1

    score_book = {}
    actor_loss_book = {}
    critic_loss_book = {}
    total_loss_book = {}

    for trial in range(n_trials):
        print('\nTrial:', trial+1)
        agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha,
                        n_epochs=n_epochs, input_dims=env.observation_space.shape,
                        fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo')
        
        score_history = []
        avg_score_history = []
        loss = []
        actor_loss = []
        critic_loss = []
        total_loss = []

        learn_iters = 0
        avg_score = 0
        n_steps = 0
        have_grad = False
        data_grad = []

        if load_checkpoint:
            agent.load_models()

        for i in tqdm(range(n_games)):
            observation = env.reset()
            done = False
            score = 0

            while not done:
                if render:
                    env.render(mode='human')
                
                action, prob, val = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)

                observation_ = noise_attack(observation_, perturbation)
                
                n_steps += 1
                score += reward
                agent.remember(observation, action, prob, val, reward, done)
                
                if not load_checkpoint:
                    if n_steps % N == 0:
                        loss.append(agent.learn())
                        learn_iters += 1

                observation = observation_
            
            if not load_checkpoint:
                avg_loss = np.mean(loss, axis=0)
                actor_loss.append(avg_loss[0])
                critic_loss.append(avg_loss[1])
                total_loss.append(avg_loss[2])

            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            avg_score_history.append(avg_score)
        
        score_book[trial] = score_history
        actor_loss_book[trial] = actor_loss
        critic_loss_book[trial] = critic_loss
        total_loss_book[trial] = total_loss

            # print('episode', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters)
    print("\nStoring rewards data...")
    a = pd.DataFrame(score_book)
    a.to_csv('data/Noise/PPO-LunarLander1000-rewards-testNoise_'+str(int(10*perturbation))+'.csv')
    if not load_checkpoint:
        print("\nStoring losses...")
        b = pd.DataFrame(actor_loss_book)
        b.to_csv('data/PPO-LunarLander1000-actor_loss.csv')
        c = pd.DataFrame(critic_loss_book)
        c.to_csv('data/PPO-LunarLander1000-critic_loss.csv')
        d = pd.DataFrame(total_loss_book)
        d.to_csv('data/PPO-LunarLander1000-total_loss.csv')
    print("Experiment finshed")