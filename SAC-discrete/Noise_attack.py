import gym
import numpy as np
import pandas as pd
from SAC import Agent
from tqdm import tqdm
from matplotlib import pyplot as plt

def noise_attack(observation, epsilon):
    noise = np.sign(np.random.normal(0, 1, len(observation)))
    perturbed_observation = observation + epsilon*noise
    return perturbed_observation

perturb = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

if __name__ == '__main__':
    for p in perturb:
        env = gym.make('LunarLander-v2')
        render = False
        use_timesteps = False
        load_checkpoint = True
        chkpt_dir = 'tmp/sacd'
        adv_dir = 'tmp/adv'
        n_trials = 10
        n_games = 100
        alpha = 0.0003
        beta = 0.0003
        gamma = 0.99
        max_size = 1000000
        batch_size = 256
        tau = 0.005
        ent_alpha = 0.2
        reward_scale = 2
        n_timesteps = 2e6
        total_timesteps = 0
        perturbation = p
        best_score = env.reward_range[0]
        reward_history = []    
        
        score_book = {}
        value_loss_book = {}
        actor_loss_book = {}
        critic_1_loss_book = {}
        critic_2_loss_book = {}
        critic_loss_book = {}

        for trial_num in range(n_trials):
            print('\nTrial num:', trial_num+1)
            agent = Agent(input_dims=env.observation_space.shape, layer1_size=256, layer2_size=256,
                        env=env, n_actions=env.action_space.n, alpha=alpha, beta=beta, 
                        gamma=gamma, max_size=max_size, tau=tau, ent_alpha=ent_alpha, batch_size=batch_size,
                        reward_scale = reward_scale, chkpt_dir=chkpt_dir)
            
            score_history = []
            loss = []
            value_loss = []
            actor_loss = []
            critic_1_loss = []
            critic_2_loss = []
            critic_loss = []

            if load_checkpoint:
                agent.load_models()

            for i in tqdm(range(n_games)):
                
                if render:
                    env.render(mode='human')

                observation = env.reset()
                done = False
                score = 0
                
                while not done:
                    if use_timesteps:
                        total_timesteps += 1
                
                    action,_,_,_ = agent.choose_actions(observation)
                    observation_, reward, done, info = env.step(action)

                    observation_ = noise_attack(observation_, perturbation)

                    score += reward
                    reward_history.append(reward)
                    
                    agent.remember(observation, action, reward, observation_, done)
                    
                    if not load_checkpoint:
                        loss.append(agent.learn())

                    observation = observation_
                
                if not load_checkpoint:
                    avg_loss = np.mean(loss, axis=0)
                    value_loss.append(avg_loss[0])
                    actor_loss.append(avg_loss[1])
                    critic_1_loss.append(avg_loss[2])
                    critic_2_loss.append(avg_loss[3])
                    critic_loss.append(avg_loss[4])

                score_history.append(score)
                avg_score = np.mean(score_history[-100:])

                if avg_score > best_score:
                    best_score = avg_score
                    if not load_checkpoint:
                        print("saving model")
                        agent.save_models()

                if done and use_timesteps and total_timesteps>=n_timesteps-1:
                    break
                
                # print('Episode', i, 'score %.1f'% score, 'avg_score %.1f'%avg_score)
            
            score_book[trial_num] = np.array(score_history)
            value_loss_book[trial_num] = np.array(value_loss)
            actor_loss_book[trial_num] = np.array(actor_loss)
            critic_1_loss_book[trial_num] = np.array(critic_1_loss)
            critic_2_loss_book[trial_num] = np.array(critic_2_loss)
            critic_loss_book[trial_num] = np.array(critic_loss)


        print("\nStoring rewards data...")
        a = pd.DataFrame(score_book)
        a.to_csv('data/Noise/SACD-LunarLander100x10-rewards-testFGSM_'+str(int(10*perturbation))+'.csv')
        # if not load_checkpoint:
        #     print("\nStoring losses...")
        #     b = pd.DataFrame(value_loss_book)
        #     b.to_csv('data/SAC-LunarLander1000-value_loss.csv')
        #     c = pd.DataFrame(actor_loss_book)
        #     c.to_csv('data/SAC-LunarLander1000-actor_loss.csv')
        #     d = pd.DataFrame(critic_1_loss_book)
        #     d.to_csv('data/SAC-LunarLander1000-critic_1_loss.csv')
        #     e = pd.DataFrame(critic_2_loss_book)
        #     e.to_csv('data/SAC-LunarLander1000-critic_2_loss.csv')
        #     f = pd.DataFrame(critic_loss_book)
        #     f.to_csv('data/SAC-LunarLander1000-critic_loss.csv')
    print("\nExperiment finished")