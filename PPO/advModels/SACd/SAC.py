import os
import torch as T
import torch.nn.functional as F
import numpy as np
from advModels.SACd.buffer import ReplayBuffer
from advModels.SACd.networks import ActorNetwork, CriticNetwork
from torch.distributions.categorical import Categorical

class Agent():
    def __init__(self, alpha=0.0003, beta= 0.0003, input_dims=[8], env=None, 
                gamma=0.99, n_actions=2, max_size=1000000, tau=0.005, 
                ent_alpha = 0.0001, batch_size=256, reward_scale=2, 
                layer1_size=256, layer2_size=256, chkpt_dir='tmp/sac'):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.ent_alpha = ent_alpha
        self.reward_scale = reward_scale

        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions, 
                                fc1_dims=layer1_size, fc2_dims=layer2_size ,
                                name='actor', chkpt_dir=chkpt_dir)

        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions, 
                                fc1_dims=layer1_size, fc2_dims=layer2_size ,name='critic_1',
                                chkpt_dir=chkpt_dir)
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions, 
                                fc1_dims=layer1_size, fc2_dims=layer2_size ,name='critic_2',
                                chkpt_dir=chkpt_dir)
        self.target_critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions, 
                                fc1_dims=layer1_size, fc2_dims=layer2_size ,name='target_critic_1',
                                chkpt_dir=chkpt_dir)
        self.target_critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions, 
                                fc1_dims=layer1_size, fc2_dims=layer2_size ,name='target_critic_2',
                                chkpt_dir=chkpt_dir)


        self.update_network_parameters(tau=1)

    def choose_actions(self, observation, learn_mode=False):
        if not learn_mode:
            state = T.Tensor([observation]).to(self.actor.device)
        else:
            state = observation

        action_probs = self.actor.forward(state)
        max_probability_action = T.argmax(action_probs, dim=-1)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample().cpu().detach().numpy()[0]

        z = action_probs == 0.0
        z = z.float()*1e-8
        log_probs = T.log(action_probs + z)

        return action, action_probs, log_probs, max_probability_action

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()

        target_critic_1_state_dict = dict(target_critic_1_params)
        critic_1_state_dict = dict(critic_1_params)

        target_critic_2_state_dict = dict(target_critic_2_params)
        critic_2_state_dict = dict(critic_2_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau*critic_1_state_dict[name].clone() + (1-tau)*target_critic_1_state_dict[name].clone()
        
        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau*critic_2_state_dict[name].clone() + (1-tau)*target_critic_2_state_dict[name].clone()
        
        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
    
    def save_models(self):
        # print('.....saving models.....')
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()
    
    def load_models(self):
        print('.....loading models.....')
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return 10, 10, 10, 10, 10
        
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done, dtype=T.float).to(self.actor.device)
        state_ = T.tensor(next_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)
        
        # Critics Learning

        with T.no_grad():
            action_, probs, log_probs, max_action = self.choose_actions(state_, learn_mode=True)
            qf1_target_ = self.target_critic_1(state_)
            qf2_target_ = self.target_critic_2(state_)
            min_qf_target_ = probs*(T.min(qf1_target_, qf2_target_) - self.ent_alpha*log_probs)
            min_qf_target_ = min_qf_target_.sum(dim=1).view(-1)
            next_q_value = reward + (1.0-done)*self.gamma*min_qf_target_
        
        action = action.view(self.batch_size, 1)
        qf1 = self.critic_1(state).gather(1, action.long())
        qf2 = self.critic_2(state).gather(1, action.long())

        self.critic_1.optimizer.zero_grad()
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf1_loss.backward(retain_graph=False)
        self.critic_1.optimizer.step()

        self.critic_2.optimizer.zero_grad()
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf2_loss.backward(retain_graph=False)
        self.critic_2.optimizer.step()

        self.update_network_parameters()

        # Actor loss
        qf1_pi = self.critic_1(state)
        qf2_pi = self.critic_2(state)
        min_qf_pi = T.min(qf1_pi, qf2_pi)
        inside_term = self.ent_alpha*log_probs - min_qf_pi
        actor_loss = (probs*inside_term).sum(dim=1).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=False)
        self.actor.optimizer.step()

        
    def compute_grads(self):
        if self.memory.mem_cntr < self.batch_size:
            return False
        
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done, dtype=T.float).to(self.actor.device)
        state_ = T.tensor(next_state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        state.requires_grad = True

        with T.no_grad():
            action_, probs, log_probs, max_action = self.choose_actions(state_, learn_mode=True)
            qf1_target_ = self.target_critic_1(state_)
            qf2_target_ = self.target_critic_2(state_)
            min_qf_target_ = probs*(T.min(qf1_target_, qf2_target_) - self.ent_alpha*log_probs)
            min_qf_target_ = min_qf_target_.sum(dim=1).view(-1)
            next_q_value = reward + (1.0-done)*self.gamma*min_qf_target_

        self.actor.optimizer.zero_grad()
        qf1_pi = self.critic_1(state)
        qf2_pi = self.critic_2(state)
        min_qf_pi = T.min(qf1_pi, qf2_pi)
        inside_term = self.ent_alpha*log_probs - min_qf_pi
        actor_loss = (probs*inside_term).sum(dim=1).mean()

        actor_loss.backward()
        data_grad = state.grad.data

        return data_grad.mean(axis=0)
