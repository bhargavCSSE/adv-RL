import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Softmax
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions=4, fc1_dims=256, fc2_dims=256, name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.critic = nn.Sequential(
                    nn.Linear(*self.input_dims, self.fc1_dims),
                    nn.ReLU(),
                    nn.Linear(self.fc2_dims, self.fc1_dims),
                    nn.ReLU(),
                    nn.Linear(self.fc2_dims, self.n_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        q = self.critic(state)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

# class ValueNetwork(nn.Module):
#     def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256, name='value', chkpt_dir='tmp/sac'):
#         super(ValueNetwork, self).__init__()
#         self.input_dims = input_dims
#         self.fc1_dims = fc1_dims
#         self.fc2_dims = fc2_dims
#         self.name = name
#         self.checkpoint_dir = chkpt_dir
#         self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

#         self.value = nn.Sequential(
#                     nn.Linear(*self.input_dims, self.fc1_dims),
#                     nn.ReLU(),
#                     nn.Linear(fc1_dims, fc2_dims),
#                     nn.ReLU(),
#                     nn.Linear(self.fc2_dims, 1)
#         )

#         self.optimizer = optim.Adam(self.parameters(), lr=beta)
#         self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
#         self.to(self.device)
    
#     def forward(self, state):
#         v = self.value(state)

#         return v
    
#     def save_checkpoint(self):
#         T.save(self.state_dict(), self.checkpoint_file)
    
#     def load_checkpoint(self):
#         self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims=256, fc2_dims=256, n_actions=4, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.base = nn.Sequential(
            nn.Linear(*self.input_dims, self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, self.n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        probs = self.base(state)
        
        return probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))