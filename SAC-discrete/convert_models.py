import os
import torch as T

def convert_actor(state_dict):
    new_dict = {}
    for var_name in state_dict:
        print(var_name, "\t", state_dict[var_name].size())
        if var_name == 'hidden_layers.0.weight':
            new_dict['base.0.weight'] = state_dict[var_name]
        elif var_name == 'hidden_layers.0.bias':
            new_dict['base.0.bias'] = state_dict[var_name]
        elif var_name == 'hidden_layers.1.weight':
            new_dict['base.2.weight'] = state_dict[var_name]
        elif var_name == 'hidden_layers.1.bias':
            new_dict['base.2.bias'] = state_dict[var_name]
        elif var_name == 'output_layers.0.weight':
            new_dict['base.4.weight'] = state_dict[var_name]
        elif var_name == 'output_layers.0.bias':
            new_dict['base.4.bias'] = state_dict[var_name]
    return new_dict

def convert_critic(state_dict):
    new_dict = {}
    for var_name in state_dict:
        print(var_name, "\t", state_dict[var_name].size())
        if var_name == 'hidden_layers.0.weight':
            new_dict['critic.0.weight'] = state_dict[var_name]
        elif var_name == 'hidden_layers.0.bias':
            new_dict['critic.0.bias'] = state_dict[var_name]
        elif var_name == 'hidden_layers.1.weight':
            new_dict['critic.2.weight'] = state_dict[var_name]
        elif var_name == 'hidden_layers.1.bias':
            new_dict['critic.2.bias'] = state_dict[var_name]
        elif var_name == 'output_layers.0.weight':
            new_dict['critic.4.weight'] = state_dict[var_name]
        elif var_name == 'output_layers.0.bias':
            new_dict['critic.4.bias'] = state_dict[var_name]
    return new_dict

def print_dict(state_dict_1):
    for var_name in state_dict_1:
        print(var_name, "\t", state_dict_1[var_name].size())

state_dict_1 = T.load('tmp/adv/actor_sac')
state_dict_2 = T.load('tmp/adv/critic_1_sac')
state_dict_3 = T.load('tmp/adv/critic_2_sac')
state_dict_4 = T.load('tmp/adv/target_critic_1_sac')
state_dict_5 = T.load('tmp/adv/target_critic_2_sac')

state_dict_1 = convert_actor(state_dict_1)
state_dict_2 = convert_critic(state_dict_2)
state_dict_3 = convert_critic(state_dict_3)
state_dict_4 = convert_critic(state_dict_4)
state_dict_5 = convert_critic(state_dict_5)
 
T.save(state_dict_1, 'tmp/adv/actor_sac')
T.save(state_dict_2, 'tmp/adv/critic_1_sac')
T.save(state_dict_3, 'tmp/adv/critic_2_sac')
T.save(state_dict_4, 'tmp/adv/target_critic_1_sac')
T.save(state_dict_5, 'tmp/adv/target_critic_2_sac')
