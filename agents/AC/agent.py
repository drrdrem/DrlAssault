import numpy as np

import torch
import torch.optim as optim
import torch.nn.utils as utils
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
from agents.AC.network import *


class A2C(object):
    """An agent based on A2C Algorithm.
       Modified from: https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
        Arguments:
            action_size (int): Size of Actions, default 20.
            lr (float): learning rate, default 1e-3.
            gamma (float): discount factor, default 0.9.
            seed (int): Random Seed, default 0.
            model_dir: load model directory, default none.
        """
    def __init__(self, action_size=2, observations_size=(3, 256, 256), 
                lr=1e-3, gamma=.9, seed=0, model_dir=None):
        torch.manual_seed(seed=seed)
        self.action_size = action_size
        self.observations_size = observations_size
                
        self.lr = lr
        self.gamma = gamma

        self.model =  PolicyNetwork(observations_size = self.observations_size, action_size = self.action_size, seed=seed)
        if model_dir:
            self.model.load_state_dict(torch.load(model_dir))
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
    
    def action(self, observations, train=True):
        """A step of agent acts 
        """
        if train:
            self.model.train()
        else:
            self.model.eval()

        observations = torch.tensor(observations)
        observations = autograd.Variable(observations).float()
        probs, value = self.model(observations)
        
        return probs, value

    def learn(self, rewards, log_probs, values, entropies):
        """A step of agent learns 
        Arguments:
            rewards (arr): The selected paper from previous action
            log_probs (arr): features text
            entropies (arr): pre_state cx, hx fromm RNN output
        Returns:
            The total loss in the stage.
        """
        R = 0
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in rewards:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)

        for log_prob, value, R in zip(log_probs, values, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss 
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([[R]])))

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss = loss / len(rewards)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()
        
        return loss.data.item()
    

    def save(self, model_dir):
        """ Save the agent. 
        Arguments:
            model_dir (str): the agent saving directory.
        """
        torch.save(self.model.state_dict(), model_dir+'_model.pkl')