import numpy as np

import torch
import torch.optim as optim
import torch.nn.utils as utils
from torch.autograd import Variable
import torch.autograd as autograd

from agents.REINFORCE.network import *

class REINFORCE(object):
    """An agent based on REINFORCE Algorithm.
       Modified from: https://github.com/chingyaoc/pytorch-REINFORCE/blob/master/reinforce_discrete.py
    """
    def __init__(self, action_size=2, observations_size=(3, 256, 256), 
                lr=1e-3, gamma=.9, alpha=0.0001, l=0.5, seed=0, model_dir=None):
        torch.manual_seed(seed=seed)
        self.action_size = action_size
        self.observations_size = observations_size

        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha
        self.l =l

        self.model =  PolicyNetwork(observations_size = self.observations_size, action_size = self.action_size, seed=seed)
        if model_dir:
            self.model.load_state_dict(torch.load(model_dir))
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
    
    def action(self, observations, train=True):
        """A step of agent acts 
        Arguments:
            prev_action (arr): The selected paper from previous action
            text (str): features text
            pre_states (tuple): pre_state cx, hx fromm RNN output

        Returns:
            Next paper id.
        """
        if train:
            self.model.train()
        else:
            self.model.eval()
        
        observations = torch.tensor(observations)
        observations = autograd.Variable(observations).float()
        probs = self.model(observations)
        
        return probs

    def learn(self, rewards, log_probs, entropies):
        """A step of agent learns 
        Arguments:
            rewards (arr): The selected paper from previous action
            log_probs (arr): features text
            entropies (arr): pre_state cx, hx fromm RNN output
        Returns:
            The total loss in the stage.
        """
        R = torch.zeros(1, 1)
        loss = 0.
        if self.l:
            baselines = [0]
            cum_r = 0
            for idx, r in enumerate(rewards):
                cum_r += r
                baselines.append((1-self.l)*baselines[idx]+self.l*cum_r)
            baselines.pop(0)

        for i in reversed(range(len(rewards))):
            R = self.gamma * R + rewards[i]
            if not self.l:
                loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i]))).sum() - (self.alpha*entropies[i]).sum()
            else:
                R_n = R - baselines[i]
                loss = loss - (log_probs[i]*(Variable(R_n).expand_as(log_probs[i]))).sum() - (self.alpha*entropies[i]).sum()

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