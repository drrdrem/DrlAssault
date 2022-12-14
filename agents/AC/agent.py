import numpy as np

import torch
import torch.optim as optim
import torch.nn.utils as utils
import torch.autograd as autograd
import torch.nn.functional as F
from network import *

torch.manual_seed(0)

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
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.model =  PolicyNetwork(observations_size = self.observations_size, action_size = self.action_size, seed=seed).to(self.device)
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
        observations = autograd.Variable(observations).float().to(self.device)
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
        eps = np.finfo(np.float32).eps.item()
        R = 0
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in rewards:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + eps)

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


if __name__== '__main__':
    import gym
    from PIL import Image
    import numpy as np
    import time
    import os
    from datetime import datetime
    import pandas as pd


    import torchvision.transforms as T
    transform = T.Resize(size = (256,256))


    env = gym.make("Assault-v4")
    action_size = env.action_space.n
    agent = A2C(action_size=action_size)

    # Collect experience
    img_path = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_A2C")
    os.mkdir(img_path)

    print("Action Space:")
    print(env.action_space)
    print()
    metadata = {"i_epi":[], "Horizon": [], "accu_rewards": [], "time_elipse":[], "loss":[]}
    cnt = -1
    st = time.time()
    for i in range(10000):
        observation = env.reset()
        entropies, log_probs, values, rewards = [], [], [], []
        done = False
        rewards_sum = 0
        horizon = 0
        while (not done) and (horizon<2000):
            cnt += 1
            horizon += 1
            # img = env.render(mode='rgb_array')
            env.env.ale.saveScreenPNG(img_path+'/{}.png'.format(cnt))

            observation = Image.fromarray(np.uint8(observation)).convert('RGB')
            observation = transform(observation)
            observation = np.array(observation)
            observation = np.transpose(observation, (2, 0, 1))
            observation = np.expand_dims(observation, axis=0)

            probs, value = agent.action(observation) # take a random action
            probs_samp = probs.clone().detach()
            action = probs_samp.multinomial(num_samples=1).data

            observation_nxt, reward, done, info = env.step(action)

            rewards_sum += reward

            prob = probs[:, action[0,0]].view(1, -1)
            log_prob = prob.log()
            entropy = - (probs*probs.log()).sum()

            entropies.append(entropy)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            observation = observation_nxt
        cnt += 1
        env.env.ale.saveScreenPNG(img_path+'/{}.png'.format(cnt))
        ct = time.time()

        loss = agent.learn(rewards, log_probs, values, entropies)
        del entropies
        del log_probs
        del values
        del rewards
        print("Epi {}: loss {}, horizon: {}, accu reward: {}, time elipse: {}".format(i, loss, horizon, rewards_sum, (ct-st)/60))
        metadata["i_epi"].append(i)
        metadata["Horizon"].append(horizon)
        metadata["accu_rewards"].append(rewards_sum)
        metadata["time_elipse"].append((ct-st)/60)
        metadata["loss"].append(loss)
        pd.DataFrame(metadata).to_csv(img_path+'/metadata.csv', index=False)