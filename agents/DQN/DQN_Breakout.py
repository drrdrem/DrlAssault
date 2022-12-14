"""
Agent learns the policy based on Q-learning with Deep Q-Network.
Based on the example here: https://morvanzhou.github.io/tutorials/machine-learning/torch/4-05-DQN/
"""
"""
The way I store state and memory is not efficient
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import torchvision.transforms as T
from PIL import Image
import time
import os
from datetime import datetime
import pandas as pd

torch.manual_seed(0)

# TODO: if time available, Tensorboard: check performance


# CNN Architecture
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed): 
        super(PolicyNetwork, self).__init__()
        torch.manual_seed(seed=seed) 

        # Initialize parameters
        self.state_size = state_size                # shape(256,256,3)
        height, width, channels = self.state_size   # channels = layers 
        self.action_size = action_size  

        # Define features 
        self.features =nn.Sequential(
                                        nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=5, stride=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(3, 3),
                                        nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(3, 3),
                                        nn.ReLU(),
                                        nn.Flatten(),   # resize 2d x 3 layers to 1D line for linear
                                        nn.Linear(16 * 27 * 27, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 64),
                                        nn.ReLU(),
                                    )

        self.out = nn.Linear(64, self.action_size) # 7 possible actions

    # Generate action
    def forward(self, state):
        state = self.features(state)
        state = F.relu(state)  
        action_value = self.out(state)
        return action_value


# Deep Q-Network Agent/Policy (2 net: [1] target_net & [2] eval_net)
class DQN(object):
    def __init__(self, lr, gamma, epsilon, target_replace_iter, memory_capacity,
                state_size, action_size, seed): 
        torch.manual_seed(seed=seed)
               
        # Initialize parameters
        self.state_size = state_size  # shape(256,256,3)
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity
 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        print(self.device)
        # Generate 2 net: [1] target_net & [2] eval_net
        self.eval_net = PolicyNetwork(self.state_size, self.action_size, seed).to(self.device)
        self.target_net = PolicyNetwork(self.state_size, self.action_size, seed).to(self.device)
 
        # Initialize memory(2D), each memory slot has size(1, (state + next state + reward + action))
        self.state_size_1D = state_size[0] * state_size[1] * state_size[2]      # transform to 1D (256*256*3)
        self.memory = np.zeros((memory_capacity, self.state_size_1D * 2 + 2))   # 2D array
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.memory_counter = 0
        self.learn_step_counter = 0                                             # for target_net update
        self.curr_loss = float('inf')


    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)    # arr(3,256,256) to tensor(1,3,256,256)
        self.epsilon = min(1, max(1/(0.05*self.learn_step_counter +1), 0.1))

        # Randomly choose action (epsilon-greedy)
        if np.random.uniform() < self.epsilon:                      # random
            action = np.random.randint(0, self.action_size)
        else:                                                       # greedy
            action_value = self.eval_net.forward(state)             # feed into eval net, get scores for each action (in curr state)
            action = torch.max(action_value, 1)[1].cpu().data.numpy()[0]  # choose the one with the largest score

        return action


    def store_transition(self, state, action, reward, next_state):       
        # Pack the experience
        state_1D = state.flatten()                                          # transform from 3D to 1D array
        next_state_1D = next_state.flatten()                                # transform from 3D to 1D array
        transition = np.hstack((state_1D, [action, reward], next_state_1D)) # stack all to 1D array

        # Replace the old memory with new memory
        index = self.memory_counter % self.memory_capacity      # index from 0 - mem_cap
        self.memory[index, :] = transition                      # store in memory (1 slot of memory)
        self.memory_counter += 1


    def learn(self):
        # Randomly select a batch of memory to learn from
        sample_index = np.random.choice(min(self.memory_counter, self.memory_capacity), self.batch_size, replace=False)  # rand choose from mem_cap, return batch_size.array.shape(1,32)
        b_memory = self.memory[sample_index, :]                                 # shape(batch_size,transition.size)

        b_state = torch.FloatTensor(b_memory[:, 0:self.state_size_1D]).to(self.device)    # transform back to 3D tensor
        b_action = torch.LongTensor(b_memory[:, self.state_size_1D:self.state_size_1D+1].astype(int)).to(self.device) 
        b_reward = torch.FloatTensor(b_memory[:, self.state_size_1D+1:self.state_size_1D+2]).to(self.device) 
        b_next_state = torch.FloatTensor(b_memory[:, -self.state_size_1D:]).to(self.device) 

        # Unflatten b_state & b_next_state: from (n, 3*256*256) -> (n, 3, 256, 256), for feed into net
        # TODO: not sure if this transformation is correct
        b_state = b_state.unflatten(1, (3,256,256)) 
        b_next_state = b_next_state.unflatten(1, (3,256,256)) 

        # Compute loss between Q values of eval_net & target_net
        for epoch in range(5):
            q_eval = self.eval_net(b_state).gather(1, b_action)                             # evaluate the Q values of the experiences, given the state & actions taken at that time
            q_next = self.target_net(b_next_state).detach()                                 # detach from graph, don't backpropagate (for now, don't train curr target_net)
            q_target = b_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)    # compute the target Q values from these experience
            loss = self.loss_func(q_eval, q_target)

            # Backpropagate
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print(loss.item())
            self.curr_loss = loss.item()

            # Update target network every few iterations (target_replace_iter), i.e. replace target_net with eval_net
            self.learn_step_counter += 1
            if self.learn_step_counter % self.target_replace_iter == 0:
                self.target_net.load_state_dict(self.eval_net.state_dict())
        

# Run 
if __name__ == '__main__':
    transform = T.Resize(size = (256,256))      # transfrom function # T: torchvision.transforms module

    # Hyper parameters
    batch_size = 128
    lr = 1e-3                                   # learning rate
    epsilon = 0.1                               # epsilon-greedy, factor to explore randomly
    gamma = 0.9                                 # reward discount factor
    target_replace_iter = 100                   # target network update frequency, i.e., every n iter copy eval_net to target_net
    memory_capacity = 2000    
    n_episodes = 10000
    seed = 0

    env = gym.make('Assault-v4')
    # Environment parameters
    action_size = env.action_space.n            # 7
    state_size = env.observation_space.shape    # observation = state # shape(210,160,3)
    state_size = np.array([256,256,3])          # resize to (256,256,3)
    # Create DQN
    dqn = DQN(lr, gamma, epsilon, target_replace_iter, memory_capacity, state_size, action_size, seed)

    # Collect experience
    img_path = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_DQN")
    os.mkdir(img_path)

    st = time.time()
    cnt = -1
    metadata = {"i_epi":[], "Horizon": [], "accu_rewards": [], "time_elipse":[], "loss":[]}
    for i_episode in range(n_episodes):
        t = 0                       # timestep
        accu_rewards = 0            # accumulate rewards for each episode
        state = env.reset()   # reset environment to initial state for each episode # state.shape(210,160,3)

        done = False
        while not done:
            cnt += 1
            # env.render()
            env.env.ale.saveScreenPNG(img_path+'/{}.png'.format(cnt))

            # Initialize state(=observation) : from img to array
            state = Image.fromarray(np.uint8(state)).convert('RGB') # convert to Pillow image # shape(210,160)
            state = transform(state)                                # img.size(256x256)
            state = np.array(state)                                 # array.shape(256,256,3)
            state = np.transpose(state, (2,0,1))                    # array.shape(3,256,256)

            # Agent takes action
            action = dqn.choose_action(state)                               # choose an action based on DQN # feed in state.shape(1,3,256,256)
            next_state, reward, done, info = env.step(action)    # do the action, get the reward

            # Transform next_state to array.shape(3,256,256)
            next_state = Image.fromarray(np.uint8(next_state)).convert('RGB')   # convert to Pillow image # shape(210,160)
            next_state = transform(next_state)                                  # img.size(256x256)
            next_state = transform(next_state)                                  # img.size(256x256)
            next_state = np.array(next_state)                                   # array.shape(256,256,3)

            # Keep the experience in memory
            dqn.store_transition(state, action, reward, next_state)

            # Accumulate reward
            accu_rewards += reward

            # If enough memory stored, agent learns from them via Q-learning (Monte-Carlo)
            # Transition to next state
            state = next_state

            # print("Timesteps: {}, Accumulated rewards: {}" .format(t, accu_rewards))
            t += 1
        cnt += 1
        env.env.ale.saveScreenPNG(img_path+'/{}.png'.format(cnt))

        ct = time.time()
        print("Episode {} finished after {} timesteps, total rewards {}, time elipse: {}".format(i_episode, t, accu_rewards, (ct-st)/60))
        if dqn.memory_counter > batch_size:
            print("Start Training.....")
            dqn.learn()
            print("Finishing Training with loss {}".format(dqn.curr_loss))
            metadata["i_epi"].append(i_episode)
            metadata["Horizon"].append(t)
            metadata["accu_rewards"].append(accu_rewards)
            metadata["time_elipse"].append((ct-st)/60)
            metadata["loss"].append(dqn.curr_loss)
            pd.DataFrame(metadata).to_csv(img_path+'/metadata.csv', index=False)


    env.close()