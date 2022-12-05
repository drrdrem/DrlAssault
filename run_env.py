import gym
from PIL import Image
import numpy as np

import agents.AC.agent as a2c 

import torchvision.transforms as T
transform = T.Resize(size = (256,256))


env = gym.make("Assault-v4")
agent = a2c.A2C(action_size=7)

observation = env.reset()
entropies, log_probs, values, rewards = [], [], [], []
for i in range(1000):
    # env.render()

    observation = Image.fromarray(np.uint8(observation)).convert('RGB')
    observation = transform(observation)
    observation = np.array(observation)
    observation = np.transpose(observation, (2, 0, 1))
    observation = np.expand_dims(observation, axis=0)

    probs, value = agent.action(observation) # take a random action
    probs_samp = probs.clone().detach()
    action = probs_samp.multinomial(num_samples=1).data

    observation_nxt, reward, done, info = env.step(action)


    prob = probs[:, action[0,0]].view(1, -1)
    log_prob = prob.log()
    entropy = - (probs*probs.log()).sum()

    entropies.append(entropy)
    log_probs.append(log_prob)
    values.append(value)
    rewards.append(reward)

    observation = observation_nxt
loss = agent.learn(rewards, log_probs, values, entropies)
