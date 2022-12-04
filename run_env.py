import gym
import matplotlib.pyplot as plt
env = gym.make("Breakout-v0")
env.reset()
for _ in range(1000):
    # env.render()
    action = env.action_space.sample() # take a random action
    print(action)
    observation, reward, done, info = env.step(action)
    print(observation.shape)