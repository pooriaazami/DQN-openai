import numpy as np
from PIL import Image   
from matplotlib import cm

import gymnasium as gym


class Environment:

    IMG_SIZE = 84

    def __init__(self):
        self.__env = gym.make("Breakout-v4", render_mode="human")

    @property
    def num_actions(self):
        return 4

    def __preprocess(self, observation):
        img = np.dot(observation[...,:3], [0.2989, 0.5870, 0.1140])

        img = Image.fromarray(np.uint8(cm.gist_earth(img)*255))
        img = img.resize((110, 84))

        width, height = img.size
        new_width, new_height = Environment.IMG_SIZE, Environment.IMG_SIZE

        left = (width - new_width) / 2
        top = (height - new_height) / 2
    
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2

        img = img.crop((left, top, right, bottom))

        return np.mean(np.array(img), axis=-1)

    def step(self, action):
        observations, rewards, ended = [], 0, False

        for _ in range(4):
            if ended:
                observation = np.zeros((210, 160, 3))
                reward = 0
                terminated, truncated = True, True
            else:
                observation, reward, terminated, truncated, _ = self.__env.step(action)

            observation = self.__preprocess(observation)
            observations.append(observation)

            rewards += reward

            if ended or terminated or truncated:
                ended = True

        return np.array(observations), rewards, ended
    

    def reset(self):
        self.__env.reset()
        return np.zeros((4, Environment.IMG_SIZE, Environment.IMG_SIZE))

        
        