import random
from collections import namedtuple, deque

MemoryElement = namedtuple('MemoryElement', ['current_state', 'reward', 'action', 'next_state'])

class Memory:
    def __init__(self, maxlen=1000):
        self.__buffer = deque(maxlen=maxlen)

    def push(self, current_state, reward, action, next_state):
        memory_element = MemoryElement(current_state, reward, action, next_state)
        self.__buffer.append(memory_element)

    def sample(self, batch_size):
        sample = random.sample(self.__buffer, batch_size)
        
        return MemoryElement(*zip(*sample))
        


