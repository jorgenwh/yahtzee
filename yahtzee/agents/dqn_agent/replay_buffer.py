from collections import deque
import random


BUFFER_SIZE = 5000


class ReplayBuffer():
    def __init__(self):
        self.buffer = deque(maxlen=BUFFER_SIZE)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
