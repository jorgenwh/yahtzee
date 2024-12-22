import random
import torch
from collections import deque

import gym

from yahtzee.agents.dqn_agent.replay_buffer import ReplayBuffer
from yahtzee.agents.dqn_agent.dqn_agent import DQNAgent
from yahtzee.agents.dqn_agent.model import Model
from yahtzee.yahtzee import Yahtzee
from yahtzee.agents.dqn_agent.utils import state_to_tensor


TRAINING_STEPS = 15
BATCH_SIZE = 16
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.95


def simple_state_to_tensor(state):
    return torch.tensor([state.number, state.cntr], dtype=torch.float32)


class Trainer():
    def __init__(self):
        # self.player = DQNAgent()
        # self.target = DQNAgent()
        self.player = Model()
        self.target = Model()
        self.replay_buffer = ReplayBuffer()

        self.losses = deque(maxlen=500)
        self.scores = deque(maxlen=500)

    def run(self):
        for i in range(1000):
            self.iterate()

            if i % 10 == 0:
                #self.player.model.load_state_dict(self.target.model.state_dict())
                self.player.load_state_dict(self.target.state_dict())

    def iterate(self):
        self.play_episode()
        self.train_model()

    def train_model(self):
        """
        Q(S_t, A_t) = (1 - LR) * Q(S_t, A_t) + LR * (R_t + DF * max_a Q(S_{t+1}, a))
        """
        #model = self.target.model
        model = self.target
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        for i in range(TRAINING_STEPS):
            (states, actions, rewards, next_states, dones) = self.replay_buffer.sample(min(BATCH_SIZE, len(self.replay_buffer)))

            states = torch.stack([simple_state_to_tensor(state) for state in states])
            actions = torch.tensor(actions)
            next_states = torch.stack([simple_state_to_tensor(state) for state in next_states])
            rewards = torch.tensor(rewards)
            dones = list(dones)

            q_values = model(states)
            next_q_values = model(next_states)

            target_q_values = q_values.clone()

            for i in range(len(dones)):
                if dones[i]:
                    target_q_values[i][actions[i]] = rewards[i]
                else:
                    #target_q_values[i][actions[i]] = rewards[i] + DISCOUNT_FACTOR * torch.max(next_q_values[i])
                    target_q_values[i][actions[i]] = torch.max(next_q_values[i])

            loss = criterion(q_values, target_q_values)
            self.losses.append(loss.item())

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            print(f"Score: {round(sum(self.scores)/max(len(self.scores), 1), 1)} - Loss: {round(sum(self.losses)/max(len(self.losses), 1), 2)}", end="\r", flush=True)

    def play_episode(self):
        #game = Yahtzee()
        game = SimpleGame()

        while True:
            state = game.state.copy()
            #action = self.player.get_action(game.state)

            action_scores = self.player(simple_state_to_tensor(state))
            if random.random() < 0.1:
                action = random.randint(0, 1)
            else:
                action = torch.argmax(action_scores).item()

            game.step(action)
            
            next_state = game.state.copy()
            done = game.is_done()
            reward = game.get_score()

            self.replay_buffer.add(state, action, reward, next_state, done)

            if done:
                self.scores.append(reward)
                print(f"Score: {round(sum(self.scores)/max(len(self.scores), 1), 2)} - Loss: {round(sum(self.losses)/max(len(self.losses), 1), 4)}", end="\r", flush=True)
                break


class SimpleState():
    def __init__(self, number = random.randint(-10, 10), cntr = 0):
        self.number = number
        self.cntr = cntr

    def copy(self):
        return SimpleState(self.number, self.cntr)

class SimpleGame():
    def __init__(self):
        self.state = SimpleState()

    def step(self, action):
        assert action in [0, 1]

        if action == 0:
            self.state.number -= 1
        else:
            self.state.number += 1

        self.state.cntr += 1

    def is_done(self):
        return self.state.cntr >= 100

    def get_score(self):
        return 10 - abs(self.state.number)
