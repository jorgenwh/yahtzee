import torch
import random
from tqdm import tqdm
from typing import List
from matplotlib import pyplot as plt

from yahtzee.yahtzee import Yahtzee
from yahtzee.constants import ACTION_SPACE
from yahtzee.agents.dqn_agent.model import Model, INPUT_SIZE
from yahtzee.agents.dqn_agent.replay_buffer import ReplayBuffer
from yahtzee.agents.dqn_agent.utils import state_to_tensor, ValueTracker


NUM_EPISODES = 2500
EPSILON = 0.1
BATCH_SIZE = 16
GAMMA = 0.99
LR = 0.001
BUFFER_SIZE = 5000
TARGET_UPDATE_FREQ = 100


class Trainer:
    def __init__(self):
        self.game = Yahtzee()
        self.replay_buffer = ReplayBuffer(capacity=BUFFER_SIZE)
        self.policy_net = Model()
        self.target_net = Model()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR)
        self.update_counter = 0

        self.score_tracker = ValueTracker()
        self.loss_tracker = ValueTracker()

    def run(self) -> None:
        for _ in tqdm(range(NUM_EPISODES)):
            self._play_episode()

        self._create_plots()

    def _select_action(
        self, state: torch.Tensor, valid_actions: List[int]
    ) -> torch.Tensor:
        sample = random.random()

        if sample > EPSILON:  # Sample greedily from the Q-network
            with torch.no_grad():
                action_values = self.policy_net(state)
                for i in range(ACTION_SPACE):
                    if valid_actions[i] == 0:
                        action_values[0][i] = -float("inf")
                return action_values.max(1).indices.view(1, 1)

        else:  # Sample randomly from the action space
            action = random.randint(0, ACTION_SPACE - 1)
            while valid_actions[action] == 0:
                action = random.randint(0, ACTION_SPACE - 1)
            return torch.tensor([action], dtype=torch.long).view(1, 1)

    def _play_episode(self) -> None:
        self.policy_net.eval()

        state = self.game.reset()
        state_tensor = state_to_tensor(state).view(1, INPUT_SIZE)

        while True:
            action = self._select_action(state_tensor, state.valid_actions)
            action = int(action.item())
            next_state = self.game.step(action)

            reward = torch.tensor([next_state.score - state.score], dtype=torch.float32)
            next_state_tensor = state_to_tensor(next_state).view(1, INPUT_SIZE)

            self.replay_buffer.push(
                state_tensor,
                action,
                reward,
                next_state_tensor,
                next_state.valid_actions,
                next_state.is_done,
            )

            state = next_state
            state_tensor = next_state_tensor

            self._update_model()

            if next_state.is_done:
                break

        self.score_tracker.add(state.score)

    def _update_model(self) -> None:
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        self.policy_net.train()

        transitions = self.replay_buffer.sample(BATCH_SIZE)

        states = torch.cat([transition.state for transition in transitions])
        assert states.shape == (BATCH_SIZE, INPUT_SIZE), states.shape

        actions = torch.tensor(
            [transition.action for transition in transitions], dtype=torch.long
        ).view(BATCH_SIZE, 1)
        assert actions.shape == (BATCH_SIZE, 1), actions.shape

        rewards = torch.cat([transition.reward for transition in transitions])
        assert rewards.shape == torch.Size([BATCH_SIZE]), rewards.shape

        # Get non-terminal transitions
        non_terminal_mask = torch.tensor(
            [not transition.done for transition in transitions], dtype=torch.bool
        )
        non_terminal_transitions = [t for t in transitions if not t.done]
        non_terminal_next_states = (
            torch.cat([t.next_state for t in non_terminal_transitions])
            if non_terminal_transitions
            else None
        )
        non_terminal_valid_actions = [t.next_valid_actions for t in non_terminal_transitions]

        # Get Q-values for current states from policy network
        state_action_values = self.policy_net(states).gather(1, actions)

        # Compute target Q-values using target network with invalid action masking
        next_state_values = torch.zeros(BATCH_SIZE)
        if non_terminal_next_states is not None:
            with torch.no_grad():
                target_q_values = self.target_net(non_terminal_next_states)
                # Mask invalid actions
                for i, valid_actions in enumerate(non_terminal_valid_actions):
                    for j in range(ACTION_SPACE):
                        if valid_actions[j] == 0:
                            target_q_values[i][j] = -float("inf")
                next_state_values[non_terminal_mask] = target_q_values.max(1).values

        expected_state_action_values = rewards + next_state_values * GAMMA

        loss = self.criterion(
            state_action_values, expected_state_action_values.view(BATCH_SIZE, 1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        status = self.loss_tracker.add(loss.item())
        if status == -1:
            self._save_model("model.pth")

    def _save_model(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def _load_model(self, path: str) -> None:
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _create_plots(self) -> None:
        print("Creating plots...")

        plt.plot(self.score_tracker.get_history())
        plt.title("score history")
        plt.xlabel("mean over 100 episodes")
        plt.ylabel("score")
        plt.savefig("score_history.png")
        plt.close()
        print("Score history plot saved to 'score_history.png'")

        plt.plot(self.loss_tracker.get_history())
        plt.title("loss history")
        plt.xlabel("mean over 100 updates")
        plt.ylabel("loss")
        plt.savefig("loss_history.png")
        plt.close()
        print("Loss history plot saved to 'loss_history.png'")
