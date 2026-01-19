import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from collections import deque
from tqdm import tqdm
from matplotlib import pyplot as plt

from yahtzee.yahtzee import Yahtzee
from yahtzee.agents.ppo_agent.model import ActorCritic
from yahtzee.agents.ppo_agent.rollout_buffer import RolloutBuffer
from yahtzee.agents.ppo_agent.utils import state_to_tensor, ValueTracker, get_device


# Hyperparameters
NUM_EPISODES = 100000
ROLLOUT_STEPS = 2048  # Steps to collect before each update
NUM_EPOCHS = 4  # Number of PPO epochs per update
MINIBATCH_SIZE = 64
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
LR = 3e-4
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
DISPLAY_AVG_WINDOW = 50
CHECKPOINT_INTERVAL = 100


class Trainer:
    def __init__(self, use_cuda: bool = False):
        self.device = get_device(use_cuda)
        print(f"Using device: {self.device}")

        self.game = Yahtzee()
        self.model = ActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.rollout_buffer = RolloutBuffer(gamma=GAMMA, gae_lambda=GAE_LAMBDA)

        self.score_tracker = ValueTracker()
        self.loss_tracker = ValueTracker()

    def run(self) -> None:
        self.recent_scores: deque[float] = deque(maxlen=DISPLAY_AVG_WINDOW)
        self.recent_losses: deque[float] = deque(maxlen=DISPLAY_AVG_WINDOW)
        self.best_avg_score: float = float("-inf")

        episode = 0
        pbar = tqdm(total=NUM_EPISODES, desc="Training")

        state = self.game.reset()

        while episode < NUM_EPISODES:
            # Collect rollouts
            self.model.eval()
            steps_collected = 0

            while steps_collected < ROLLOUT_STEPS:
                state_tensor = state_to_tensor(state, self.device)
                valid_actions_mask = torch.tensor(
                    state.valid_actions, dtype=torch.float32, device=self.device
                )

                # Get action from policy
                with torch.no_grad():
                    action_probs = self.model.get_action_probs(
                        state_tensor.unsqueeze(0), valid_actions_mask.unsqueeze(0)
                    )
                    value = self.model.get_value(state_tensor.unsqueeze(0))

                    dist = Categorical(action_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                action_idx = int(action.item())
                next_state = self.game.step(action_idx)
                reward = next_state.score - state.score

                # Store transition
                self.rollout_buffer.add(
                    state=state_tensor,
                    action=action_idx,
                    reward=reward,
                    done=next_state.is_done,
                    log_prob=log_prob.item(),
                    value=value.item(),
                    valid_actions=state.valid_actions,
                )

                steps_collected += 1
                state = next_state

                if next_state.is_done:
                    self.score_tracker.add(next_state.score)
                    self.recent_scores.append(next_state.score)
                    episode += 1
                    pbar.update(1)

                    avg_score = (
                        sum(self.recent_scores) / len(self.recent_scores)
                        if self.recent_scores
                        else 0
                    )
                    avg_loss = (
                        sum(self.recent_losses) / len(self.recent_losses)
                        if self.recent_losses
                        else 0
                    )
                    pbar.set_postfix(score=f"{avg_score:.1f}", loss=f"{avg_loss:.4f}")

                    # Checkpoint if score improved
                    if (
                        episode % CHECKPOINT_INTERVAL == 0
                        and avg_score > self.best_avg_score
                    ):
                        self.best_avg_score = avg_score
                        self._save_model("ppo_model.pth")
                        tqdm.write(
                            f"Saved checkpoint at episode {episode} (avg score: {avg_score:.1f})"
                        )

                    if episode >= NUM_EPISODES:
                        break

                    state = self.game.reset()

            # Compute last value for GAE
            with torch.no_grad():
                state_tensor = state_to_tensor(state, self.device)
                last_value = self.model.get_value(state_tensor.unsqueeze(0)).item()

            # PPO update
            loss = self._ppo_update(last_value)
            self.loss_tracker.add(loss)
            self.recent_losses.append(loss)
            self.rollout_buffer.clear()

        pbar.close()
        self._save_model("ppo_model.pth")
        self._create_plots()

    def _ppo_update(self, last_value: float) -> float:
        """Perform PPO update on collected rollouts."""
        self.model.train()

        # Get data from buffer
        (
            states,
            actions,
            old_log_probs,
            returns,
            advantages,
            valid_actions_mask,
        ) = self.rollout_buffer.compute_returns_and_advantages(last_value, self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n_samples = len(states)
        total_loss = 0.0
        num_updates = 0

        for _ in range(NUM_EPOCHS):
            # Shuffle indices
            indices = torch.randperm(n_samples, device=self.device)

            for start in range(0, n_samples, MINIBATCH_SIZE):
                end = min(start + MINIBATCH_SIZE, n_samples)
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_valid_mask = valid_actions_mask[batch_indices]

                # Get current policy outputs
                logits, values = self.model(batch_states)

                # Mask invalid actions
                masked_logits = torch.where(
                    batch_valid_mask == 1,
                    logits,
                    torch.tensor(-1e8, device=self.device),
                )
                action_probs = F.softmax(masked_logits, dim=-1)
                dist = Categorical(action_probs)

                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()

                # Compute ratio and clipped surrogate objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)
                    * batch_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                values = values.squeeze(-1)
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    actor_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

                total_loss += loss.item()
                num_updates += 1

        return total_loss / num_updates if num_updates > 0 else 0.0

    def _save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def _create_plots(self) -> None:
        print("Creating plots...")

        plt.plot(self.score_tracker.get_history())
        plt.title("PPO Score History")
        plt.xlabel("Mean over 100 episodes")
        plt.ylabel("Score")
        plt.savefig("ppo_score_history.png")
        plt.close()
        print("Score history plot saved to 'ppo_score_history.png'")

        plt.plot(self.loss_tracker.get_history())
        plt.title("PPO Loss History")
        plt.xlabel("Mean over 100 updates")
        plt.ylabel("Loss")
        plt.savefig("ppo_loss_history.png")
        plt.close()
        print("Loss history plot saved to 'ppo_loss_history.png'")
