import random

from yahtzee.yahtzee import State, Yahtzee
from yahtzee.constants import ACTION_SPACE


def play_out_game(state: State, initial_action: int) -> int:
    game = Yahtzee(state)
    s = game.step(initial_action)

    while not s.is_done:
        valid_action_indices = [
            i for i, valid_action in enumerate(s.valid_actions) if valid_action
        ]
        action = random.choice(valid_action_indices)
        s = game.step(action)

    return s.score


class MonteCarloTreeSearch:
    def search(self, state: State, iters: int) -> int:
        num_valid_actions = sum(state.valid_actions)
        iters_per_action = max(iters // num_valid_actions, 1)
        action_scores = [0] * ACTION_SPACE

        for initial_action in range(ACTION_SPACE):
            if not state.valid_actions[initial_action]:
                continue

            for _ in range(iters_per_action):
                score = play_out_game(state, initial_action)
                action_scores[initial_action] += score

        return action_scores.index(max(action_scores))
