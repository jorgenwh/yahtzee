import random

from yahtzee.yahtzee import State, Yahtzee
from yahtzee.constants import ACTION_SPACE


def play_out_game(yahtzee: Yahtzee) -> int:
    while not yahtzee.is_done():
        valid_action_indices = [
            i for i, valid_action in enumerate(yahtzee.state.valid_actions) if valid_action
        ]
        action = random.choice(valid_action_indices)
        yahtzee.step(action)
    return yahtzee.get_score()


class MonteCarloTreeSearch():
    def search(self, state: State, iters: int) -> int:
        num_valid_actions = sum(state.valid_actions)
        iters_per_action = iters // num_valid_actions
        action_scores = [0] * ACTION_SPACE

        for action in range(ACTION_SPACE):
            if not state.valid_actions[action]:
                continue

            for _ in range(iters_per_action):
                yahtzee = Yahtzee(state)
                yahtzee.step(action)
                score = play_out_game(yahtzee)
                action_scores[action] += score

        return action_scores.index(max(action_scores))

