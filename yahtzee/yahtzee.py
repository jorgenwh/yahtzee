from dataclasses import dataclass, field
from typing import List
import random
import re

from yahtzee.constants import NUM_DICE, CATEGORIES, ACTIONS, ACTION_SPACE


@dataclass
class State:
    score: int = 0
    remaining_rolls: int = 2
    is_done: bool = False
    dice: List[int] = field(
        default_factory=lambda: [random.randint(0, 6) for _ in range(NUM_DICE)]
    )
    available_categories: List[int] = field(
        default_factory=lambda: [1] * len(CATEGORIES)
    )
    valid_actions: List[int] = field(default_factory=lambda: [1] * ACTION_SPACE)


class Yahtzee:
    def __init__(self, state: State = State()):
        self._state = State(
            score=state.score,
            dice=state.dice.copy(),
            available_categories=state.available_categories.copy(),
            remaining_rolls=state.remaining_rolls,
            valid_actions=state.valid_actions.copy(),
            is_done=state.is_done,
        )

    def reset(self) -> State:
        self._state = State()
        return self._state

    def step(self, action: int) -> State:
        if not self._state.valid_actions[action]:
            raise ValueError(f"Invalid action: {action}")

        if self._action_is_roll(action):
            action_name = ACTIONS[action]
            dice_indices = [int(dice) - 1 for dice in re.findall(r"\d{1}", action_name)]
            self._roll_dice(dice_indices=dice_indices)
        else:
            category_idx = action - 31
            category = CATEGORIES[category_idx]

            self._state.score += check_category(category, self._state.dice)
            self._state.available_categories[category_idx] = 0
            self._state.remaining_rolls = 3
            self._roll_dice()

        self._update_valid_actions()
        self._state.is_done = not any(self._state.available_categories)

        return State(
            score=self._state.score,
            dice=self._state.dice.copy(),
            available_categories=self._state.available_categories.copy(),
            remaining_rolls=self._state.remaining_rolls,
            valid_actions=self._state.valid_actions.copy(),
            is_done=self._state.is_done,
        )

    def _action_is_roll(self, action: int) -> bool:
        return action < 31

    def _roll_dice(self, dice_indices=None):
        if dice_indices is None:
            dice_indices = range(NUM_DICE)
        for i in dice_indices:
            self._state.dice[i] = random.randint(1, 6)
        self._state.remaining_rolls -= 1

    def _update_valid_actions(self) -> None:
        roll_actions = [int(self._state.remaining_rolls > 0)] * 31
        category_actions = self._state.available_categories
        valid_actions = roll_actions + category_actions
        self._state.valid_actions = valid_actions


def check_category(category: str, dice: List[int]) -> int:
    match category:
        case "aces":
            return check_aces(dice)
        case "twos":
            return check_twos(dice)
        case "threes":
            return check_threes(dice)
        case "fours":
            return check_fours(dice)
        case "fives":
            return check_fives(dice)
        case "sixes":
            return check_sixes(dice)
        case "chance":
            return check_chance(dice)
        case "three_of_a_kind":
            return check_three_of_a_kind(dice)
        case "four_of_a_kind":
            return check_four_of_a_kind(dice)
        case "full_house":
            return check_full_house(dice)
        case "small_straight":
            return check_small_straight(dice)
        case "large_straight":
            return check_large_straight(dice)
        case "yahtzee":
            return check_yahtzee(dice)
        case _:
            raise ValueError(f"Invalid category: {category}")


def check_aces(dice: List[int]) -> int:
    return dice.count(1)


def check_twos(dice: List[int]) -> int:
    return dice.count(2) * 2


def check_threes(dice: List[int]) -> int:
    return dice.count(3) * 3


def check_fours(dice: List[int]) -> int:
    return dice.count(4) * 4


def check_fives(dice: List[int]) -> int:
    return dice.count(5) * 5


def check_sixes(dice: List[int]) -> int:
    return dice.count(6) * 6


def check_chance(dice: List[int]) -> int:
    return sum(dice)


def check_three_of_a_kind(dice: List[int]) -> int:
    for i in range(1, 7):
        if dice.count(i) >= 3:
            return sum(dice)
    return 0


def check_four_of_a_kind(dice: List[int]) -> int:
    for i in range(1, 7):
        if dice.count(i) >= 4:
            return sum(dice)
    return 0


def check_full_house(dice: List[int]) -> int:
    if len(set(dice)) == 2:
        return 25
    return 0


def check_small_straight(dice: List[int]) -> int:
    for valid_sequence in [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [1, 2, 3, 4, 5]]:
        if sorted(set(dice)) == valid_sequence:
            return 30
    return 0


def check_large_straight(dice: List[int]) -> int:
    if sorted(set(dice)) in [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]:
        return 40
    return 0


def check_yahtzee(dice: List[int]) -> int:
    return int(len(set(dice)) == 1) * 50
