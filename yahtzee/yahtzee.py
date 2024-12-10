from typing import List, Dict
import random
import re

from yahtzee.constants import NUM_DICE, CATEGORIES, ACTIONS, ACTION_SPACE


class State:
    def __init__(
        self,
        dice: List[int] = [0] * NUM_DICE,
        available_categories: Dict[str, int] = {category: 1 for category in CATEGORIES},
        remaining_rolls: int = 3,
        valid_actions: List[int] = [1] * ACTION_SPACE,
    ):
        self.dice = dice
        self.available_categories = available_categories
        self.remaining_rolls = remaining_rolls
        self.valid_actions = valid_actions

    def __str__(self) -> str:
        s = "--------- Game State ---------\n"
        s += f"Dice: {self.dice}\n"
        s += f"Remaining rolls: {self.remaining_rolls}\n"
        s += f"Available categories: {self.available_categories}\n"
        s += f"Valid actions: {self.valid_actions}"
        return s

    def __repr__(self) -> str:
        return self.__str__()


class Yahtzee:
    def __init__(self):
        self._score = 0
        self._state = State()
        self.roll_dice()

        for category in CATEGORIES:
            self._state.available_categories[category] = 1

    @property
    def state(self) -> State:
        return self._state

    @state.setter
    def state(self, state: State):
        self._state = state

    def update_valid_actions(self) -> None:
        roll_actions = [int(self._state.remaining_rolls > 0)] * 31
        category_actions = [
            self._state.available_categories[category] for category in CATEGORIES
        ]
        valid_actions = roll_actions + category_actions
        self._state.valid_actions = valid_actions

    def get_score(self) -> int:
        return self._score

    def is_done(self) -> bool:
        return not any(self._state.available_categories.values())

    def step(self, action: int) -> None:
        if not self._state.valid_actions[action]:
            raise ValueError(f"Invalid action: {action}")

        if action >= 31:
            category = CATEGORIES[action - 31]
            self._score += self.check_category(category)
            self._state.available_categories[category] = 0
            self._state.remaining_rolls = 3
            self.roll_dice()
        else:
            action_name = ACTIONS[action]
            dice_indices = [int(dice) - 1 for dice in re.findall(r"\d{1}", action_name)]
            self.roll_dice(dice_indices=dice_indices)

        self.update_valid_actions()

    def roll_dice(self, dice_indices=None):
        if dice_indices is None:
            dice_indices = range(NUM_DICE)
        for i in dice_indices:
            self._state.dice[i] = random.randint(1, 6)
        self._state.remaining_rolls -= 1

    def check_category(self, category: str) -> int:
        match category:
            case "aces":
                return _check_aces(self._state.dice)
            case "twos":
                return _check_twos(self._state.dice)
            case "threes":
                return _check_threes(self._state.dice)
            case "fours":
                return _check_fours(self._state.dice)
            case "fives":
                return _check_fives(self._state.dice)
            case "sixes":
                return _check_sixes(self._state.dice)
            case "chance":
                return _check_chance(self._state.dice)
            case "three_of_a_kind":
                return _check_three_of_a_kind(self._state.dice)
            case "four_of_a_kind":
                return _check_four_of_a_kind(self._state.dice)
            case "full_house":
                return _check_full_house(self._state.dice)
            case "small_straight":
                return _check_small_straight(self._state.dice)
            case "large_straight":
                return _check_large_straight(self._state.dice)
            case "yahtzee":
                return _check_yahtzee(self._state.dice)
            case _:
                raise ValueError(f"Invalid category: {category}")


def _check_aces(dice: List[int]) -> int:
    return dice.count(1)


def _check_twos(dice: List[int]) -> int:
    return dice.count(2) * 2


def _check_threes(dice: List[int]) -> int:
    return dice.count(3) * 3


def _check_fours(dice: List[int]) -> int:
    return dice.count(4) * 4


def _check_fives(dice: List[int]) -> int:
    return dice.count(5) * 5


def _check_sixes(dice: List[int]) -> int:
    return dice.count(6) * 6


def _check_chance(dice: List[int]) -> int:
    return sum(dice)


def _check_three_of_a_kind(dice: List[int]) -> int:
    for i in range(1, 7):
        if dice.count(i) >= 3:
            return sum(dice)
    return 0


def _check_four_of_a_kind(dice: List[int]) -> int:
    for i in range(1, 7):
        if dice.count(i) >= 4:
            return sum(dice)
    return 0


def _check_full_house(dice: List[int]) -> int:
    if len(set(dice)) == 2:
        return 25
    return 0


def _check_small_straight(dice: List[int]) -> int:
    for valid_sequence in [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [1, 2, 3, 4, 5]]:
        if sorted(set(dice)) == valid_sequence:
            return 30
    return 0


def _check_large_straight(dice: List[int]) -> int:
    if sorted(set(dice)) in [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]:
        return 40
    return 0


def _check_yahtzee(dice: List[int]) -> int:
    return int(len(set(dice)) == 1) * 50
