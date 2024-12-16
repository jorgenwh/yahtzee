from typing import List, Dict
import random
import re

from yahtzee.constants import (
        MAXI_YATZY_NUM_DICE, 
        MAXI_YATZY_CATEGORIES, 
        MAXI_YATZY_ACTIONS, 
        MAXI_YATZY_ACTION_SPACE
)


class MaxiYatzyState:
    def __init__(
        self,
        score: int = 0,
        dice: List[int] = [0] * MAXI_YATZY_NUM_DICE,
        available_categories: Dict[str, int] = {category: 1 for category in MAXI_YATZY_CATEGORIES},
        remaining_rolls: int = 3,
        valid_actions: List[int] = [1] * MAXI_YATZY_ACTION_SPACE,
    ):
        self.score = score
        self.dice = dice
        self.available_categories = available_categories
        self.remaining_rolls = remaining_rolls
        self.valid_actions = valid_actions

    def copy(self):
        return MaxiYatzyState(
            score=self.score,
            dice=self.dice.copy(),
            available_categories=self.available_categories.copy(),
            remaining_rolls=self.remaining_rolls,
            valid_actions=self.valid_actions.copy(),
        )

    def __str__(self) -> str:
        s = "--------- Maxi Yatzy Game State ---------\n"
        s += f"Score: {self.score}\n"
        s += f"Dice: {self.dice}\n"
        s += f"Remaining rolls: {self.remaining_rolls}\n"
        s += f"Available categories: {self.available_categories}\n"
        s += f"Valid actions: {self.valid_actions}"
        return s

    def __repr__(self) -> str:
        return self.__str__()


class MaxiYatzy:
    def __init__(self, state=None):
        if state is None:
            self._state = MaxiYatzyState()
            self.roll_dice()
            for category in MAXI_YATZY_CATEGORIES:
                self._state.available_categories[category] = 1
        else:
            self._state = state.copy()

    @property
    def state(self) -> MaxiYatzyState:
        return self._state

    @state.setter
    def state(self, state: MaxiYatzyState):
        self._state = state

    def update_valid_actions(self) -> None:
        roll_actions = [int(self._state.remaining_rolls > 0)] * 31
        category_actions = [
            self._state.available_categories[category] for category in MAXI_YATZY_CATEGORIES
        ]
        valid_actions = roll_actions + category_actions
        self._state.valid_actions = valid_actions

    def get_score(self) -> int:
        return self._state.score

    def is_done(self) -> bool:
        return not any(self._state.available_categories.values())

    def step(self, action: int) -> None:
        if not self._state.valid_actions[action]:
            raise ValueError(f"Invalid action: {action}")

        if action >= 31:
            category = MAXI_YATZY_CATEGORIES[action - 31]
            self._state.score += check_category(category, self._state.dice)
            self._state.available_categories[category] = 0
            self._state.remaining_rolls = 3
            self.roll_dice()
        else:
            action_name = MAXI_YATZY_ACTIONS[action]
            dice_indices = [int(dice) - 1 for dice in re.findall(r"\d{1}", action_name)]
            self.roll_dice(dice_indices=dice_indices)

        self.update_valid_actions()

    def roll_dice(self, dice_indices=None):
        if dice_indices is None:
            dice_indices = range(MAXI_YATZY_NUM_DICE)
        for i in dice_indices:
            self._state.dice[i] = random.randint(1, 6)
        self._state.remaining_rolls -= 1


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
