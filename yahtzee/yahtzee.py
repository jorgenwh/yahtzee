
NUM_DICE = 5
CATEGORIES = [
    "aces",
    "twos",
    "threes",
    "fours",
    "fives",
    "sixes",
    "chance",
    "three_of_a_kind",
    "four_of_a_kind",
    "full_house",
    "small_straight",
    "large_straight",
    "yahtzee",
]

class Yahtzee():
    def __init__(self):
        self.dice = [0] * NUM_DICE
