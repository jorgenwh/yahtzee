from yahtzee.agent import Agent
from yahtzee.yahtzee import State, check_category
from yahtzee.constants import CATEGORIES, CATEGORY_TO_ACTION


class VeryGreedyAgent(Agent):
    def get_action(self, state: State) -> int:
        dice = state.dice

        best_category = ""
        best_score = -1

        for category in CATEGORIES:
            if state.available_categories[category]:
                score = check_category(category, dice)
                if score > best_score:
                    best_score = score
                    best_category = category

        return CATEGORY_TO_ACTION[best_category]

    def get_name(self) -> str:
        return "Very Greedy Agent"
