import random
from typing import List, Type
from matplotlib import pyplot as plt

from yahtzee.agent import Agent
from yahtzee.evaluator import Evaluator, EvaluationResult
from yahtzee.constants import MAXIMUM_THEORETICAL_SCORE, EXPECTED_RANDOM_PLAY_SCORE


# Import your agent(s) here
from yahtzee.agents import (
    #RandomAgent, 
    LowestActionAgent,
)


# Add your agent(s) here
AGENTS: List[Type[Agent]] = [
        #RandomAgent, 
        LowestActionAgent,
]


class Arena():
    def __init__(self, num_episodes: int = 1000):
        self.evaluation_results: List[EvaluationResult] = []
        self.num_episodes = num_episodes

    def run(self) -> None:
        for agent_cls in AGENTS:
            evaluator = Evaluator(agent_cls, self.num_episodes)
            evaluation_result = evaluator.evaluate()
            self.evaluation_results.append(evaluation_result)

    def graph_results(self, output_filename: str) -> None:
        agent_names = [result.agent_name for result in self.evaluation_results]
        mean_scores = [result.mean for result in self.evaluation_results]
        colors = ["pink", "green", "orange", "cyan", "red", "olive"]
        random.shuffle(colors)
        colors = [colors[i % 6] for i in range(len(agent_names))]

        _, ax = plt.subplots(figsize=(9, 9))
        bars = ax.bar(agent_names, mean_scores, color=colors)

        # Make edge colors of the axes to white
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')

        # Make tick labels white
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        for bar, score in zip(bars, mean_scores):
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            ax.text(x, y, f"{score:.1f}", ha="center", va="bottom", color="white")

        # Add a line indicating the theoretical maximum score
        plt.axhline(
                y=MAXIMUM_THEORETICAL_SCORE, 
                color="white", 
                linestyle="--", 
                label="Theoretical maximum score"
        )

        # Add a line indicating the mean score of random play
        plt.axhline(
                y=EXPECTED_RANDOM_PLAY_SCORE, 
                color="cyan", 
                linestyle="--", 
                label="Random play"
        )

        plt.title(f"mean agent scores after {self.num_episodes} episodes", color="white")
        plt.xlabel("agent", color="white")
        plt.ylabel("mean score", color="white")

        plt.legend(
                loc="upper center", 
                bbox_to_anchor=(0.5, -0.1), 
                ncol=3,
                frameon=False,
        )
        for text in ax.get_legend().get_texts():
            text.set_color("white")
        plt.subplots_adjust(bottom=0.2)

        # set background color of image and plot
        plt.gcf().set_facecolor((0.1, 0.1, 0.2))
        ax.set_facecolor((0.1, 0.1, 0.2))

        plt.savefig(output_filename)
