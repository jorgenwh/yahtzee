from typing import List, Type
from matplotlib import pyplot as plt

from yahtzee.agent import Agent
from yahtzee.evaluator import Evaluator, EvaluationResult

# Import your agent(s) here
from yahtzee.agents import RandomAgent, LowestActionAgent

# Add your agent(s) here
AGENTS: List[Type[Agent]] = [RandomAgent, LowestActionAgent]

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
        c = ["blue", "green", "red", "cyan"]
        colors = [c[i % len(c)] for i in range(len(agent_names))]

        _, ax = plt.subplots()
        bars = ax.bar(agent_names, mean_scores, color=colors)
        for bar, score in zip(bars, mean_scores):
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            ax.text(x, y, f"{score:.1f}", ha="center", va="bottom")

        plt.title(f"Mean agent scores after {self.num_episodes} episodes")
        plt.xlabel("agent")
        plt.ylabel("mean score")
        plt.savefig(output_filename)
