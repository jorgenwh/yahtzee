from typing import List, Type

from yahtzee.yahtzee import Yahtzee
from yahtzee.agent import Agent

class EvaluationResult():
    def __init__(self, agent_name: str, scores: List[int]):
        self.agent_name = agent_name
        self.scores = scores
        self.mean = sum(scores) / len(scores)
        self.max = max(scores)
        self.min = min(scores)
        self.episodes = len(scores)

    def __str__(self) -> str:
        s = f"--- Evaluation Result for agent '{self.agent_name}' ---\n"
        s += f"Episodes: {self.episodes}\n"
        s += f"Mean Score: {self.mean}\n"
        s += f"Max Score: {self.max}\n"
        s += f"Min Score: {self.min}\n"
        s += f"Number of episodes: {self.episodes}\n"
        return s

    def __repr__(self) -> str:
        return self.__str__()

class Evaluator():
    def __init__(self, agent_cls: Type[Agent], num_episodes: int = 1000):
        self.agent = agent_cls()
        self.num_episodes = num_episodes

    def evaluate(self) -> EvaluationResult:
        scores = []
        for e in range(self.num_episodes):
            score = self.play_episode()
            scores.append(score)
        return EvaluationResult(self.agent.get_name(), scores)

    def play_episode(self) -> int:
        game = Yahtzee()

        while not game.is_done():
            action = self.agent.get_action(game.state)
            game.step(action)

        score = game.get_score()
        return score

