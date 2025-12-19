import time
from typing import List, Type, Union

from yahtzee.yahtzee import Yahtzee
from yahtzee.agent import Agent
from yahtzee.utils import s2ts


class EvaluationResult:
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


class Evaluator:
    def __init__(self, agent: Union[Type[Agent], Agent], num_episodes: int = 1000):
        # Accept either an agent class or an agent instance
        if isinstance(agent, Agent):
            self.agent = agent
        else:
            self.agent = agent()
        self.num_episodes = num_episodes

    def evaluate(self) -> EvaluationResult:
        print(f"Evaluating agent '{self.agent.get_name()}'")

        scores = []
        t0 = time.time()
        for e in range(self.num_episodes):
            t = time.time() - t0
            time_per_episode = t / (e + 1)
            remaining_time = time_per_episode * (self.num_episodes - (e + 1))

            elapsed = s2ts(int(t))
            remaining = s2ts(int(remaining_time))

            print(
                f"Playing episode {e + 1:,}/{self.num_episodes:,} - Elapsed time: {elapsed} - Remaining time: {remaining}",
                end="\r" if e < self.num_episodes - 1 else "\n",
                flush=(e < self.num_episodes - 1),
            )

            score = self.play_episode()
            scores.append(score)

        return EvaluationResult(self.agent.get_name(), scores)

    def play_episode(self) -> int:
        game = Yahtzee()
        state = game.reset()

        while not state.is_done:
            action = self.agent.get_action(state)
            state = game.step(action)

        return state.score
