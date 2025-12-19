import argparse

from yahtzee import Arena
from yahtzee.agents import (
    LowestActionAgent,
    VeryGreedyAgent,
    MctsAgent,
    DQNAgent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        type=int,
        default=2500,
        help="Number of episodes to run each agent for",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA GPU for DQN agent (default: CPU)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model.pth",
        help="Path to DQN model file (default: model.pth)",
    )
    return parser.parse_args()


args = parse_args()

agents = [
    LowestActionAgent,
    VeryGreedyAgent,
    MctsAgent,
    DQNAgent(model_path=args.model, use_cuda=args.cuda),
]

arena = Arena(agents=agents, num_episodes=args.episodes)
arena.run()
arena.graph_results("assets/agentperformances.png")
