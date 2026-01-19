import argparse

from yahtzee import Arena
from yahtzee.agents import (
    LowestActionAgent,
    VeryGreedyAgent,
    MctsAgent,
    PPOAgent,
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
        help="Use CUDA GPU for PPO agent (default: CPU)",
    )
    parser.add_argument(
        "--ppo-model",
        type=str,
        default="ppo_model.pth",
        help="Path to PPO model file (default: ppo_model.pth)",
    )
    return parser.parse_args()


args = parse_args()

agents = [
    LowestActionAgent,
    VeryGreedyAgent,
    MctsAgent,
    PPOAgent(model_path=args.ppo_model, use_cuda=args.cuda),
]

arena = Arena(agents=agents, num_episodes=args.episodes)
arena.run()
arena.graph_results("assets/agentperformances.png")
