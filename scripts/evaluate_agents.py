import argparse

from yahtzee import Arena


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        type=int,
        default=10000,
        help="Number of episodes to run each agent for",
    )
    return parser.parse_args()


args = parse_args()

arena = Arena(num_episodes=args.episodes)
arena.run()
arena.graph_results("assets/agentperformances.png")
