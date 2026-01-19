import argparse

from yahtzee.agents.ppo_agent.train import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA GPU for training (default: CPU)",
    )
    return parser.parse_args()


args = parse_args()

trainer = Trainer(use_cuda=args.cuda)
trainer.run()
