# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (uses uv)
uv sync

# Install with dev dependencies
uv sync --all-extras

# Lint and format
black yahtzee/
ruff check --fix yahtzee/
pyright yahtzee/

# Evaluate all agents
python scripts/evaluate_agents.py --episodes 10000

# Train DQN agent
python scripts/train_dqn_agent.py
```

## Architecture

This is a Yahtzee game environment with pluggable AI agents.

### Core Game (`yahtzee/yahtzee.py`)
- `State` dataclass: Holds game state (score, dice, remaining_rolls, available_categories, valid_actions, is_done)
- `Yahtzee` class: Game environment with `reset()` and `step(action)` methods
- Action space: 44 total actions (indices 0-30 are dice roll combinations, 31-43 are category selections)

### Agent System (`yahtzee/agent.py`)
Base `Agent` class requires implementing:
- `get_action(state: State) -> int`: Return action index
- `get_name() -> str`: Agent display name

### Adding New Agents
1. Create `yahtzee/agents/<your_agent>/<your_agent>.py` inheriting from `Agent`
2. Export in `yahtzee/agents/__init__.py`
3. Import and add to `AGENTS` list in `yahtzee/arena.py`

### Existing Agents
- `LowestActionAgent`: Always picks lowest valid action (baseline)
- `VeryGreedyAgent`: Greedy category selection
- `MctsAgent`: Monte Carlo Tree Search
- `DQNAgent`: Deep Q-Network (PyTorch) - model in `yahtzee/agents/dqn_agent/`

### Evaluation (`yahtzee/arena.py`, `yahtzee/evaluator.py`)
`Arena` runs all agents in `AGENTS` list and generates comparison chart. `Evaluator` plays episodes and collects statistics.
