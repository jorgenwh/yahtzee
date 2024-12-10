#!/bin/bash

# Run black
black yahtzee/

# Run ruff
ruff check --fix yahtzee/

# Run pyright
pyright yahtzee/
