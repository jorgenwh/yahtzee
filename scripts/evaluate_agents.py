from yahtzee import Arena

arena = Arena(num_episodes=10000)
arena.run()
arena.graph_results("assets/agentperformances.png")
