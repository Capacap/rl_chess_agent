# Chess AI Project: Reinforcement Learning Agent ♟️

## Project Overview
This project challenges you to design and implement a chess-playing AI agent using reinforcement learning (RL) techniques. Your agent will compete in a class-wide tournament to determine the best chess AI. You will gain hands-on experience with RL algorithms, game theory, and competitive AI development.

**Submission Deadline: October 22, 2025**

# Technical Specifications

## What to Implement
Your agent must be a Python class that inherits from Agent in agent_interface.py. You are required to implement the make_move() method, which must return a legal move within a 2-second time limit.

## Required Files
Submit only your agent file, renamed to [your_name].py (e.g., my_agent.py). The base code provided includes agent_interface.py, requirements.txt, and other necessary files to get you started.

## Environment Constraints
- CPU only: No GPU acceleration is allowed.
- Time limit: A maximum of 2 seconds per move.
- Memory limit: A maximum of 2GB of memory usage.
- Allowed libraries: Only libraries specified in requirements.txt.

## Implementation Rules
- Your agent must only return legal moves.
- No external communication (e.g., network calls, file system access).
- No use of existing chess engines (e.g., Stockfish).
- No hard-coded opening books or endgame tables.
- You must implement the learning algorithm yourself.

# Technical Approach Suggestions

## Recommended RL Techniques

Consider using one of the following methods:
- Q-learning with state-action value approximation
- Policy gradient methods (REINFORCE)
- Monte Carlo Tree Search (MCTS) with neural networks
- Deep Q-networks (DQN) with experience replay
- Actor-critic methods

## State Representation Ideas

A good state representation is key to a successful agent. Consider using:
- A board representation (e.g., an 8x8x12 tensor for piece types).
- Game state features (e.g., castling rights, en passant).
- Material balance and positional evaluation.
- Historical positions for repetition detection.

## Reward Structure

Define your reward system carefully to guide your agent's learning:
- Win/loss outcome: A simple reward structure (+1 for a win, -1 for a loss, 0 for a draw).
- Material advantage: Reward based on the differential piece values.
- Positional evaluation: Reward for controlling the center, piece activity, or other strategic factors.
- Checkmate threats and tactical opportunities.

## Tips for Success 👍

Do:
- Start simple: Begin with a basic baseline (random or greedy agent) and make incremental improvements.
- Test rigorously: Test your agent against different opponent types.
- Analyze your games: Use PGN output to review your agent's performance and identify areas for improvement.
- Document your approach thoroughly.

Don't:
- Procrastinate: Do not wait until the last minute.
- Use prohibited libraries or external help.
- Hard-code specific sequences like opening moves.
- Ignore the time limits.
