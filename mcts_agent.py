"""
mcts_agent.py
A Monte Carlo Tree Search agent for chess using random rollouts.
This is a foundation implementation that can later be enhanced with a neural network.
"""

import chess
import random
import math
import time
from agent_interface import Agent


class MCTSNode:
    """
    A node in the MCTS tree.
    Each node represents a game state and tracks visit statistics.
    """
    
    def __init__(self, board: chess.Board, parent=None, move=None):
        """
        Args:
            board: The chess board state at this node
            parent: The parent MCTSNode
            move: The move that led to this node from parent
        """
        self.board = board.copy()  # Store the board state
        self.parent = parent
        self.move = move  # The move that led to this state
        
        # MCTS statistics
        self.visits = 0
        self.wins = 0.0  # Can be fractional for draws
        
        # Children
        self.children = []
        self.untried_moves = list(board.legal_moves)
        random.shuffle(self.untried_moves)  # Randomize exploration order
        
    def is_fully_expanded(self):
        """Check if all possible moves from this node have been tried."""
        return len(self.untried_moves) == 0
    
    def is_terminal(self):
        """Check if this node represents a game-over state."""
        return self.board.is_game_over()
    
    def best_child(self, exploration_weight=1.41):
        """
        Select the best child using UCB1 formula.
        
        UCB1 = (wins/visits) + exploration_weight * sqrt(ln(parent_visits)/visits)
        
        Args:
            exploration_weight: Balance between exploitation and exploration (c parameter)
        
        Returns:
            The child node with highest UCB1 score
        """
        def ucb1_score(child):
            if child.visits == 0:
                return float('inf')  # Prioritize unvisited nodes
            
            # Exploitation term: win rate from parent's perspective
            exploit = child.wins / child.visits
            
            # Exploration term: uncertainty bonus
            explore = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            
            return exploit + explore
        
        return max(self.children, key=ucb1_score)
    
    def expand(self):
        """
        Expand the tree by adding a new child node for an untried move.
        
        Returns:
            The newly created child node
        """
        move = self.untried_moves.pop()
        
        # Create new board state
        new_board = self.board.copy()
        new_board.push(move)
        
        # Create child node
        child = MCTSNode(new_board, parent=self, move=move)
        self.children.append(child)
        
        return child
    
    def update(self, result):
        """
        Update node statistics after a simulation.
        
        Args:
            result: 1 for win, 0 for loss, 0.5 for draw (from current player's perspective)
        """
        self.visits += 1
        self.wins += result


class MCTSAgent(Agent):
    """
    A chess agent using Monte Carlo Tree Search with random rollouts.
    """
    
    def __init__(self, board: chess.Board, color: chess.Color):
        super().__init__(board, color)
        self.color = color
        
    def make_move(self, board: chess.Board, time_limit: float) -> chess.Move:
        """
        Use MCTS to select the best move.
        
        Args:
            board: Current game state
            time_limit: Maximum time allowed for this move
        
        Returns:
            The best move found by MCTS
        """
        # Reserve some time for safety margin
        time_budget = time_limit - 0.1  # Reserve 100ms
        start_time = time.time()
        
        # Create root node
        root = MCTSNode(board)
        
        # Run MCTS simulations until time runs out
        simulations = 0
        while time.time() - start_time < time_budget:
            # Run one MCTS iteration
            self._mcts_iteration(root)
            simulations += 1
            
            # Safety check: if we're getting close to time limit, stop
            if time.time() - start_time > time_budget * 0.95:
                break
        
        # Select the move from the most visited child
        # (Most robust selection criterion)
        if not root.children:
            # Fallback: if no simulations completed, play random move
            return random.choice(list(board.legal_moves))
        
        best_child = max(root.children, key=lambda c: c.visits)
        
        # Optional: Print statistics for debugging
        # print(f"MCTS: {simulations} simulations, chose {best_child.move} "
        #       f"(visits: {best_child.visits}, win_rate: {best_child.wins/best_child.visits:.2%})")
        
        return best_child.move
    
    def _mcts_iteration(self, root: MCTSNode):
        """
        Perform one complete MCTS iteration: Selection, Expansion, Simulation, Backpropagation.
        
        Args:
            root: The root node of the search tree
        """
        # Phase 1: Selection - walk down tree using UCB1
        node = root
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child()
        
        # Phase 2: Expansion - add a new child if possible
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()
        
        # Phase 3: Simulation - play out the game randomly
        result = self._simulate(node.board)
        
        # Phase 4: Backpropagation - update statistics up the tree
        self._backpropagate(node, result)
    
    def _simulate(self, board: chess.Board) -> float:
        """
        Simulate a random game from the given board state.

        Args:
            board: Starting position for simulation

        Returns:
            Result from root player's perspective: 1.0 (win), 0.0 (loss), 0.5 (draw)
        """
        sim_board = board.copy()
        root_player = sim_board.turn

        # Play random moves until game ends
        # Reduced from 200 to 50 for faster rollouts
        max_moves = 50
        moves_played = 0

        while not sim_board.is_game_over() and moves_played < max_moves:
            legal_moves = list(sim_board.legal_moves)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            sim_board.push(move)
            moves_played += 1

        # Evaluate result from root player's perspective
        if sim_board.is_game_over():
            outcome = sim_board.outcome()
            if outcome.winner is None:
                return 0.5  # Draw
            elif outcome.winner == root_player:
                return 1.0  # Win
            else:
                return 0.0  # Loss
        else:
            # Hit move limit - use simple material evaluation
            return self._evaluate_position(sim_board, root_player)
    
    def _evaluate_position(self, board: chess.Board, player: chess.Color) -> float:
        """
        Simple material-based evaluation when rollout hits move limit.

        Args:
            board: Board position to evaluate
            player: Player to evaluate from perspective of

        Returns:
            Score between 0.0 and 1.0
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }

        white_score = 0
        black_score = 0

        for piece_type in piece_values:
            white_score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
            black_score += len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]

        # Normalize to 0-1 range
        total = white_score + black_score
        if total == 0:
            return 0.5

        if player == chess.WHITE:
            return white_score / total
        else:
            return black_score / total

    def _backpropagate(self, node: MCTSNode, result: float):
        """
        Propagate simulation result up the tree, updating statistics.

        Args:
            node: The leaf node where simulation started
            result: Game result from the perspective of the player at that node
        """
        while node is not None:
            # Update this node's statistics
            # Note: result is flipped at each level (opponent's perspective)
            node.update(result)

            # Move up the tree and flip result
            result = 1.0 - result
            node = node.parent


# Test the agent
if __name__ == "__main__":
    test_board = chess.Board()
    agent = MCTSAgent(test_board, chess.WHITE)
    
    print("Testing MCTSAgent...")
    print(f"Starting position:\n{test_board}\n")
    
    # Make a move with 2 second time limit
    move = agent.make_move(test_board, 2.0)
    print(f"MCTSAgent chose: {move}")
    
    test_board.push(move)
    print(f"\nPosition after move:\n{test_board}")
