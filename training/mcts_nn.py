"""
mcts_nn.py
Neural network-guided MCTS using PUCT algorithm (AlphaZero-style).

Replaces random rollouts with:
- Policy network for move priors (exploration guidance)
- Value network for leaf evaluation (position assessment)
"""

import chess
import torch
import math
from typing import Optional, Dict, Callable, Tuple
from model.network import ChessNet
from encoding.state import encode_board
from encoding.move import encode_move, create_legal_move_mask


class NeuralMCTSNode:
    """
    MCTS node enhanced with neural network guidance.

    Uses PUCT (Predictor + Upper Confidence Bound) for selection:
        PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    """

    def __init__(
        self,
        board: chess.Board,
        parent: Optional['NeuralMCTSNode'] = None,
        move: Optional[chess.Move] = None,
        prior: float = 0.0
    ):
        """
        Args:
            board: Chess board state at this node
            parent: Parent node (None for root)
            move: Move that led to this state
            prior: Policy network prior probability for this move
        """
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.prior = prior

        # MCTS statistics
        self.visits = 0
        self.total_value = 0.0  # Sum of backpropagated values

        # Children: {move: NeuralMCTSNode}
        self.children: Dict[chess.Move, NeuralMCTSNode] = {}
        self.is_expanded = False

    def is_terminal(self) -> bool:
        """Check if node represents game-over state."""
        return self.board.is_game_over()

    def q_value(self) -> float:
        """
        Mean action value (exploitation term).

        Returns:
            Average value from all simulations through this node
        """
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def select_child(self, c_puct: float = 1.0) -> 'NeuralMCTSNode':
        """
        Select child using PUCT formula.

        Args:
            c_puct: Exploration constant (higher = more exploration)

        Returns:
            Child with highest PUCT score
        """
        def puct_score(child: NeuralMCTSNode) -> float:
            """Compute PUCT = Q + U"""
            # Exploitation: Q-value
            q_value = child.q_value()

            # Exploration: U-value with prior
            u_value = c_puct * child.prior * math.sqrt(self.visits) / (1 + child.visits)

            return q_value + u_value

        return max(self.children.values(), key=puct_score)

    def expand(self, inference_fn: Callable[[str], Tuple[dict, float]]) -> None:
        """
        Expand node with children, using inference function for priors.

        Args:
            inference_fn: Function (board_fen) -> (policy_dict, value)
                         policy_dict = {action_idx: probability}

        Side effects:
            - Creates child nodes for all legal moves
            - Sets self.is_expanded = True
        """
        # Get legal moves
        legal_moves = list(self.board.legal_moves)

        # Request inference
        policy_dict, _ = inference_fn(self.board.fen())

        # Create child for each legal move with prior from policy
        for move in legal_moves:
            action_idx = encode_move(move)
            prior_prob = policy_dict.get(action_idx, 0.0)

            new_board = self.board.copy()
            new_board.push(move)

            self.children[move] = NeuralMCTSNode(
                board=new_board,
                parent=self,
                move=move,
                prior=prior_prob
            )

        self.is_expanded = True

    def evaluate(self, inference_fn: Callable[[str], Tuple[dict, float]]) -> float:
        """
        Evaluate position using inference function.

        Args:
            inference_fn: Function (board_fen) -> (policy_dict, value)

        Returns:
            Value in [-1, 1] from current player's perspective
        """
        _, value = inference_fn(self.board.fen())
        return value

    def update(self, value: float) -> None:
        """
        Update node statistics.

        Args:
            value: Backpropagated value from simulation
        """
        self.visits += 1
        self.total_value += value


def mcts_search(
    board: chess.Board,
    network: ChessNet = None,
    inference_fn: Callable[[str], Tuple[dict, float]] = None,
    num_simulations: int = 40,
    c_puct: float = 1.0,
    time_budget: float = None
) -> Dict[chess.Move, int]:
    """
    Run MCTS search from given position.

    Args:
        board: Starting position
        network: Neural network (creates inference_fn if provided)
        inference_fn: Inference function (board_fen) -> (policy_dict, value)
        num_simulations: Number of MCTS iterations
        c_puct: Exploration constant
        time_budget: Optional time limit in seconds (90% used as safety margin)

    Returns:
        Dictionary mapping moves to visit counts
    """
    import time

    # Create inference function from network if needed
    if inference_fn is None:
        if network is None:
            raise ValueError("Must provide either network or inference_fn")
        inference_fn = _create_direct_inference(network)

    root = NeuralMCTSNode(board)

    # Expand root first to get initial children
    if not root.is_terminal():
        root.expand(inference_fn)

    # Time-based cutoff if budget specified
    start_time = time.time() if time_budget else None
    safety_margin = 0.9  # Use 90% of budget to account for overhead

    for sim in range(num_simulations):
        # Check time budget before each simulation
        if time_budget and (time.time() - start_time) > (time_budget * safety_margin):
            break

        _mcts_iteration(root, inference_fn, c_puct)

    # Return visit counts for each child
    return {move: child.visits for move, child in root.children.items()}


def _create_direct_inference(network: ChessNet) -> Callable[[str], Tuple[dict, float]]:
    """
    Create inference function from network (for backward compatibility).

    Args:
        network: Neural network

    Returns:
        Inference function (board_fen) -> (policy_dict, value)
    """
    def inference_fn(board_fen: str) -> Tuple[dict, float]:
        board = chess.Board(board_fen)
        state = encode_board(board)
        mask = create_legal_move_mask(board)
        legal_mask_tensor = torch.tensor([mask], dtype=torch.bool)

        with torch.no_grad():
            policy, value = network(state, legal_mask_tensor)

        policy_dict = {
            idx: prob.item()
            for idx, prob in enumerate(policy[0])
            if prob > 1e-6
        }

        return policy_dict, value.item()

    return inference_fn


def _mcts_iteration(
    root: NeuralMCTSNode,
    inference_fn: Callable[[str], Tuple[dict, float]],
    c_puct: float
) -> None:
    """
    Single MCTS iteration: select, expand, evaluate, backpropagate.

    Args:
        root: Root node of search tree
        inference_fn: Inference function (board_fen) -> (policy_dict, value)
        c_puct: Exploration constant
    """
    # Phase 1: Selection - traverse tree using PUCT
    node = root
    search_path = [node]

    while node.is_expanded and not node.is_terminal():
        node = node.select_child(c_puct)
        search_path.append(node)

    # Phase 2: Expansion & Evaluation
    if node.is_terminal():
        # Terminal node: use game outcome
        outcome = node.board.outcome()
        if outcome.winner is None:
            value = 0.0
        elif outcome.winner == node.board.turn:
            value = 1.0
        else:
            value = -1.0
    else:
        # Non-terminal: expand and evaluate with inference
        node.expand(inference_fn)
        value = node.evaluate(inference_fn)

    # Phase 3: Backpropagation
    _backpropagate(search_path, value)


def _backpropagate(search_path: list[NeuralMCTSNode], value: float) -> None:
    """
    Propagate value up the search path.

    Args:
        search_path: List of nodes from root to leaf
        value: Evaluation from leaf node's perspective
    """
    for node in reversed(search_path):
        node.update(value)
        value = -value  # Flip perspective for parent
