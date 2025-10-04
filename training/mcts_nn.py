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
from typing import Optional, Dict
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

    def expand(self, network: ChessNet) -> None:
        """
        Expand node with children, using network for priors and evaluation.

        Args:
            network: Neural network for policy/value prediction

        Side effects:
            - Creates child nodes for all legal moves
            - Sets self.is_expanded = True
        """
        # Encode board state
        state = encode_board(self.board)

        # Get legal moves mask
        legal_moves = list(self.board.legal_moves)
        mask = create_legal_move_mask(self.board)
        legal_mask_tensor = torch.tensor([mask], dtype=torch.bool)

        # Forward pass through network (CPU for inference)
        network.eval()
        network.cpu()
        with torch.no_grad():
            policy, _ = network(state, legal_mask_tensor)

        # Create child for each legal move with prior from policy
        for move in legal_moves:
            action_idx = encode_move(move)
            prior_prob = policy[0, action_idx].item()

            new_board = self.board.copy()
            new_board.push(move)

            self.children[move] = NeuralMCTSNode(
                board=new_board,
                parent=self,
                move=move,
                prior=prior_prob
            )

        self.is_expanded = True

    def evaluate(self, network: ChessNet) -> float:
        """
        Evaluate position using value network.

        Args:
            network: Neural network for value prediction

        Returns:
            Value in [-1, 1] from current player's perspective
        """
        # Encode board state
        state = encode_board(self.board)

        # Get legal mask (needed for forward pass)
        mask = create_legal_move_mask(self.board)
        legal_mask_tensor = torch.tensor([mask], dtype=torch.bool)

        # Forward pass (CPU for inference)
        network.eval()
        network.cpu()
        with torch.no_grad():
            _, value = network(state, legal_mask_tensor)

        return value.item()

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
    network: ChessNet,
    num_simulations: int = 40,
    c_puct: float = 1.0
) -> Dict[chess.Move, int]:
    """
    Run MCTS search from given position.

    Args:
        board: Starting position
        network: Neural network for guidance
        num_simulations: Number of MCTS iterations
        c_puct: Exploration constant

    Returns:
        Dictionary mapping moves to visit counts
    """
    root = NeuralMCTSNode(board)

    # Expand root first to get initial children
    if not root.is_terminal():
        root.expand(network)

    for _ in range(num_simulations):
        _mcts_iteration(root, network, c_puct)

    # Return visit counts for each child
    return {move: child.visits for move, child in root.children.items()}


def _mcts_iteration(root: NeuralMCTSNode, network: ChessNet, c_puct: float) -> None:
    """
    Single MCTS iteration: select, expand, evaluate, backpropagate.

    Args:
        root: Root node of search tree
        network: Neural network for evaluation
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
        # Non-terminal: expand and evaluate with network
        node.expand(network)
        value = node.evaluate(network)

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
