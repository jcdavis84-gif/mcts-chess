import chess
import numpy as np
from typing import Optional, Dict, List
from .board_encoder import BoardEncoder
from .action_converter import ActionConverter


class MCTSNode:
    """
    A node in the MCTS tree.

    Attributes:
        prior: P(s, a) from the network
        visit_count: N(s, a)
        total_value: W(s, a)
        children: Dict[action_idx, MCTSNode]
    """

    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.children: Dict[int, MCTSNode] = {}

    def value(self) -> float:
        """Q(s, a) = W(s, a) / N(s, a)"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def is_expanded(self) -> bool:
        return len(self.children) > 0


class MCTS:
    """
    Monte Carlo Tree Search with PUCT selection, Dirichlet noise, and chess integration.

    Key features:
    - PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(sum N) / (1 + N(s,a))
    - Dirichlet noise at root (self-play only)
    - Single board instance for tree traversal
    - Proper backup with value negation
    """

    def __init__(
        self,
        model,
        encoder: BoardEncoder,
        converter: ActionConverter,
        num_simulations: int = 20,
        c_puct: float = 1.5,
        temperature: float = 1.0,
        add_dirichlet_noise: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        verbose: bool = False,
        verbose_interval: int = 20,
        trace_move: Optional[str] = None,
    ):
        self.model = model
        self.encoder = encoder
        self.converter = converter
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.add_dirichlet_noise = add_dirichlet_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.trace_move = trace_move  # UCI notation (e.g., "d2h6") to trace

        # Tree reuse: maintain persistent root across moves
        self.root: Optional[MCTSNode] = None
        self.root_board_hash: Optional[int] = None

    def search(self, board: chess.Board) -> np.ndarray:
        """
        Run MCTS simulations and return the visit count distribution.

        Args:
            board: Current chess board state

        Returns:
            pi: (4672,) array of visit count proportions
        """
        # Calculate board hash for tree reuse
        board_hash = hash(board.fen())

        # Try to reuse existing tree
        if self.root is not None and self.root_board_hash == board_hash:
            root = self.root
            if self.verbose:
                print(f"[TREE REUSE] Reusing tree with {root.visit_count} visits")
        else:
            # Create new root node
            root = MCTSNode(prior=1.0)
            self._expand_node(root, board)
            if self.verbose:
                print(f"[NEW TREE] Created fresh tree")

        # Add Dirichlet noise to root if in self-play mode
        if self.add_dirichlet_noise and root.is_expanded():
            self._add_dirichlet_noise(root)

        # Run simulations
        for sim_num in range(self.num_simulations):
            # Clone board for this simulation
            sim_board = board.copy()

            # Check if we should log this simulation
            should_log = self.verbose and (sim_num % self.verbose_interval == 0)

            # Check if trace move will be explored (need to peek at selection)
            if self.trace_move and not should_log and root.is_expanded():
                # Peek at which move would be selected
                action_idx, _ = self._select_child(root, sim_board, should_log=False)
                move = self.converter.action_to_move(action_idx, sim_board)
                if move.uci() == self.trace_move:
                    should_log = True
                    print(f"\n[TRACE: {self.trace_move}]")

            if should_log:
                print(f"\n{'='*70}")
                print(f"Simulation {sim_num + 1}/{self.num_simulations}")
                print(f"{'='*70}")

            self._simulate(root, sim_board, sim_num, should_log)

        # Store root for potential reuse
        self.root = root
        self.root_board_hash = board_hash

        # Extract visit counts
        pi = np.zeros(4672, dtype=np.float32)
        total_visits = sum(child.visit_count for child in root.children.values())

        if total_visits > 0:
            for action_idx, child in root.children.items():
                pi[action_idx] = child.visit_count / total_visits

        return pi

    def select_action(self, board: chess.Board = None, policy: np.ndarray = None) -> int:
        """
        Select an action based on visit counts and temperature.

        Args:
            board: Current chess board state (optional if policy is provided)
            policy: Pre-computed policy from search() (optional, will call search if not provided)

        Returns:
            action_idx: Selected action index
        """
        if policy is None:
            if board is None:
                raise ValueError("Must provide either board or policy")
            policy = self.search(board)

        # Apply temperature
        if self.temperature == 0.0:
            # Argmax (deterministic)
            action_idx = int(np.argmax(policy))
        else:
            # Sample proportional to visit counts ^ (1 / temperature)
            visit_counts = policy * (np.sum(policy) if np.sum(policy) > 0 else 1.0)
            if self.temperature != 1.0:
                visit_counts = visit_counts ** (1.0 / self.temperature)
                visit_counts = visit_counts / (np.sum(visit_counts) + 1e-8)

            # Sample
            action_idx = np.random.choice(4672, p=visit_counts)

        return action_idx

    def advance_tree(self, action_idx: int, new_board: chess.Board):
        """
        Advance the tree to the child corresponding to the action taken.

        After a move is made, we keep the subtree rooted at that move
        to reuse all the search effort from previous simulations.

        Args:
            action_idx: The action that was taken
            new_board: The board state after the move
        """
        if self.root is not None and action_idx in self.root.children:
            # Advance root to the child node
            self.root = self.root.children[action_idx]
            self.root_board_hash = hash(new_board.fen())

            if self.verbose:
                print(f"[ADVANCE] Moved root to child with {self.root.visit_count} visits")
        else:
            # Move not in tree (shouldn't happen in self-play), start fresh
            self.root = None
            self.root_board_hash = None

            if self.verbose:
                print(f"[ADVANCE] Move not in tree, will create fresh tree next search")

    def reset_tree(self):
        """
        Clear the cached tree. Useful when starting a new game.
        """
        self.root = None
        self.root_board_hash = None

    def _simulate(self, node: MCTSNode, board: chess.Board, sim_num: int, should_log: bool):
        """
        Run a single MCTS simulation from the given node.

        This implements the Selection -> Expansion -> Backup cycle.
        """
        # Track path for backup
        path = []
        depth = 0

        if should_log:
            print("\n[SELECTION PHASE]")
            print(f"Starting from root node")

        # Selection: traverse tree until we reach a leaf
        while node.is_expanded():
            if should_log:
                print(f"\nDepth {depth}: Selecting child using PUCT formula...")

            action_idx, node = self._select_child(node, board, should_log)
            path.append((node, action_idx))

            # Apply move to board
            move = self.converter.action_to_move(action_idx, board)

            if should_log:
                print(f"  → Selected: {move.uci()} ({board.san(move)})")
                print(f"  → Node stats: N={node.visit_count}, W={node.total_value:.3f}, Q={node.value():.3f}, P={node.prior:.4f}")

            board.push(move)
            depth += 1

            # Check if terminal
            if self._is_terminal(board):
                value = self._get_terminal_value(board)
                if should_log:
                    print(f"\n[TERMINAL] Reached terminal state at depth {depth}")
                    print(f"  Terminal value: {value:.3f} (from perspective of side that just moved)")
                self._backup(path, value, should_log)
                return

        # Expansion: if not terminal, expand the node
        if not self._is_terminal(board):
            if should_log:
                print(f"\n[EXPANSION PHASE]")
                print(f"Expanding leaf node at depth {depth}")

            self._expand_node(node, board, should_log)

            # Evaluate the leaf node. The network returns the value from the
            # side-to-move's perspective, but _backup expects it from the
            # perspective of the side that just moved (same convention as
            # _get_terminal_value), so negate it.
            value = -self._evaluate_leaf(board, should_log)

            if should_log:
                print(f"  Backup value (side that just moved): {value:.4f}")
        else:
            value = self._get_terminal_value(board)
            if should_log:
                print(f"\n[TERMINAL] Terminal value: {value:.3f} (from perspective of side that just moved)")

        # Backup: propagate value up the tree
        if should_log:
            print(f"\n[BACKUP PHASE]")
            print(f"Backing up value {value:.4f} through {len(path)} edges")

        self._backup(path, value, should_log)

    def _select_child(self, node: MCTSNode, board: chess.Board, should_log: bool = False) -> tuple:
        """
        Select the best child using the PUCT formula.

        PUCT: Q(s,a) + c_puct * P(s,a) * sqrt(sum N) / (1 + N(s,a))
        """
        total_visits = sum(child.visit_count for child in node.children.values())
        sqrt_total = np.sqrt(total_visits)

        best_score = -float('inf')
        best_action = None
        best_child = None

        # For logging: collect all scores
        if should_log:
            candidates = []

        for action_idx, child in node.children.items():
            q_value = child.value()
            u_value = self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q_value + u_value

            if should_log:
                move = self.converter.action_to_move(action_idx, board)
                candidates.append({
                    'move': move,
                    'action_idx': action_idx,
                    'q': q_value,
                    'u': u_value,
                    'score': score,
                    'prior': child.prior,
                    'visits': child.visit_count,
                })

            if score > best_score:
                best_score = score
                best_action = action_idx
                best_child = child

        if should_log:
            # Sort by score and show top 5
            candidates.sort(key=lambda x: x['score'], reverse=True)
            print(f"  PUCT scores (sqrt_total_visits={sqrt_total:.2f}, c_puct={self.c_puct}):")
            print(f"  {'Move':<8} {'Q':<8} {'U':<8} {'PUCT':<8} {'P':<8} {'N':<6}")
            for i, c in enumerate(candidates[:5]):
                marker = "→" if i == 0 else " "
                print(f"  {marker} {board.san(c['move']):<6} {c['q']:>7.4f} {c['u']:>7.4f} {c['score']:>7.4f} {c['prior']:>7.4f} {c['visits']:>5}")

        return best_action, best_child

    def _expand_node(self, node: MCTSNode, board: chess.Board, should_log: bool = False):
        """
        Expand a node by evaluating the position and creating children for all legal moves.
        """
        # Get legal moves
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 0:
            return

        # Convert to action indices
        legal_actions = []
        for move in legal_moves:
            try:
                action_idx = self.converter.move_to_action(move, board)
                legal_actions.append(action_idx)
            except ValueError:
                # Skip moves that can't be encoded (shouldn't happen)
                continue

        if should_log:
            print(f"  Found {len(legal_moves)} legal moves")

        # Create legal moves mask
        legal_mask = np.zeros(4672, dtype=bool)
        legal_mask[legal_actions] = True

        # Get policy from network
        board_tensor = self.encoder.encode(board)
        policy, _ = self.model.predict(board_tensor, legal_mask)

        # Create children
        for action_idx in legal_actions:
            prior = policy[action_idx]
            node.children[action_idx] = MCTSNode(prior=prior)

        if should_log:
            # Show top 5 moves by prior
            move_priors = [(self.converter.action_to_move(action_idx, board), policy[action_idx])
                          for action_idx in legal_actions]
            move_priors.sort(key=lambda x: x[1], reverse=True)
            print(f"  Top 5 moves by network prior:")
            for i, (move, prior) in enumerate(move_priors[:5], 1):
                print(f"    {i}. {board.san(move)}: P={prior:.4f}")

    def _evaluate_leaf(self, board: chess.Board, should_log: bool = False) -> float:
        """
        Evaluate a leaf node using the neural network.

        Returns value from the perspective of the side to move.
        """
        board_tensor = self.encoder.encode(board)
        _, value = self.model.predict(board_tensor)

        if should_log:
            turn = "White" if board.turn == chess.WHITE else "Black"
            print(f"  Evaluating position ({turn} to move)")
            print(f"  Raw network value: {value:.4f} (positive = good for {turn})")

        return value

    def _backup(self, path: List[tuple], value: float, should_log: bool = False):
        """
        Backup the value through the path, flipping sign at each level.

        Value convention: positive = good for the player who just moved.
        So we negate value as we go up the tree.
        """
        if should_log:
            print(f"  Propagating value up the tree (flipping sign each level):")

        for i, (node, action_idx) in enumerate(reversed(path)):
            node.visit_count += 1
            node.total_value += value

            if should_log:
                depth = len(path) - i - 1
                print(f"    Depth {depth}: N={node.visit_count}, W={node.total_value:.3f}, Q={node.value():.3f}, value={value:.4f}")

            # Flip value for opponent
            value = -value

    def _is_terminal(self, board: chess.Board) -> bool:
        """Check if the board is in a terminal state."""
        return (
            board.is_checkmate()
            or board.is_stalemate()
            or board.is_insufficient_material()
            or len(board.move_stack) >= 200
        )

    def _get_terminal_value(self, board: chess.Board) -> float:
        """
        Get the terminal value from the perspective of the side that just moved.

        Returns:
            +1 if the side that just moved won
            -1 if the side that just moved lost (shouldn't happen in normal play)
            0 for draw
        """
        if board.is_checkmate():
            # Checkmate
            return 1.0
        else:
            # Draw (stalemate, insufficient material, ply cap)
            return 0.0

    def _add_dirichlet_noise(self, root: MCTSNode):
        """
        Add Dirichlet noise to root prior probabilities for exploration.

        P_root(a) = (1 - epsilon) * P_network(a) + epsilon * noise(a)
        """
        actions = list(root.children.keys())
        if len(actions) == 0:
            return

        # Generate Dirichlet noise
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))

        # Mix noise with priors
        for i, action_idx in enumerate(actions):
            child = root.children[action_idx]
            child.prior = (1 - self.dirichlet_epsilon) * child.prior + \
                         self.dirichlet_epsilon * noise[i]
