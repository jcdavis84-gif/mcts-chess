import chess
import numpy as np
import torch
import torch.multiprocessing as mp
from typing import List, Tuple, Optional, Dict
from board_encoder import BoardEncoder, calculate_material_balance
from action_converter import ActionConverter
from mcts import MCTS


class SelfPlayGame:
    """
    Manages a single self-play game from start to finish.

    Stores positions and MCTS policies, then backfills outcomes when game ends.
    """

    def __init__(
        self,
        model,
        encoder: BoardEncoder,
        converter: ActionConverter,
        num_simulations: int = 20,
        temperature_threshold: int = 30,
        discard_draws: bool = False,
        last_n_plies: int = None,
    ):
        self.model = model
        self.encoder = encoder
        self.converter = converter
        self.num_simulations = num_simulations
        self.temperature_threshold = temperature_threshold
        self.discard_draws = discard_draws
        self.last_n_plies = last_n_plies

        # Game history: list of (board_tensor, policy, side_to_move, material_balance)
        self.history: List[Tuple[np.ndarray, np.ndarray, bool, float]] = []

    def play_game(self, starting_fen: str = None) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float, float]], float, int]:
        """
        Play a full self-play game.

        Args:
            starting_fen: Optional FEN string for starting position (default: standard chess start)

        Returns:
            Tuple of:
                - List of (board_tensor, policy, outcome, material) tuples for training (after filtering)
                - Game outcome from White's perspective (+1.0, 0.0, -1.0)
                - Original game length in plies (before filtering)
        """
        board = chess.Board(starting_fen) if starting_fen else chess.Board()
        self.history = []

        move_count = 0

        # Create MCTS instance once per game for tree reuse
        mcts = MCTS(
            model=self.model,
            encoder=self.encoder,
            converter=self.converter,
            num_simulations=self.num_simulations,
            c_puct=1.5,
            temperature=1.0,  # Will update per move
            add_dirichlet_noise=True,
        )

        while not self._is_terminal(board):
            # Determine temperature based on move count
            temperature = 1.0 if move_count < self.temperature_threshold else 0.0
            mcts.temperature = temperature

            # Get MCTS policy (tree is reused automatically)
            policy = mcts.search(board)

            # Store position, policy, and material balance
            board_tensor = self.encoder.encode(board)
            side_to_move = board.turn  # chess.WHITE or chess.BLACK
            material_balance = calculate_material_balance(board)
            self.history.append((board_tensor, policy, side_to_move, material_balance))

            # Select and execute move (using the policy we just computed)
            action_idx = mcts.select_action(policy=policy)
            move = self.converter.action_to_move(action_idx, board)
            board.push(move)

            # Advance tree to the new position
            mcts.advance_tree(action_idx, board)

            move_count += 1

        # Game is over - determine outcome
        outcome = self._get_game_outcome(board)
        original_length = len(self.history)

        # Filter: discard drawn games if requested
        if self.discard_draws and outcome == 0:
            return [], outcome, original_length

        # Backfill outcomes from each player's perspective
        training_data = []
        for board_tensor, policy, side_to_move, material_balance in self.history:
            # Outcome is from white's perspective; flip for black
            if side_to_move == chess.WHITE:
                value = outcome
            else:
                value = -outcome

            # Material balance is absolute (does NOT flip based on side to move)
            training_data.append((board_tensor, policy, value, material_balance))

        # Filter: keep only last N plies if requested
        if self.last_n_plies is not None and len(training_data) > self.last_n_plies:
            training_data = training_data[-self.last_n_plies:]

        return training_data, outcome, original_length

    def _is_terminal(self, board: chess.Board) -> bool:
        """Check if the board is in a terminal state."""
        return (
            board.is_checkmate()
            or board.is_stalemate()
            or board.is_insufficient_material()
            or len(board.move_stack) >= 200
        )

    def _get_game_outcome(self, board: chess.Board) -> float:
        """
        Get game outcome from White's perspective.

        Returns:
            +1.0 if White won
            -1.0 if Black won
            0.0 for draw
        """
        if board.is_checkmate():
            # Winner is the side that just moved (opposite of board.turn)
            if board.turn == chess.BLACK:
                return 1.0  # White won
            else:
                return -1.0  # Black won
        else:
            # Draw
            return 0.0


def _estimate_safe_workers(model, num_workers: Optional[int], device: str) -> int:
    """
    Estimate safe number of workers based on model size and available VRAM.

    Args:
        model: Neural network model
        num_workers: Requested number of workers (None = auto-detect)
        device: Device string ("cuda" or "cpu")

    Returns:
        Safe number of workers to use
    """
    # If CPU or single worker requested, no multiprocessing
    if device == "cpu" or num_workers == 1:
        return 1

    # If CUDA not available, use 1 worker
    if not torch.cuda.is_available():
        return 1

    # Get GPU VRAM
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    # Calculate model size
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

    # Conservative estimate:
    # - 1000MB PyTorch/CUDA base overhead
    # - model_size_mb for training model (stays in main process)
    # - (model_size_mb + 150MB) per worker (model copy + MCTS cache)
    reserved_mb = 1000 + model_size_mb
    available_mb = (total_vram_gb * 1024) - reserved_mb
    per_worker_mb = model_size_mb + 150

    max_safe_workers = max(1, int(available_mb / per_worker_mb))

    # Use requested num_workers or auto-detected, whichever is smaller
    if num_workers is None:
        # Auto-detect: cap at 8 for coordination overhead
        return min(max_safe_workers, 8)
    else:
        # User specified: use their value but warn if unsafe
        if num_workers > max_safe_workers:
            print(f"Warning: Requested {num_workers} workers but only {max_safe_workers} estimated safe")
            print(f"  Model size: {model_size_mb:.1f}MB, Available VRAM: {available_mb:.1f}MB")
            print(f"  If you encounter CUDA OOM, try --num-workers {max_safe_workers}")
        return min(num_workers, max_safe_workers)


def _play_single_game_worker(args):
    """
    Worker function for parallel self-play.

    Args:
        args: Tuple of (model_config, encoder, converter, num_simulations, discard_draws, last_n_plies, device)

    Returns:
        Tuple of:
            - training_data: List of (board_tensor, policy, value, material) tuples
            - outcome: Game outcome from White's perspective
            - original_length: Original game length in plies
    """
    model_config, encoder, converter, num_simulations, discard_draws, last_n_plies, device = args

    try:
        # Import here to avoid issues with multiprocessing
        from model import AlphaZeroNetwork

        # Recreate model in this process
        model = AlphaZeroNetwork(
            d_model=model_config['d_model'],
            num_blocks=model_config['num_blocks'],
            num_heads=model_config['num_heads'],
            d_policy=model_config['d_policy'],
        )
        model.load_state_dict(model_config['state_dict'])
        model.to(device)
        model.eval()

        # Play game
        game = SelfPlayGame(
            model=model,
            encoder=encoder,
            converter=converter,
            num_simulations=num_simulations,
            discard_draws=discard_draws,
            last_n_plies=last_n_plies,
        )

        return game.play_game()

    except torch.cuda.OutOfMemoryError as e:
        print(f"\nCUDA Out of Memory in worker process!")
        print(f"Try reducing --num-workers or using a smaller model.")
        raise
    except Exception as e:
        print(f"\nError in worker process: {e}")
        raise


def play_games(
    model,
    encoder: BoardEncoder,
    converter: ActionConverter,
    num_games: int,
    num_simulations: int = 20,
    discard_draws: bool = False,
    last_n_plies: int = None,
    num_workers: Optional[int] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float, float]], Dict[str, float]]:
    """
    Play multiple self-play games and collect training data.

    Args:
        model: Neural network for evaluation
        encoder: Board encoder
        converter: Action converter
        num_games: Number of games to play
        num_simulations: MCTS simulations per move
        discard_draws: If True, discard all positions from drawn games
        last_n_plies: If set, only keep last N plies from each game
        num_workers: Number of parallel workers (None = auto-detect, 1 = sequential)

    Returns:
        Tuple of:
            - List of (board_tensor, policy, outcome, material) tuples (after filtering)
            - Dict with pre-filter game statistics (outcomes, lengths)
    """
    # Get device from model
    device = str(next(model.parameters()).device)

    # Estimate safe number of workers
    actual_workers = _estimate_safe_workers(model, num_workers, device)

    # If single worker, use sequential path (no multiprocessing overhead)
    if actual_workers == 1:
        all_data = []
        outcomes = []
        game_lengths = []

        for game_idx in range(num_games):
            game = SelfPlayGame(
                model=model,
                encoder=encoder,
                converter=converter,
                num_simulations=num_simulations,
                discard_draws=discard_draws,
                last_n_plies=last_n_plies,
            )
            game_data, outcome, original_length = game.play_game()
            all_data.extend(game_data)
            outcomes.append(outcome)
            game_lengths.append(original_length)

        # Compute statistics from original game data (before filtering)
        stats = {
            'white_wins': sum(1 for v in outcomes if v > 0.5),
            'draws': sum(1 for v in outcomes if -0.5 < v < 0.5),
            'black_wins': sum(1 for v in outcomes if v < -0.5),
            'avg_game_length': sum(game_lengths) / len(game_lengths) if game_lengths else 0,
        }

        return all_data, stats

    # Multi-worker path: use multiprocessing with spawn
    # Prepare model config for workers (pickleable)
    model_config = {
        'state_dict': model.state_dict(),
        'd_model': model.d_model,
        'num_blocks': model.num_blocks,
        'num_heads': model.num_heads,
        'd_policy': model.d_policy,
    }

    # Create argument list for each game
    args_list = [
        (model_config, encoder, converter, num_simulations, discard_draws, last_n_plies, device)
        for _ in range(num_games)
    ]

    # Run games in parallel using spawn context (required for CUDA)
    try:
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=actual_workers) as pool:
            results = pool.map(_play_single_game_worker, args_list)

        # Flatten results and collect statistics
        all_data = []
        outcomes = []
        game_lengths = []

        for game_data, outcome, original_length in results:
            all_data.extend(game_data)
            outcomes.append(outcome)
            game_lengths.append(original_length)

        # Compute statistics from original game data (before filtering)
        stats = {
            'white_wins': sum(1 for v in outcomes if v > 0.5),
            'draws': sum(1 for v in outcomes if -0.5 < v < 0.5),
            'black_wins': sum(1 for v in outcomes if v < -0.5),
            'avg_game_length': sum(game_lengths) / len(game_lengths) if game_lengths else 0,
        }

        return all_data, stats

    except Exception as e:
        print(f"\nMultiprocessing failed: {e}")
        print("Falling back to sequential execution...")

        # Fallback to sequential
        all_data = []
        outcomes = []
        game_lengths = []

        for game_idx in range(num_games):
            game = SelfPlayGame(
                model=model,
                encoder=encoder,
                converter=converter,
                num_simulations=num_simulations,
                discard_draws=discard_draws,
                last_n_plies=last_n_plies,
            )
            game_data, outcome, original_length = game.play_game()
            all_data.extend(game_data)
            outcomes.append(outcome)
            game_lengths.append(original_length)

        # Compute statistics from original game data (before filtering)
        stats = {
            'white_wins': sum(1 for v in outcomes if v > 0.5),
            'draws': sum(1 for v in outcomes if -0.5 < v < 0.5),
            'black_wins': sum(1 for v in outcomes if v < -0.5),
            'avg_game_length': sum(game_lengths) / len(game_lengths) if game_lengths else 0,
        }

        return all_data, stats
