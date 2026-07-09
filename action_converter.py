import chess


class ActionConverter:
    """
    Translates between chess.Move objects and integer indices [0, 4671].
    Manages the '73 move types' logic.

    Action space: 64 squares × 73 move types = 4672 actions

    Move types (73 total):
    - Queen moves (56): 8 directions × 7 distances
    - Knight moves (8): 8 L-shaped moves
    - Underpromotions (9): 3 pieces (R, B, N) × 3 directions
    """

    def __init__(self):
        # On init, generate the bidirectional lookup tables
        self.move_to_idx_map = {}  # Key: (from_sq, to_sq, promotion), Value: action_idx
        self.idx_to_move_map = {}  # Key: action_idx, Value: (from_sq, to_sq, promotion)
        self._generate_mappings()

    def _generate_mappings(self):
        """
        Iterates 0..63 (source squares).
        Iterates 0..72 (move types).
        Calculates target squares/promotion types based on logic.
        Populates the internal lookup dicts.
        """
        # Direction vectors for queen moves (8 directions)
        queen_directions = [
            (-1, 0),  # N
            (1, 0),   # S
            (0, 1),   # E
            (0, -1),  # W
            (-1, 1),  # NE
            (-1, -1), # NW
            (1, 1),   # SE
            (1, -1),  # SW
        ]

        # Knight move offsets (8 L-shapes)
        knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]

        # Underpromotion: 3 directions × 3 piece types
        # Directions: forward, capture-left, capture-right
        underpromotion_pieces = [chess.ROOK, chess.BISHOP, chess.KNIGHT]

        move_type_idx = 0

        # Queen moves: 8 directions × 7 distances = 56 move types
        self.queen_move_start = move_type_idx
        for direction in queen_directions:
            for distance in range(1, 8):
                for from_sq in range(64):
                    from_rank = from_sq // 8
                    from_file = from_sq % 8

                    to_rank = from_rank + direction[0] * distance
                    to_file = from_file + direction[1] * distance

                    # Check if target is on board
                    if 0 <= to_rank < 8 and 0 <= to_file < 8:
                        to_sq = to_rank * 8 + to_file
                        action_idx = from_sq * 73 + move_type_idx

                        self.move_to_idx_map[(from_sq, to_sq, None)] = action_idx
                        self.idx_to_move_map[action_idx] = (from_sq, to_sq, None)

                move_type_idx += 1

        # Knight moves: 8 move types
        self.knight_move_start = move_type_idx
        for knight_offset in knight_moves:
            for from_sq in range(64):
                from_rank = from_sq // 8
                from_file = from_sq % 8

                to_rank = from_rank + knight_offset[0]
                to_file = from_file + knight_offset[1]

                # Check if target is on board
                if 0 <= to_rank < 8 and 0 <= to_file < 8:
                    to_sq = to_rank * 8 + to_file
                    action_idx = from_sq * 73 + move_type_idx

                    self.move_to_idx_map[(from_sq, to_sq, None)] = action_idx
                    self.idx_to_move_map[action_idx] = (from_sq, to_sq, None)

            move_type_idx += 1

        # Underpromotions: 3 pieces × 3 directions = 9 move types
        self.underpromotion_start = move_type_idx
        for promotion_piece in underpromotion_pieces:
            # Forward, capture-left, capture-right
            for delta_file in [0, -1, 1]:
                for from_sq in range(64):
                    from_rank = from_sq // 8
                    from_file = from_sq % 8

                    # White pawns promote from rank 6 to 7
                    if from_rank == 6:
                        to_rank = 7
                        to_file = from_file + delta_file
                        if 0 <= to_file < 8:
                            to_sq = to_rank * 8 + to_file
                            action_idx = from_sq * 73 + move_type_idx

                            self.move_to_idx_map[(from_sq, to_sq, promotion_piece)] = action_idx
                            self.idx_to_move_map[action_idx] = (from_sq, to_sq, promotion_piece)

                    # Black pawns promote from rank 1 to 0
                    if from_rank == 1:
                        to_rank = 0
                        to_file = from_file + delta_file
                        if 0 <= to_file < 8:
                            to_sq = to_rank * 8 + to_file
                            action_idx = from_sq * 73 + move_type_idx

                            self.move_to_idx_map[(from_sq, to_sq, promotion_piece)] = action_idx
                            self.idx_to_move_map[action_idx] = (from_sq, to_sq, promotion_piece)

                move_type_idx += 1

    def move_to_action(self, move: chess.Move, board: chess.Board) -> int:
        """
        Converts a chess.Move to the integer index.

        Logic:
        1. Identify source square (0-63).
        2. Identify move type (0-72) based on diff(rank), diff(file).
           - Handle sliding (Queen/Rook/Bishop) logic.
           - Handle Knight logic.
           - Handle Underpromotion logic (N, R, B).
        3. Return source * 73 + move_type.
        """
        from_sq = move.from_square
        to_sq = move.to_square
        promotion = move.promotion

        # Handle underpromotions (not queen)
        if promotion is not None and promotion != chess.QUEEN:
            key = (from_sq, to_sq, promotion)
        else:
            # Queen promotions are handled as normal queen moves
            key = (from_sq, to_sq, None)

        if key not in self.move_to_idx_map:
            raise ValueError(f"Move {move} not found in action space mapping")

        return self.move_to_idx_map[key]

    def action_to_move(self, action_idx: int, board: chess.Board) -> chess.Move:
        """
        Converts an integer index back to a chess.Move.

        Logic:
        1. source_sq = action_idx // 73
        2. move_type = action_idx % 73
        3. Retrieve delta/promotion info from pre-computed map.
        4. Calculate target square.
        5. Construct chess.Move(source, target, promotion).
        """
        if action_idx not in self.idx_to_move_map:
            raise ValueError(f"Action index {action_idx} not found in mapping")

        from_sq, to_sq, promotion = self.idx_to_move_map[action_idx]

        # Handle queen promotions: if moving to back rank with a pawn, it's a queen promotion
        if promotion is None:
            piece = board.piece_at(from_sq)
            if piece is not None and piece.piece_type == chess.PAWN:
                to_rank = to_sq // 8
                # White pawn reaching rank 7 or Black pawn reaching rank 0
                if (piece.color == chess.WHITE and to_rank == 7) or \
                   (piece.color == chess.BLACK and to_rank == 0):
                    promotion = chess.QUEEN

        return chess.Move(from_sq, to_sq, promotion)
