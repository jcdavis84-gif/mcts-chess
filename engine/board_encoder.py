import chess
import numpy as np


def calculate_material_balance(board: chess.Board) -> float:
    """
    Calculate material balance of a chess position.

    Args:
        board: chess.Board object

    Returns:
        Material balance from White's perspective (positive = White advantage)
        Standard piece values: P=1, N=3, B=3, R=5, Q=9, K=0
    """
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,  # King doesn't count toward material
    }

    balance = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            value = piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                balance += value
            else:
                balance -= value

    return float(balance)


class BoardEncoder:
    """
    Responsible for converting a chess.Board object into the
    AlphaZero-style 18x8x8 input tensor.
    """

    def __init__(self):
        # Define plane indices for readability
        self.PAWN, self.ROOK, self.KNIGHT, self.BISHOP, self.QUEEN, self.KING = 0, 1, 2, 3, 4, 5

        # Mapping from chess piece types to plane indices
        self.piece_to_plane = {
            chess.PAWN: self.PAWN,
            chess.ROOK: self.ROOK,
            chess.KNIGHT: self.KNIGHT,
            chess.BISHOP: self.BISHOP,
            chess.QUEEN: self.QUEEN,
            chess.KING: self.KING,
        }

    def encode(self, board: chess.Board) -> np.ndarray:
        """
        Input: A python-chess board object.
        Output: A (18, 8, 8) numpy array (float32).

        Logic:
        1. Initialize zeros(18, 8, 8).
        2. Iterate through piece types (P, R, N, B, Q, K):
           - Fill planes 0-5 for White pieces.
           - Fill planes 6-11 for Black pieces.
        3. Fill plane 12 based on board.turn (1.0 if White, 0.0 if Black).
        4. Fill planes 13-16 based on board.has_kingside_castling_rights(color)
           and queenside rights.
        5. Fill plane 17 based on board.ep_square (1.0 at that square if not None).
        6. Return generated tensor.
        """
        # Initialize the tensor
        planes = np.zeros((18, 8, 8), dtype=np.float32)

        # Planes 0-11: Piece positions
        # Planes 0-5: White pieces (P, R, N, B, Q, K)
        # Planes 6-11: Black pieces (P, R, N, B, Q, K)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                rank = chess.square_rank(square)
                file = chess.square_file(square)

                plane_idx = self.piece_to_plane[piece.piece_type]
                if piece.color == chess.BLACK:
                    plane_idx += 6  # Offset for black pieces

                planes[plane_idx, rank, file] = 1.0

        # Plane 12: Side to move (1.0 if White's turn, 0.0 if Black's turn)
        if board.turn == chess.WHITE:
            planes[12, :, :] = 1.0

        # Planes 13-16: Castling rights
        # Plane 13: White Kingside
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[13, :, :] = 1.0

        # Plane 14: White Queenside
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[14, :, :] = 1.0

        # Plane 15: Black Kingside
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[15, :, :] = 1.0

        # Plane 16: Black Queenside
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[16, :, :] = 1.0

        # Plane 17: En passant square
        if board.ep_square is not None:
            rank = chess.square_rank(board.ep_square)
            file = chess.square_file(board.ep_square)
            planes[17, rank, file] = 1.0

        return planes
