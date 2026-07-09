#!/usr/bin/env python3
"""
Play against a model checkpoint in the browser.

Serves a chessboard.js UI (play.html) and answers move requests by running
MCTS with the loaded checkpoint. Uses only the Python standard library for
the server — no new dependencies.

Usage:
    python play.py --checkpoint checkpoints/model_final.pt
    python play.py --checkpoint model_latest.pt --color black --simulations 200
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import chess
import numpy as np
import torch

from action_converter import ActionConverter
from board_encoder import BoardEncoder
from mcts import MCTS
from model import AlphaZeroNetwork


def load_model(checkpoint_path: str, device: str) -> AlphaZeroNetwork:
    """
    Load a checkpoint, inferring architecture from metadata keys when present
    and from state-dict shapes otherwise (handles both train.py and DPO saves).
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception:
        # torch >= 2.6 defaults to weights_only=True, which rejects some
        # older checkpoints; these are our own files, so fall back.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        meta = checkpoint
    elif isinstance(checkpoint, dict) and "token_proj.weight" in checkpoint:
        state_dict = checkpoint
        meta = {}
    else:
        raise ValueError(f"Unrecognized checkpoint format: {checkpoint_path}")

    block_indices = [int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")]
    model = AlphaZeroNetwork(
        d_model=meta.get("d_model", state_dict["token_proj.weight"].shape[0]),
        num_blocks=meta.get("num_blocks", max(block_indices) + 1 if block_indices else 4),
        num_heads=meta.get("num_heads", 2),
        d_policy=meta.get("d_policy", state_dict["policy_proj.weight"].shape[0]),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    total_steps = meta.get("total_steps", "unknown")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded: {checkpoint_path}")
    print(f"  Parameters: {num_params:,} | trained steps: {total_steps} | device: {device}")
    return model


class Engine:
    """Thread-safe wrapper: replay a move history, pick the model's reply."""

    def __init__(self, model, num_simulations: int, temperature: float):
        self.model = model
        self.encoder = BoardEncoder()
        self.converter = ActionConverter()
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.mcts = MCTS(
            model=model,
            encoder=self.encoder,
            converter=self.converter,
            num_simulations=num_simulations,
            temperature=temperature,
            add_dirichlet_noise=False,
        )
        self.lock = threading.Lock()

    def choose_move(self, moves: list) -> dict:
        with self.lock:
            board = chess.Board()
            for uci in moves:
                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves:
                    raise ValueError(f"Illegal move in history: {uci}")
                board.push(move)

            if board.is_game_over():
                raise ValueError("Game is already over")

            start = time.time()
            self.mcts.reset_tree()
            pi = self.mcts.search(board)
            action_idx = self.mcts.select_action(policy=pi)
            move = self.converter.action_to_move(action_idx, board)
            san = board.san(move)
            elapsed_ms = int((time.time() - start) * 1000)

            # Evaluate the root position for the eval panel. Value is from the
            # side-to-move's perspective; material is White - Black.
            board_tensor = torch.from_numpy(self.encoder.encode(board)).float()
            device = next(self.model.parameters()).device
            with torch.no_grad():
                _, value, material = self.model(board_tensor.unsqueeze(0).to(device))
            value = value.item()
            value_white = value if board.turn == chess.WHITE else -value

            side = "White" if board.turn == chess.WHITE else "Black"
            print(
                f"[move {board.fullmove_number}] {side} (model): {san} "
                f"({move.uci()}) | value={value:+.3f} | "
                f"material={material.item():+.2f} | {elapsed_ms}ms",
                flush=True,
            )

            return {
                "move": move.uci(),
                "san": san,
                "value": round(value, 4),
                "value_white": round(value_white, 4),
                "material": round(material.item(), 2),
                "simulations": self.num_simulations,
                "elapsed_ms": elapsed_ms,
            }


def make_handler(engine: Engine, config: dict):
    html_path = Path(__file__).parent / "play.html"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # engine prints its own, more useful log lines

        def _send_json(self, payload: dict, status: int = 200):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = html_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/config":
                self._send_json(config)
            else:
                self._send_json({"error": "Not found"}, status=404)

        def do_POST(self):
            if self.path != "/api/move":
                self._send_json({"error": "Not found"}, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                result = engine.choose_move(payload.get("moves", []))
                self._send_json(result)
            except (ValueError, KeyError) as e:
                self._send_json({"error": str(e)}, status=400)
            except Exception as e:
                print(f"[error] {type(e).__name__}: {e}")
                self._send_json({"error": f"{type(e).__name__}: {e}"}, status=500)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Play against a model checkpoint in the browser")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--color", type=str, default="white", choices=["white", "black"],
                        help="Your color (model plays the other side)")
    parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Move selection temperature (0 = strongest, deterministic)")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/mps/cpu/auto)")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open the browser")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    np.random.seed()  # temperature > 0 samples moves; don't repeat games

    model = load_model(args.checkpoint, device)
    engine = Engine(model, args.simulations, args.temperature)
    config = {
        "checkpoint": args.checkpoint,
        "user_color": args.color,
        "simulations": args.simulations,
        "temperature": args.temperature,
    }

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine, config))
    url = f"http://127.0.0.1:{args.port}"
    print(f"✓ Serving on {url} — you play {args.color}, model plays "
          f"{'black' if args.color == 'white' else 'white'}")
    print("  Press Ctrl+C to stop.", flush=True)

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
