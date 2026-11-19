from __future__ import annotations

import json
from pathlib import Path

CONFIGS = ((30, 3, 3, 303), (50, 5, 5, 505), (100, 8, 8, 1008))


def validate_board(size: int, snakes: dict[int, int], ladders: dict[int, int]) -> None:
    if size < 7:
        raise ValueError("Board must contain at least 7 squares")
    starts = set(snakes) | set(ladders)
    if set(snakes) & set(ladders):
        raise ValueError("Snake and ladder starts must be distinct")
    if 1 in starts or size in starts:
        raise ValueError("Start and goal cannot be transition starts")
    if any(not (1 < a < size and 1 <= b < a) for a, b in snakes.items()):
        raise ValueError("Each snake must lead backward to a legal square")
    if any(not (1 < a < size and a < b <= size) for a, b in ladders.items()):
        raise ValueError("Each ladder must lead forward to a legal square")
    if any(b in starts for b in (*snakes.values(), *ladders.values())):
        raise ValueError("Chained transitions are not supported")


def generate_board(size: int, n_snakes: int, n_ladders: int, seed: int) -> dict:
    import numpy as np
    rng = np.random.default_rng(seed)
    possible_snake_starts = np.arange(max(3, n_ladders + 3), size)
    possible_ladder_starts = np.arange(2, size)
    rng.shuffle(possible_snake_starts); rng.shuffle(possible_ladder_starts)
    snake_starts = [int(x) for x in possible_snake_starts[:n_snakes]]
    ladder_starts = [int(x) for x in possible_ladder_starts if x not in snake_starts][:n_ladders]
    snakes: dict[int, int] = {s: -1 for s in snake_starts}
    ladders: dict[int, int] = {s: -1 for s in ladder_starts}
    starts = set(snakes) | set(ladders)
    snakes = {s: int(rng.choice([d for d in range(2, s) if d not in starts])) for s in snakes}
    ladders = {s: int(rng.choice([d for d in range(s + 1, size + 1) if d not in starts])) for s in ladders}
    validate_board(size, snakes, ladders)
    return {"size": size, "snakes": dict(sorted(snakes.items())), "ladders": dict(sorted(ladders.items())), "seed": seed}


def save_required_boards(directory: Path) -> list[dict]:
    directory.mkdir(parents=True, exist_ok=True)
    boards = [generate_board(*cfg) for cfg in CONFIGS]
    for i, board in enumerate(boards, 1):
        (directory / f"board_{i}_{board['size']}.json").write_text(json.dumps(board, indent=2), encoding="utf-8")
    return boards

