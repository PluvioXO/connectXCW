"""Verify the frozen oracle using an independent column-grid implementation."""

import argparse
from collections import Counter
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path


def place(grid, column, player):
    result = list(grid)
    result[column] = result[column] + (player,)
    return tuple(result)


def alignment(grid, column, row, player):
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        contiguous = 1
        for sign in (-1, 1):
            x, y = column + sign * dx, row + sign * dy
            while 0 <= x < 7 and 0 <= y < len(grid[x]) and grid[x][y] == player:
                contiguous += 1
                x, y = x + sign * dx, y + sign * dy
        if contiguous >= 4:
            return True
    return False


def ending_labels(prefix):
    grid = ((),) * 7
    player = 1
    for column in prefix:
        assert len(grid[column]) < 6
        grid = place(grid, column, player)
        assert not alignment(grid, column, len(grid[column]) - 1, player)
        player = 3 - player

    @lru_cache(maxsize=None)
    def value(position, mover):
        best = -1
        for column in range(7):
            if len(position[column]) == 6:
                continue
            outcome = after_move(position, column, mover)
            if outcome == 1:
                return outcome
            best = max(best, outcome)
        return best

    def after_move(position, column, mover):
        child = place(position, column, mover)
        if alignment(child, column, len(child[column]) - 1, mover):
            return 1
        if all(len(stack) == 6 for stack in child):
            return 0
        return -value(child, 3 - mover)

    return [after_move(grid, c, player) if len(grid[c]) < 6 else None for c in range(7)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("testpack", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    original = args.testpack.read_bytes()
    rows = json.loads(original)["oracle"]
    counts = Counter()
    for row in rows:
        assert ending_labels(row["prefix"]) == row["values_by_column"]
        counts[row["group"]] += 1
    result = {"testpack_sha256": sha256(original).hexdigest(),
              "positions_verified": len(rows), "verified_by_group": dict(counts),
              "algorithm": "Independent column-grid exhaustive win/draw/loss recursion"}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
