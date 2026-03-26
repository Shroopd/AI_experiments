from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import tkinter
from tkinter import ttk

MOVES_FIRST_TURN = 1
MOVES_PER_TURN = 2
WIN_LENGTH = 6


class Team(IntEnum):
    TX = 0
    TO = 1

    @classmethod
    def first(cls):
        return cls.TX


class Pos(NamedTuple):
    x: int
    y: int

    def __add__(self, other):
        return Pos(self.x + other.x, self.y + other.y)

    def __mul__(self, other):
        return Pos(self.x * other, self.y * other)

    @classmethod
    def zero(cls):
        return cls(0, 0)


class HexGame:
    def __init__(self) -> None:
        self.board = HexBoard()
        self.ui = HexUI(self.board)


class HexUI:
    def __init__(self, board: HexBoard) -> None:
        self.board = board
        self.root = tkinter.Tk()
        self.mainframe = ttk.Frame(self.root, padding=(3, 3, 3, 3))


class HexBoard:
    directions_to_check = list(
        Pos(x, y)
        for x, y in [
            (1, 0),
            (-1, 0),
            (1, 1),
            (-1, -1),
            (0, 1),
            (0, -1),
        ]
    )

    def __init__(self) -> None:
        self.map: dict[Pos, Team | None] = dict()
        self.turn = 0

        self.map[Pos.zero()] = None

        if MOVES_FIRST_TURN == 1:
            self.map[Pos.zero()] = Team.first()
            self.turn = 1

    def play(self, current_team: Team, moves: set[Pos]) -> Team | None:
        if self.turn == 0 and len(moves) != MOVES_FIRST_TURN:
            raise ValueError("Incorrect number of moves for first turn")
        elif len(moves) != MOVES_PER_TURN:
            raise ValueError("Incorrect number of moves")

        for m in moves:
            if m in self.map and self.map[m] is not None:
                raise ValueError("Invalid move; overwrites existing move")
            else:
                self.map[m] = current_team
        for m in moves:
            if self.check(m):
                return current_team

    def check(self, pos: Pos):
        # raise NotImplementedError

        current_team = self.map[pos]
        assert current_team is not None

        for dpos in self.directions_to_check:
            if self.check_direction(pos, dpos):
                return True
        return False

    def check_direction(self, pos: Pos, direction: Pos):
        current_team = self.map[pos]

        for i in range(1, WIN_LENGTH):
            tmp = pos + (direction * i)
            if tmp not in self.map or self.map[tmp] != current_team:
                return False
            else:
                pass
        return True


def main():
    print("MAIN")


if __name__ == "__main__":
    main()
