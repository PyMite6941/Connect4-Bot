"""Run with:  python -m unittest discover tests"""
import unittest

from clyde.search import (COLS, DECAY, ROWS, WIN, drop, flip, is_win, legal_moves, move_values,
                          search_value, to_relative)

EMPTY = (0,) * (ROWS * COLS)
zero = lambda boards: [0.0] * len(boards)  # dummy evaluator: every leaf looks even


def place(cells):
    """{(row, col): value} -> relative board. Row 5 is the bottom."""
    b = list(EMPTY)
    for (r, c), v in cells.items():
        b[r * COLS + c] = v
    return tuple(b)


class TestBoardHelpers(unittest.TestCase):
    def test_drop_lands_on_bottom_then_stacks(self):
        b, r = drop(EMPTY, 3)
        self.assertEqual(r, 5)
        _, r2 = drop(b, 3)
        self.assertEqual(r2, 4)

    def test_is_win_all_directions(self):
        cases = {
            "horizontal": {(5, 0): 1, (5, 1): 1, (5, 2): 1, (5, 3): 1},
            "vertical": {(5, 0): 1, (4, 0): 1, (3, 0): 1, (2, 0): 1},
            "diagonal /": {(5, 0): 1, (4, 1): 1, (3, 2): 1, (2, 3): 1},
            "diagonal \\": {(2, 0): 1, (3, 1): 1, (4, 2): 1, (5, 3): 1},
        }
        for name, cells in cases.items():
            with self.subTest(name):
                last = max(cells)  # any of the four works
                self.assertTrue(is_win(place(cells), *last))

    def test_three_is_not_a_win(self):
        self.assertFalse(is_win(place({(5, 0): 1, (5, 1): 1, (5, 2): 1}), 5, 2))

    def test_to_relative_and_flip(self):
        state = ("Clyde",) + ("smartBot",) + (0,) * 40
        self.assertEqual(to_relative(state, "Clyde")[:3], (1, -1, 0))
        self.assertEqual(flip(to_relative(state, "Clyde"))[:3], (-1, 1, 0))

    def test_full_column_not_legal(self):
        b = place({(r, 2): 1 for r in range(ROWS)})
        self.assertNotIn(2, legal_moves(b))


class TestSearch(unittest.TestCase):
    def test_takes_win_in_one(self):
        b = place({(5, 0): 1, (4, 0): 1, (3, 0): 1, (5, 1): -1, (5, 2): -1})
        values = move_values(b, 1, zero)
        self.assertEqual(values[0], WIN)
        self.assertEqual(max(values, key=values.get), 0)

    def test_blocks_loss_in_one_at_depth_2(self):
        # Opponent has three stacked in column 0; every move except 0 lets them win.
        b = place({(5, 0): -1, (4, 0): -1, (3, 0): -1, (5, 1): 1, (5, 2): 1})
        values = move_values(b, 2, zero)
        self.assertEqual(max(values, key=values.get), 0)
        for col, v in values.items():
            if col != 0:
                self.assertAlmostEqual(v, -DECAY * WIN)  # opponent's win inside the subtree, not ours

    def test_depth_1_cannot_see_the_threat(self):
        # Same position: a 1-ply search with a blank evaluator has no reason to prefer column 0.
        # This is exactly the gap the trained value head has to fill.
        b = place({(5, 0): -1, (4, 0): -1, (3, 0): -1, (5, 1): 1, (5, 2): 1})
        self.assertEqual(len(set(move_values(b, 1, zero).values())), 1)

    def test_search_value_matches_best_move_value(self):
        b = place({(5, 0): -1, (4, 0): -1, (3, 0): -1, (5, 1): 1, (5, 2): 1})
        self.assertAlmostEqual(search_value(b, 2, zero), max(move_values(b, 2, zero).values()))

    def test_full_board_is_a_draw(self):
        full = tuple((1 if (i // COLS + i % COLS // 2) % 2 else -1) for i in range(ROWS * COLS))
        self.assertEqual(search_value(full, 3, zero), 0.0)

    def test_evaluator_called_once_per_search(self):
        calls = []

        def spy(boards):
            calls.append(len(boards))
            return [0.0] * len(boards)

        search_value(EMPTY, 3, spy)
        self.assertEqual(calls, [7 ** 3])  # all leaves batched into a single call


if __name__ == "__main__":
    unittest.main()
