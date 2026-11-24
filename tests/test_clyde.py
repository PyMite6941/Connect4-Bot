"""Run with:  python -m unittest discover tests"""
import os
import tempfile
import unittest

import torch

from arena.match import Play
from clyde.connect4 import Game
from clyde.dqn import COLS, DQNAgent, encode_state, mirror_state, valid_mask


def board_from_moves(moves, p1="Clyde", p2="smartBot"):
    """Play a list of columns alternately, p1 first."""
    g = Game(p1, p2)
    for i, col in enumerate(moves):
        g.make_move(col, (p1, p2)[i % 2])
    return g


class TestGame(unittest.TestCase):
    def test_vertical_win(self):
        g = board_from_moves([0, 1, 0, 1, 0, 1, 0])
        self.assertEqual(g.is_terminal("Clyde"), {"end": True, "winner": "Clyde"})

    def test_horizontal_win(self):
        g = board_from_moves([0, 0, 1, 1, 2, 2, 3])
        self.assertTrue(g.check_win("Clyde")["success"])

    def test_diagonal_win(self):
        g = board_from_moves([0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3])
        self.assertTrue(g.check_win("Clyde")["success"])

    def test_no_false_win_for_opponent(self):
        g = board_from_moves([0, 1, 0, 1, 0, 1, 0])
        self.assertFalse(g.check_win("smartBot")["success"])

    def test_full_column_not_a_move(self):
        g = board_from_moves([3] * 6)
        self.assertNotIn(3, g.possible_moves())


class TestRewards(unittest.TestCase):
    def test_final_reward_perspective(self):
        play = Play("Clyde", "smartBot", log=False)
        self.assertEqual(play.final_reward("Clyde"), 1.0)
        self.assertEqual(play.final_reward("smartBot"), -1.0)
        self.assertEqual(play.final_reward("Draw"), 0.5)


class TestEncoding(unittest.TestCase):
    def test_encode_is_perspective_relative(self):
        state = board_from_moves([3, 4]).get_state()
        x = encode_state(state, "Clyde")
        self.assertEqual(x.shape, (2, 6, 7))
        self.assertEqual(x[0, 5, 3].item(), 1.0)  # Clyde's piece, bottom row, column 3
        self.assertEqual(x[1, 5, 4].item(), 1.0)  # opponent's piece
        self.assertEqual(x.sum().item(), 2.0)

    def test_mirror_is_involution_and_flips_columns(self):
        state = board_from_moves([0, 1, 1]).get_state()
        m = mirror_state(state)
        self.assertEqual(mirror_state(m), state)
        self.assertEqual(m[5 * COLS + 6], "Clyde")  # column 0 -> column 6

    def test_valid_mask(self):
        self.assertEqual(valid_mask([0, 6]).tolist(), [True, False, False, False, False, False, True])


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.agent = DQNAgent(path=os.path.join(self.tmp, "t.pt"))

    def test_never_picks_full_column(self):
        state = board_from_moves([]).get_state()
        for _ in range(50):
            self.assertIn(self.agent.choose_moves(state, [2, 5], training=False), [2, 5])
            self.assertIn(self.agent.choose_moves(state, [2, 5], training=True), [2, 5])

    def test_update_stores_mirrored_pair(self):
        state = board_from_moves([3]).get_state()
        nxt = board_from_moves([3, 0, 1]).get_state()
        self.agent.update_Q(state, 1, 0.01, nxt, list(range(7)))
        self.agent.update_Q(nxt, 2, 1.0, None, [])
        self.assertEqual(len(self.agent.memory), 4)
        actions = [t[1] for t in self.agent.memory.buffer]
        self.assertEqual(actions, [1, 5, 2, 4])

    def test_train_step_runs_after_warmup(self):
        self.agent.warmup = 8
        self.agent.batch_size = 8
        state = board_from_moves([]).get_state()
        for _ in range(4):
            self.agent.update_Q(state, 3, -1.0, None, [])
        self.assertIsInstance(self.agent.train_step(), float)

    def test_save_load_roundtrip(self):
        self.agent.epsilon, self.agent.extra = 0.42, {"depth": 3}
        self.agent.save()
        other = DQNAgent(path=self.agent.path)
        self.assertEqual(other.load(), {"depth": 3})
        self.assertAlmostEqual(other.epsilon, 0.42)
        for a, b in zip(self.agent.policy_net.parameters(), other.policy_net.parameters()):
            self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
