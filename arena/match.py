from clyde.connect4 import Game
from clyde.q_learning import QLearning
from arena.smartbot_socket import Socket
from arena.log import Log


class Play:
    def __init__(self, p1, p2, agent=None, depth=4, log=True):
        self.game = Game(p1, p2)
        self.q_learning = agent if agent is not None else QLearning()
        self.socket = Socket(self.game, p1, p2)
        self.log = Log() if log else None
        self.depth = depth

        self.p1 = p1
        self.p2 = p2
        self.current = p1

    def step_reward(self):
        """Reward for a non-terminal move."""
        return 0.01

    def final_reward(self, winner, player='Clyde'):
        """Reward for the move that ended the game, from `player`'s perspective."""
        if winner == player:
            return 1.0
        if winner == 'Draw':
            return 0.5
        return -1.0

    def socket_get_smartBot_move(self):
        action = self.socket.get_dfs_move(self.depth)
        self.game.make_move(action, self.current)

    def swap_turns(self):
        self.current = 'smartBot' if self.current == 'Clyde' else 'Clyde'

    def log_move(self):
        if self.log:
            self.log.write(self.game.get_state())
