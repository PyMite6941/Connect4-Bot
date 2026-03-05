from pixel.connect4 import Game
from pixel.q_learning import QLearning
from pixel.smartBot_Socket import Socket
from pixel.log import Log

class Play:
    def __init__(self,p1,p2):
        self.game = Game(p1,p2)
        self.q_learning = QLearning()
        self.socket = Socket(self.game,p1,p2)
        #self.log = Log()

        self.p1 = p1
        self.p2 = p2
        self.current = p1

    def reward(self,player='Pixel'):
        result = self.game.is_terminal(player)
        if result['end']:
            winner = self.game.winner
            if winner == player:
                return 1.0
            elif winner is None:
                return 0.5
            else:
                return -1.0
        else:
            return 0.01
        
    def socket_get_smartBot_move(self):
        action = self.socket.get_dfs_move()
        self.game.make_move(action,self.current)
        
    def swap_turns(self):
        self.current = 'smartBot' if self.current == 'Pixel' else 'Pixel'

    #def log_move(self):
        #self.log.write(self.game.get_state())