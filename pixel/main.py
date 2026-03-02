from pixel.connect4 import Game
from pixel.q_learning import QLearning

class Play:
    def __init__(self,p1,p2):
        self.game = Game()
        self.q_learning = QLearning()

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
        
    def swap_turns(self):
        self.current = 'smartBot' if self.current == 'Pixel' else 'Pixel'