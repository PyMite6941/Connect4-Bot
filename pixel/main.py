from connect4 import Game
from q_learning import QLearning
from socket import Socket

class Play:
    def __init__(self,p1,p2):
        self.game = Game()
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
        
    def play_game(self,training=True):
        ql = QLearning()
        ql.load_Q()
        socket = Socket("moves.txt")
        while not self.game.is_over():
            state = self.game.get_state()
            valid_moves = self.game.possible_moves()
            if training:
                move = ql.choose_moves(state,valid_moves,training=True)
            else:
                move = socket.get_move(self.game.turn)
            self.game.make_move(move,self.game.turn)
            reward = self.reward(self.game.board,self.game.turn)
            next_state = self.game.get_state()
            ql.update_Q(state,move,reward,next_state,self.game.possible_moves())
        ql.save_Q()