from connect4 import Game
from q_learning import QLearning
from socket import Socket

class Play:
    def __init__(self):
        self.game = Game()

    def reward(self,board,player):
        if self.game.is_over():
            winner = self.game.winner
            if winner == player:
                return 1.0
            elif winner is None:
                return 0.5
            else:
                return -1.0
        else:
            return 0.01
        
    def play_game(self,training=True):
        game = Game()
        ql = QLearning()
        ql.load_Q()
        socket = Socket("moves.txt")
        while not game.is_over():
            state = game.get_state()
            valid_moves = game.possible_moves()
            if training:
                move = ql.choose_moves(state,valid_moves,training=True)
            else:
                move = socket.get_move(game.turn)
            game.make_move(move,game.turn)
            reward = self.reward(game.board,game.turn)
            next_state = game.get_state()
            ql.update_Q(state,move,reward,next_state,game.possible_moves())
        ql.save_Q()