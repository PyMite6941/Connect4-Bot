import random

import DFS
from pixel.main import Play

def play_game(p1,p2,training=True):
    play = Play(p1,p2)
    play.q_learning.load_Q()
    total_reward = 0.0
    while not play.game.is_terminal(play.current)['end']:
        result = play.game.is_terminal(play.current)
        if result['end']:
            if training:
                play.q_learning.update_Q(state, action, reward, None, valid_moves)
            print(f"Game Over! Winner: {play.game.winner}, Total Reward: {total_reward}")
            break
        if play.current == 'Pixel':
            state = play.game.get_state()
            valid_moves = play.game.possible_moves()
            if training:
                move = play.q_learning.choose_moves(state,valid_moves,training=True)
            else:
                move = play.socket.get_dfs_move()
            play.game.make_move(move,play.current)
        else:
            action = play.socket.get_dfs_move()
            play.game.make_move(action,play.current)
        reward = play.reward(play.current)
        total_reward += reward
        play.log_move()
        play.swap_turns()

if __name__ == "__main__":
    random_int = random.randint(1,2)
    p1 = 'Pixel' if random_int %2 == 0 else 'smartBot'
    p2 = 'smartBot' if p1 != 'smartBot' else 'Pixel'
    play_game(p1,p2,training=True)