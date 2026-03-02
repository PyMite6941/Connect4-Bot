import random

import DFS
from pixel.main import Play

def play_game(self,training=True):
    Play.q_learning.load_Q()
    while not Play.game.is_terminal(Play.current):
        Play.game.get_state()

if __name__ == "__main__":
    random_int = random.randint(1,2)
    p1 = 'Pixel' if random_int %2 == 0 else 'smartBot'
    p2 = 'smartBot' if p1 != 'smartBot' else 'Pixel'
    play = Play(p1, p2)
    play.play_game(training=True)