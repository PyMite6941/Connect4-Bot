"""Pit Clyde (learning agent) against smartBot (search engine).

    python run.py                      # 1 game, DQN Clyde, learns and saves
    python run.py -n 100 --eval        # greedy Clyde, no learning, no save
    python run.py -n 50 --depth 2      # weaker smartBot
    python run.py -n 100 --eval --search-depth 3   # let Clyde look 3 plies ahead
    python run.py --agent q            # old tabular Q-learning agent

For long automated training with a curriculum, use train.py instead.
"""
import argparse
import random

from arena.match import Play
from clyde.q_learning import QLearning


def make_agent(kind, search_depth=1):
    if kind == 'dqn':
        from clyde.dqn import DQNAgent  # torch import only when needed
        return DQNAgent(search_depth=search_depth)
    return QLearning()


def play_game(agent, depth=4, training=True, log=True):
    """Play one game with random first mover. Returns 'Clyde', 'smartBot' or 'Draw'."""
    p1, p2 = random.sample(['Clyde', 'smartBot'], 2)
    play = Play(p1, p2, agent=agent, depth=depth, log=log)
    pending = None  # Clyde's last (state, action), waiting to learn its outcome
    observe = training and hasattr(agent, 'observe')  # value-head labelling (DQN only)
    if observe:
        agent.observe(play.game.get_state(), play.current)

    while True:
        if play.current == 'Clyde':
            state = play.game.get_state()
            valid = play.game.possible_moves()
            if training and pending:
                agent.update_Q(*pending, play.step_reward(), state, valid)
            move = agent.choose_moves(state, valid, training=training)
            play.game.make_move(move, 'Clyde')
            pending = (state, move)
        else:
            play.socket_get_smartBot_move()
        play.log_move()

        result = play.game.is_terminal(play.current)
        if result['end']:
            if training and pending:
                agent.update_Q(*pending, play.final_reward(result['winner']), None, [])
            return result['winner']
        play.swap_turns()
        if observe:
            agent.observe(play.game.get_state(), play.current)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-n', '--games', type=int, default=1, help='number of games (default 1)')
    parser.add_argument('--eval', action='store_true', help='greedy play, no learning, no save')
    parser.add_argument('--depth', type=int, default=4, help='smartBot search depth, 0 = random (default 4)')
    parser.add_argument('--agent', choices=['dqn', 'q'], default='dqn', help='which Clyde agent (default dqn)')
    parser.add_argument('--search-depth', type=int, default=1, help="Clyde's lookahead in plies, 0 = Q-head only (default 1)")
    args = parser.parse_args()

    training = not args.eval
    agent = make_agent(args.agent, args.search_depth)
    agent.load_Q()

    tally = {'Clyde': 0, 'smartBot': 0, 'Draw': 0}
    for i in range(1, args.games + 1):
        winner = play_game(agent, depth=args.depth, training=training)
        tally[winner] += 1
        if training and hasattr(agent, 'decay_epsilon'):
            agent.decay_epsilon()
        print(f"Game {i}: winner {winner}")

    if training:
        agent.save_Q()
    print(f"\nClyde {tally['Clyde']} / smartBot {tally['smartBot']} / Draw {tally['Draw']}"
          f"  ({args.agent}, {'training' if training else 'eval'}, smartBot depth {args.depth}"
          f"{f', Clyde search {args.search_depth}' if args.agent == 'dqn' else ''})")


if __name__ == '__main__':
    main()
