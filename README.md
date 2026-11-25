# Connect4-Bot

Two Connect 4 engines built on a shared board model, plus a self-play harness that runs them against each other:

- **A depth-limited search engine** (`DFS.py`) with a transposition cache and horizontal-symmetry deduplication.
- **A tabular Q-learning agent** (`clyde/q_learning.py`) that learns from self-play and persists its Q-table between runs.

The point of the project is the comparison: a hand-written search engine that knows the rules of Connect 4 versus a reinforcement learner that starts knowing nothing and has to play its way into competence.

## Layout

```
train_clyde.py      training: interactive menus with no args, headless with flags
run.py              play N games, Clyde vs smartBot (quick checks / eval)
arena/              match harness - wires the two engines together
  match.py          Play: turn order, rewards, logging
  smartbot_socket.py  converts Clyde's board into smartBot's format
  log.py            writes the last game's boards to log.txt
smartbot/           search engine (author: wynn-t)
  connect4.py       Board: rules, move generation, win detection, eval
  DFS.py            depth-limited search with transposition + mirror caching
clyde/              learning agents (author: -XO-OX-)
  connect4.py       Game: Clyde's board model
  q_learning.py     tabular Q-learning (baseline)
  dqn.py            CNN with Q-head + value head, replay buffers, Double-DQN, checkpoints
  search.py         negamax lookahead with the value net at the leaves
tests/              unit tests (win detection, encoding, masking, replay, checkpoints)
```

Generated files (`clyde/dqn.pt`, `clyde/dqn_best.pt`, `clyde/q_table.pkl`, `clyde/training_log.csv`, `log.txt`) are gitignored.

## Running it

```bash
pip install -r requirements.txt

python train_clyde.py              # interactive: pick a duration, tune settings, watch it train, evaluate
python train_clyde.py --hours 8    # headless (scripts / Task Scheduler); stop/resume any time with Ctrl+C
python run.py -n 100 --eval        # greedy Clyde vs smartBot depth 4, no learning
python run.py -n 50 --eval --depth 2
python run.py --agent q -n 500     # the old tabular agent
python run.py -n 100 --eval --search-depth 3   # same net, looking 3 plies ahead
python -m unittest discover tests  # unit tests
```

Training starts Clyde against a random opponent (smartBot depth 0). Every 250 games it plays 100 greedy games at the current depth; at a 75% win rate it moves up a depth (to `--max-depth`, default 5). Each evaluation is logged to `clyde/training_log.csv`; the best weights at the hardest depth reached are kept in `clyde/dqn_best.pt`. `--fresh` starts over.

## The search engine

`DFS.py` walks the move tree to a fixed depth and evaluates leaves, with two optimizations that do most of the work:

**Transposition caching.** Connect 4 positions are reachable by many different move orders. The cache is keyed on the flattened board state, so a position already evaluated is never re-searched regardless of the path that reached it.

**Symmetry deduplication.** A Connect 4 board is mirror-symmetric about its center column — a position and its horizontal reflection have identical value. Before searching a node, `reflect()` flips each row and checks the cache for the mirrored state. This roughly halves the effective search space near the root, where the branching factor costs the most.

Move generation supports a `reasonable=True` flag that prunes obviously bad candidates rather than enumerating all legal columns, which keeps the tree narrow enough for the cache to pay off.

## The Q-learning agent

### Tabular baseline (`q_learning.py`)

Standard tabular Q-learning: state is the flattened board, actions are the legal columns, and `update_Q` applies the Bellman update against the best available next action. Move selection is ε-greedy while training and greedy during evaluation. The table is pickled to `q_table.pkl` and reloaded on start, so training accumulates across runs.

The tabular approach is a deliberate constraint rather than an oversight — Connect 4's state space is far too large to enumerate, so the agent only ever learns the regions of it that self-play actually visits. That limitation is the interesting part, and it's the motivation for moving to a function approximator (see below).

### DQN + value head (`dqn.py`, `search.py`)

The board is encoded from the side to move's perspective as two 6x7 planes (my pieces / opponent's), so one network plays either side. A 4x4 convolution (exactly one possible four-in-a-row window per position) feeds a dense layer, then two heads: **Q(s, a)** for 7 columns, trained with Double-DQN, a target network, illegal-column masking and mirror augmentation; and **V(s)**, a single position value in [-1, 1].

Clyde chooses moves with a shallow lookahead (`--search-depth`, default 1 ply) that scores each resulting position with V. V is trained by **search distillation**: every position seen in training, from both sides, is labelled with a deeper negamax search (`--target-depth`, default 3) whose leaves are scored by the target network and whose wins and draws are exact. V learns to predict that search; the next labels are built from the improved V, so they describe deeper play again. The result is a 1-ply player whose value function has absorbed many plies of calculation.
