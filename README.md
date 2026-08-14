# Connect4-Bot

Two Connect 4 engines built on a shared board model, plus a self-play harness that runs them against each other:

- **A depth-limited search engine** (`DFS.py`) with a transposition cache and horizontal-symmetry deduplication.
- **A tabular Q-learning agent** (`pixel/q_learning.py`) that learns from self-play and persists its Q-table between runs.

The point of the project is the comparison: a hand-written search engine that knows the rules of Connect 4 versus a reinforcement learner that starts knowing nothing and has to play its way into competence.

## Layout

| File | What it does |
|---|---|
| `connect4.py` | `Board` — the shared game model. Configurable `rows`, `columns`, and `connect` length, so it isn't hardcoded to 6×7×4. Move generation, win detection, evaluation, terminal-state checks. |
| `DFS.py` | Depth-limited search over the move tree, with caching. |
| `pixel/q_learning.py` | `QLearning` — ε-greedy tabular agent, Q-table pickled to `q_table.pkl`. |
| `pixel/smartBot_Socket.py` | Translation layer between the agent's game state and the format `DFS.py` expects, so the two engines can play each other. |
| `pixel/main.py` | `Play` — match driver: turn order, reward shaping, move logging. |
| `run.py` | Entry point. Runs one game, randomizing who moves first. |
| `test.py` | Scratch harness. |

## Running it

```bash
python run.py
```

Randomly assigns first move between `Pixel` (the Q-learning agent) and `smartBot` (the search engine), plays one game to termination, and prints the winner and the agent's accumulated reward. `training=True` updates and saves the Q-table; set it to `False` to evaluate the trained agent without further learning.

## The search engine

`DFS.py` walks the move tree to a fixed depth and evaluates leaves, with two optimizations that do most of the work:

**Transposition caching.** Connect 4 positions are reachable by many different move orders. The cache is keyed on the flattened board state, so a position already evaluated is never re-searched regardless of the path that reached it.

**Symmetry deduplication.** A Connect 4 board is mirror-symmetric about its center column — a position and its horizontal reflection have identical value. Before searching a node, `reflect()` flips each row and checks the cache for the mirrored state. This roughly halves the effective search space near the root, where the branching factor costs the most.

Move generation supports a `reasonable=True` flag that prunes obviously bad candidates rather than enumerating all legal columns, which keeps the tree narrow enough for the cache to pay off.

## The Q-learning agent

Standard tabular Q-learning: state is the flattened board, actions are the legal columns, and `update_Q` applies the Bellman update against the best available next action. Move selection is ε-greedy while training and greedy during evaluation. The table is pickled to `q_table.pkl` and reloaded on start, so training accumulates across runs.

The tabular approach is a deliberate constraint rather than an oversight — Connect 4's state space is far too large to enumerate, so the agent only ever learns the regions of it that self-play actually visits. That limitation is the interesting part, and it's the motivation for moving to a function approximator (see below).

## Status and next steps

Working: the board model, the search engine, the Q-learning agent, and the self-play loop.

The natural next step is replacing the lookup table with a neural network — a DQN — so the agent generalizes across positions it has never seen instead of memorizing the ones it has. Design notes for that are in `ideas.md`.
