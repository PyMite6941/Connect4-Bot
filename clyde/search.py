"""Negamax lookahead for Clyde, with the value network at the leaves.

Boards here are *relative*: a flat 42-tuple where 1 = side to move, -1 = the other
side, 0 = empty. After a move the board is negated, so the side to move is always 1
and one function scores both players (negamax).

Values are from the side to move's view: +1 = winning, -1 = losing, 0 = draw/unclear.

Two uses:
  * play time   - best_move(board, depth=1): try each column, score the result with the net.
  * train time  - search_value(board, depth=3): a deeper look whose result becomes the
                  training target for V(board). The net learns to predict depth-3 search,
                  so later depth-3 searches (whose leaves are that net) see further still.
                  Repeat, and a 1-ply player ends up "knowing" what deep search would find.

All leaves of a search are scored in one batched forward pass, so full minimax (no
alpha-beta pruning) is faster here than pruning with one net call per leaf.
"""
ROWS, COLS = 6, 7
WIN = 1.0
DECAY = 0.99  # a win now is worth slightly more than a win later -> Clyde doesn't dawdle

_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def to_relative(state, me):
    """Game.get_state() tuple (player names / 0) -> relative board with `me` to move."""
    return tuple(0 if c == 0 else (1 if c == me else -1) for c in state)


def legal_moves(board):
    return [c for c in range(COLS) if board[c] == 0]


def drop(board, col):
    """Place a 1 in `col`. Returns (new_board, row). Assumes the column isn't full."""
    for r in range(ROWS - 1, -1, -1):
        i = r * COLS + col
        if board[i] == 0:
            b = list(board)
            b[i] = 1
            return tuple(b), r
    raise ValueError(f"column {col} is full")


def is_win(board, row, col):
    """Did the 1 just placed at (row, col) make four in a row?"""
    for dr, dc in _DIRECTIONS:
        count = 1
        for sign in (1, -1):
            r, c = row + dr * sign, col + dc * sign
            while 0 <= r < ROWS and 0 <= c < COLS and board[r * COLS + c] == 1:
                count += 1
                r, c = r + dr * sign, c + dc * sign
        if count >= 4:
            return True
    return False


def flip(board):
    return tuple(-x for x in board)


def _expand(board, depth, leaves):
    """Build the search tree. Returns a float (exact value) or a structure to back up later."""
    moves = legal_moves(board)
    if not moves:
        return 0.0  # board full: draw
    children = []
    for col in moves:
        nb, row = drop(board, col)
        if is_win(nb, row, col):
            return WIN  # a win in one beats anything else; no need to look further
        if depth <= 1:
            leaves.append(flip(nb))
            children.append(("leaf", len(leaves) - 1))
        else:
            children.append(_expand(flip(nb), depth - 1, leaves))
    return children


def _backup(node, leaf_values):
    """Value of `node` for its side to move (child values are for the opponent, hence the minus)."""
    if isinstance(node, float):
        return node
    best = -float("inf")
    for child in node:
        if isinstance(child, tuple):
            v = leaf_values[child[1]]
        else:
            v = _backup(child, leaf_values)
        best = max(best, -DECAY * v)
    return best


def search_value(board, depth, evaluate):
    """Negamax value of `board` (side to move = 1). `evaluate(list_of_boards)` -> list of floats."""
    leaves = []
    tree = _expand(board, depth, leaves)
    values = evaluate(leaves) if leaves else []
    return _backup(tree, values)


def move_values(board, depth, evaluate):
    """{column: value of playing it} for the side to move, searching `depth` plies in total."""
    leaves, subtrees, result = [], {}, {}
    for col in legal_moves(board):
        nb, row = drop(board, col)
        if is_win(nb, row, col):
            result[col] = WIN
        elif depth <= 1:
            leaves.append(flip(nb))
            subtrees[col] = ("leaf", len(leaves) - 1)
        else:
            subtrees[col] = _expand(flip(nb), depth - 1, leaves)
    values = evaluate(leaves) if leaves else []
    for col, node in subtrees.items():
        if isinstance(node, tuple):
            result[col] = -DECAY * values[node[1]]
        else:
            result[col] = -DECAY * _backup(node, values)
    return result
