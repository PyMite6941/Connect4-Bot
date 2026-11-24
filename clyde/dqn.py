"""Clyde - neural-network agent (DQN + value head + lookahead).

Drop-in for QLearning: same choose_moves / update_Q / load_Q / save_Q shape, so the
arena and run.py can use either agent.

Pieces:
  encode_*       boards -> (2, 6, 7) tensors from the side to move's perspective
  QNetwork       shared CNN trunk with two heads:
                   q_head  7 action values Q(s, a)  - trained by Double-DQN (update_Q)
                   v_head  1 position value V(s)    - trained to match a deeper search (observe)
  DQNAgent       picks moves with a `search_depth`-ply lookahead using V at the leaves
                 (or plain argmax Q when search_depth = 0)

How "deep calculation from one ply" works:
  Every position seen in training is labelled with search_value(depth = target_depth),
  a lookahead that scores its leaves with the (target) value net, plus exact results
  for wins and draws found inside the tree. V is trained toward those labels. As V gets
  better, the labels it produces describe deeper and deeper play, so at game time
  a 1-ply lookahead over V already reflects many plies of calculation.
"""
import os
import random
from collections import deque

import torch
from torch import nn

from clyde.search import COLS, ROWS, move_values, search_value, to_relative

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dqn.pt")


# ---- encoding -------------------------------------------------------------------
def encode_relative(boards):
    """List of relative boards (1 = side to move, -1 = other, 0 = empty) -> (N, 2, 6, 7)."""
    t = torch.tensor(boards, dtype=torch.int8).view(-1, ROWS, COLS)
    return torch.stack(((t == 1), (t == -1)), dim=1).float()


def encode_state(state, me="Clyde"):
    """Game.get_state() tuple (player names / 0) -> (2, 6, 7) tensor from `me`'s view."""
    return encode_relative([to_relative(state, me)])[0]


def mirror_state(state):
    """Reflect a flat board left-right (Connect 4 is symmetric about the centre column)."""
    return tuple(state[r * COLS + (COLS - 1 - c)] for r in range(ROWS) for c in range(COLS))


def valid_mask(valid_actions):
    mask = torch.zeros(COLS, dtype=torch.bool)
    mask[list(valid_actions)] = True
    return mask


# ---- network --------------------------------------------------------------------
class QNetwork(nn.Module):
    """(batch, 2, 6, 7) -> (Q-values (batch, 7), V (batch,)).

    A 4x4 convolution sees exactly one possible four-in-a-row window (horizontal,
    vertical or diagonal) at each position, so the first layer learns "threat"
    detectors; the dense layer combines them; two heads read the result.
    """

    def __init__(self, channels=64, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(2, channels, 4), nn.ReLU(),          # -> (channels, 3, 4)
            nn.Flatten(),
            nn.Linear(channels * 3 * 4, hidden), nn.ReLU(),
        )
        self.q_head = nn.Linear(hidden, COLS)
        self.v_head = nn.Sequential(nn.Linear(hidden, 1), nn.Tanh())  # value in [-1, 1]

    def forward(self, x):
        h = self.trunk(x)
        return self.q_head(h), self.v_head(h).squeeze(1)


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *item):
        self.buffer.append(item)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ---- agent ----------------------------------------------------------------------
class DQNAgent:
    def __init__(self, name="Clyde", path=MODEL_PATH, search_depth=1, target_depth=3):
        self.name = name
        self.path = path
        self.search_depth = search_depth  # plies Clyde looks ahead when choosing a move (0 = Q-head only)
        self.target_depth = target_depth  # plies of search used to label positions for the value head

        self.gamma = 0.99
        self.epsilon, self.epsilon_min, self.epsilon_decay = 1.0, 0.05, 0.999  # decayed once per game
        self.batch_size = 64
        self.train_every = 2              # gradient step every N Clyde moves
        self.warmup = 1_000               # samples in a buffer before that loss is used
        self.target_sync_every = 500      # train steps between target <- policy copies
        self.steps = 0
        self.moves = 0
        self.memory = ReplayBuffer()               # (s, a, r, s', done, next_mask) for the Q head
        self.value_memory = ReplayBuffer(50_000)   # (s, search target) for the V head
        self.extra = {}  # training metadata (curriculum stage, games played) kept in the checkpoint

        self.policy_net = QNetwork()
        self.target_net = QNetwork()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=5e-4)
        self.loss_fn = nn.SmoothL1Loss()

    # ---- evaluation ---------------------------------------------------------------
    def _values(self, net, boards):
        with torch.no_grad():
            return net(encode_relative(boards))[1].tolist()

    def evaluate(self, state, me=None):
        """Scalar position value of a Game state for `me` (default Clyde) to move, in [-1, 1]."""
        return self._values(self.policy_net, [to_relative(state, me or self.name)])[0]

    def q_values(self, state):
        with torch.no_grad():
            return self.policy_net(encode_state(state, self.name).unsqueeze(0))[0][0]

    # ---- acting -------------------------------------------------------------------
    def choose_moves(self, state, valid_actions, training=True):
        """Epsilon-greedy while training, greedy otherwise. Never picks a full column."""
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        if self.search_depth > 0:
            values = move_values(to_relative(state, self.name), self.search_depth,
                                 lambda boards: self._values(self.policy_net, boards))
            values = {c: v for c, v in values.items() if c in valid_actions}
            best = max(values.values())
            return random.choice([c for c, v in values.items() if v >= best - 1e-6])
        q = self.q_values(state)
        q[~valid_mask(valid_actions)] = -float("inf")
        return int(q.argmax())

    # ---- learning -----------------------------------------------------------------
    def observe(self, state, to_move):
        """Label a non-terminal position with a deeper search and store it for the value head.

        Called for every position in a training game (both sides to move), so V also
        learns from the opponent's strong moves.
        """
        board = to_relative(state, to_move)
        target = search_value(board, self.target_depth, lambda b: self._values(self.target_net, b))
        self.value_memory.push(board, target)
        self.value_memory.push(mirror_state(board), target)

    def update_Q(self, state, action, reward, next_state, possible_next_actions):
        """Same signature as QLearning.update_Q: store the transition (plus its mirror) and train."""
        done = next_state is None
        self._remember(state, action, reward, next_state, possible_next_actions, done)
        self._remember(mirror_state(state), COLS - 1 - action, reward,
                       None if done else mirror_state(next_state),
                       [COLS - 1 - a for a in possible_next_actions], done)
        self.moves += 1
        if self.moves % self.train_every == 0:
            self.train_step()

    def _remember(self, state, action, reward, next_state, next_valid, done):
        s = encode_state(state, self.name)
        s2 = torch.zeros_like(s) if done else encode_state(next_state, self.name)
        mask = valid_mask(next_valid) if next_valid else torch.zeros(COLS, dtype=torch.bool)
        self.memory.push(s, action, reward, s2, float(done), mask)

    def _q_loss(self):
        s, a, r, s2, done, mask = zip(*self.memory.sample(self.batch_size))
        s, s2, mask = torch.stack(s), torch.stack(s2), torch.stack(mask)
        a = torch.tensor(a).unsqueeze(1)
        r, done = torch.tensor(r, dtype=torch.float32), torch.tensor(done)

        q = self.policy_net(s)[0].gather(1, a).squeeze(1)
        with torch.no_grad():
            # Double DQN: policy net picks the next action, target net scores it.
            next_a = self.policy_net(s2)[0].masked_fill(~mask, -float("inf")).argmax(1, keepdim=True)
            next_q = self.target_net(s2)[0].gather(1, next_a).squeeze(1)
            next_q = torch.where(done.bool(), torch.zeros_like(next_q), next_q)
            target = r + self.gamma * next_q
        return self.loss_fn(q, target)

    def _v_loss(self):
        boards, targets = zip(*self.value_memory.sample(self.batch_size))
        v = self.policy_net(encode_relative(list(boards)))[1]
        return nn.functional.mse_loss(v, torch.tensor(targets, dtype=torch.float32))

    def train_step(self):
        losses = []
        if len(self.memory) >= max(self.warmup, self.batch_size):
            losses.append(self._q_loss())
        if len(self.value_memory) >= max(self.warmup, self.batch_size):
            losses.append(self._v_loss())
        if not losses:
            return None
        loss = sum(losses)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_sync_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ---- persistence --------------------------------------------------------------
    def save(self, extra=None, path=None):
        torch.save({
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
            "extra": self.extra if extra is None else extra,
        }, path or self.path)

    def load(self, path=None):
        """Load a checkpoint if one exists. Returns its `extra` dict ({} when starting fresh)."""
        path = path or self.path
        if not os.path.exists(path):
            return {}
        ckpt = torch.load(path, weights_only=False)
        try:
            self.policy_net.load_state_dict(ckpt["policy"])
            self.target_net.load_state_dict(ckpt["target"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
        except (RuntimeError, ValueError) as e:
            raise SystemExit(f"{path} was saved by an older network layout and can't be loaded.\n"
                             f"Start over with:  python train.py --fresh\n({e.__class__.__name__})")
        self.epsilon, self.steps = ckpt["epsilon"], ckpt["steps"]
        self.extra = ckpt.get("extra", {})
        return self.extra

    # QLearning-compatible aliases so run.py can treat both agents the same
    def load_Q(self):
        return self.load()

    def save_Q(self):
        self.save()
