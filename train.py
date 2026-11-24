"""Automated DQN training for Clyde, with a curriculum against smartBot.

Clyde starts against a random opponent (smartBot depth 0). Every --eval-every games
it plays --eval-games greedy games against the current depth; if its win rate reaches
--promote, it moves up one depth. Progress, weights, optimizer and epsilon live in
clyde/dqn.pt, so you can stop (Ctrl+C) and rerun at any time and it picks up where it left off.

    python train.py                    # train until Ctrl+C
    python train.py --hours 8          # overnight run
    python train.py --games 2000       # fixed budget
    python train.py --fresh            # throw away the checkpoint and start over
    python train.py --target-depth 4   # deeper labels for the value head (slower per game)

Clyde plays with a --search-depth lookahead (default 1 ply). Every position it sees is
labelled with a --target-depth search (default 3) and the value head learns to predict
it, so the 1-ply player absorbs what the deeper search knows.

Every evaluation appends a row to clyde/training_log.csv (open it in Excel or plot it).
The best-scoring weights at the hardest depth reached are kept in clyde/dqn_best.pt.
"""
import argparse
import csv
import os
import random
import sys
import time
from dataclasses import dataclass

from clyde.dqn import DQNAgent, MODEL_PATH
from run import play_game

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "clyde", "training_log.csv")
BEST_PATH = os.path.join(HERE, "clyde", "dqn_best.pt")
LOG_FIELDS = ["time", "games", "depth", "epsilon", "train_win", "eval_win", "eval_draw", "eval_loss", "steps"]


def evaluate(agent, depth, games):
    tally = {"Clyde": 0, "smartBot": 0, "Draw": 0}
    for _ in range(games):
        tally[play_game(agent, depth=depth, training=False, log=False)] += 1
    return {k: v / games for k, v in tally.items()}


def pick_opponent_depth(current):
    """Mostly the current depth; sometimes an easier one so earlier lessons aren't forgotten."""
    if current > 0 and random.random() < 0.2:
        return random.randint(0, current - 1)
    return current


def append_log(row):
    new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


@dataclass
class TrainConfig:
    games: int = 0             # stop after this many training games (0 = no limit)
    hours: float = 0           # stop after this many hours (0 = no limit)
    eval_every: int = 250      # training games between evaluations
    eval_games: int = 100      # greedy games per evaluation
    promote: float = 0.75      # eval win rate needed to move up a depth
    max_depth: int = 5         # hardest smartBot depth to train against
    search_depth: int = 1      # Clyde's lookahead when choosing moves
    target_depth: int = 3      # plies of search used to label positions for the value head
    fresh: bool = False        # delete checkpoint, best weights and log first


def load_status():
    """Summary of the saved checkpoint without building a training run (for the TUI)."""
    if not os.path.exists(MODEL_PATH):
        return None
    agent = DQNAgent()
    meta = agent.load()
    return {"games": meta.get("games", 0), "depth": meta.get("depth", 0), "epsilon": agent.epsilon,
            "steps": agent.steps, "best": meta.get("best", {"depth": -1, "win": 0.0})}


def read_log(last=None):
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-last:] if last else rows


def train(cfg, on_start=None, on_game=None, on_eval=None, should_stop=None):
    """Run the curriculum. Callbacks (all optional):
        on_start(meta, agent)                       once, after loading the checkpoint
        on_game(played, meta, winner, elapsed)      after every training game
        on_eval(row, promoted)                      after every evaluation (row = the CSV row)
        should_stop() -> bool                       checked between games
    Ctrl+C stops cleanly. Always saves before returning. Returns (games played this run, meta).
    """
    agent = DQNAgent(search_depth=cfg.search_depth, target_depth=cfg.target_depth)
    if cfg.fresh:
        for path in (MODEL_PATH, BEST_PATH, LOG_PATH):
            if os.path.exists(path):
                os.remove(path)
    meta = agent.load()
    meta.setdefault("depth", 0)
    meta.setdefault("games", 0)
    meta.setdefault("best", {"depth": -1, "win": 0.0})
    agent.extra = meta
    if on_start:
        on_start(meta, agent)

    start, played, wins_since_eval = time.time(), 0, 0

    def out_of_budget():
        return ((cfg.games and played >= cfg.games)
                or (cfg.hours and time.time() - start >= cfg.hours * 3600)
                or (should_stop and should_stop()))

    try:
        while not out_of_budget():
            winner = play_game(agent, depth=pick_opponent_depth(meta["depth"]), training=True, log=False)
            agent.decay_epsilon()
            played += 1
            meta["games"] += 1
            wins_since_eval += winner == "Clyde"
            if on_game:
                on_game(played, meta, winner, time.time() - start)

            if played % cfg.eval_every == 0:
                result = evaluate(agent, meta["depth"], cfg.eval_games)
                row = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"), "games": meta["games"], "depth": meta["depth"],
                    "epsilon": round(agent.epsilon, 4), "train_win": round(wins_since_eval / cfg.eval_every, 3),
                    "eval_win": result["Clyde"], "eval_draw": result["Draw"], "eval_loss": result["smartBot"],
                    "steps": agent.steps,
                }
                append_log(row)
                wins_since_eval = 0

                best = meta["best"]
                if (meta["depth"], result["Clyde"]) > (best["depth"], best["win"]):
                    meta["best"] = {"depth": meta["depth"], "win": result["Clyde"]}
                    agent.save(path=BEST_PATH)

                promoted = result["Clyde"] >= cfg.promote and meta["depth"] < cfg.max_depth
                if promoted:
                    meta["depth"] += 1
                    agent.epsilon = max(agent.epsilon, 0.3)  # explore again against the stronger opponent
                agent.save()
                if on_eval:
                    on_eval(row, promoted)
    except KeyboardInterrupt:
        pass
    agent.save()
    return played, meta


def main():
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")  # live output; safe for non-ASCII paths
    d = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", type=int, default=d.games, help="stop after this many training games (0 = no limit)")
    p.add_argument("--hours", type=float, default=d.hours, help="stop after this many hours (0 = no limit)")
    p.add_argument("--eval-every", type=int, default=d.eval_every, help="training games between evaluations")
    p.add_argument("--eval-games", type=int, default=d.eval_games, help="greedy games per evaluation")
    p.add_argument("--promote", type=float, default=d.promote, help="eval win rate needed to move up a depth")
    p.add_argument("--max-depth", type=int, default=d.max_depth, help="hardest smartBot depth to train against")
    p.add_argument("--search-depth", type=int, default=d.search_depth,
                   help="Clyde's lookahead when choosing moves (default 1)")
    p.add_argument("--target-depth", type=int, default=d.target_depth,
                   help="plies of search used to label positions for the value head (default 3)")
    p.add_argument("--fresh", action="store_true", help="ignore and overwrite the existing checkpoint")
    cfg = TrainConfig(**vars(p.parse_args()))

    def on_start(meta, agent):
        print(f"{'Resuming' if meta['games'] else 'Starting'}: {meta['games']} games played, "
              f"depth {meta['depth']}, epsilon {agent.epsilon:.3f}")

    def on_eval(row, promoted):
        print(f"[{row['time']}] games {row['games']:>6}  depth {row['depth']}  eps {row['epsilon']:.3f}  "
              f"train win {row['train_win']:.2f}  eval W/D/L {row['eval_win']:.2f}/{row['eval_draw']:.2f}/"
              f"{row['eval_loss']:.2f}")
        if promoted:
            print(f"  -> promoted to depth {row['depth'] + 1}")

    played, meta = train(cfg, on_start=on_start, on_eval=on_eval)
    print(f"Saved clyde/dqn.pt after {played} games this run ({meta['games']} total, depth {meta['depth']}).")


if __name__ == "__main__":
    main()
