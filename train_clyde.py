"""Interactive trainer for Clyde (rich + questionary).

    python train_clyde.py

Menus to train for a chosen amount of time (or games), tune the training settings,
watch progress live, evaluate against smartBot, and browse the training history.
Everything goes through train.py, so `python train.py ...` still works for scripts.
"""
import os
import re
import sys
import time

import questionary
from questionary import Choice
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

import train
from clyde.dqn import DQNAgent, MODEL_PATH
from run import play_game

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252, which can't draw the bars
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
console = Console()

STYLE = questionary.Style([
    ("qmark", "fg:#e5a50a bold"),
    ("question", "bold"),
    ("pointer", "fg:#e5a50a bold"),
    ("highlighted", "fg:#e5a50a bold"),
    ("answer", "fg:#62a0ea bold"),
])

DURATION_PRESETS = [
    Choice("Quick test      5 minutes", ("hours", 5 / 60)),
    Choice("Short session   30 minutes", ("hours", 0.5)),
    Choice("Session         1 hour", ("hours", 1.0)),
    Choice("Long session    3 hours", ("hours", 3.0)),
    Choice("Overnight       8 hours", ("hours", 8.0)),
    Choice("Custom time...", "custom_time"),
    Choice("Fixed number of games...", "custom_games"),
    Choice("Until I press Ctrl+C", ("forever", 0)),
]

TARGET_DEPTH_CHOICES = [
    Choice("2 plies   fastest (~0.06 s/game), shallow labels", 2),
    Choice("3 plies   balanced (~0.2 s/game), recommended", 3),
    Choice("4 plies   deep labels, ~4x slower (~0.9 s/game)", 4),
]


# ---- helpers ----------------------------------------------------------------------
def parse_duration(text):
    """'45m', '2h', '1h30m', '1.5h', '90' (minutes) -> hours as float, or None if invalid."""
    text = text.strip().lower().replace(" ", "")
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text) / 60
    m = re.fullmatch(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?:in)?)?", text)
    if not m or not any(m.groups()):
        return None
    hours = float(m.group(1) or 0) + float(m.group(2) or 0) / 60
    return hours if hours > 0 else None


def fmt_hours(h):
    minutes = round(h * 60)
    return f"{minutes // 60}h {minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def pct(x):
    return f"{float(x) * 100:.0f}%"


def win_bar(rate, width=20):
    filled = round(float(rate) * width)
    color = "green" if float(rate) >= 0.75 else "yellow" if float(rate) >= 0.4 else "red"
    return Text("█" * filled, style=color) + Text("░" * (width - filled), style="grey37")


def ask(question):
    """Run a questionary prompt; Ctrl+C / Esc returns None instead of raising."""
    try:
        return question.ask()
    except KeyboardInterrupt:
        return None


# ---- status & history -------------------------------------------------------------
def status_panel():
    status = None
    try:
        status = train.load_status()
    except SystemExit as e:  # checkpoint from an older network layout
        return Panel(Text(str(e), style="red"), title="Clyde", border_style="red")

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()
    if status is None:
        grid.add_row("Checkpoint", Text("none yet - the first training run starts from scratch", style="yellow"))
    else:
        best = status["best"]
        grid.add_row("Games trained", f"{status['games']:,}")
        grid.add_row("Curriculum", f"smartBot depth {status['depth']}")
        grid.add_row("Exploration", f"epsilon {status['epsilon']:.3f}")
        grid.add_row("Best so far", f"{pct(best['win'])} wins at depth {best['depth']}" if best["depth"] >= 0 else "-")
        last = train.read_log(last=1)
        if last:
            r = last[0]
            grid.add_row("Last eval", Text.assemble(win_bar(r["eval_win"], 12), f"  {pct(r['eval_win'])} win / "
                                                     f"{pct(r['eval_draw'])} draw at depth {r['depth']}  ({r['time'][5:16]})"))
    return Panel(grid, title="[bold]Clyde[/] - Connect 4 neural net", border_style="#e5a50a", box=box.ROUNDED)


def eval_table(rows, title="Evaluations"):
    t = Table(title=title, box=box.SIMPLE_HEAVY, title_style="bold", expand=False)
    for col, kw in [("time", {}), ("games", {"justify": "right"}), ("depth", {"justify": "center"}),
                    ("eps", {"justify": "right"}), ("train win", {"justify": "right"}),
                    ("eval W/D/L", {"justify": "right"}), ("", {})]:
        t.add_column(col, **kw)
    for r in rows:
        t.add_row(str(r["time"])[11:16] if len(str(r["time"])) > 11 else str(r["time"]),
                  f"{int(r['games']):,}", str(r["depth"]), f"{float(r['epsilon']):.3f}", pct(r["train_win"]),
                  f"{pct(r['eval_win'])}/{pct(r['eval_draw'])}/{pct(r['eval_loss'])}", win_bar(r["eval_win"], 10))
    return t


def show_history():
    rows = train.read_log()
    if not rows:
        console.print("[yellow]No training history yet.[/]\n")
        return
    console.print(eval_table(rows[-30:], title=f"Last {min(30, len(rows))} of {len(rows)} evaluations"))

    by_depth = {}
    for r in rows:
        by_depth.setdefault(r["depth"], []).append(r)
    summary = Table(title="Per curriculum depth", box=box.SIMPLE_HEAVY, title_style="bold")
    for col in ("depth", "evals", "game range", "best win", "latest win"):
        summary.add_column(col, justify="right")
    for depth, rs in sorted(by_depth.items(), key=lambda kv: int(kv[0])):
        summary.add_row(depth, str(len(rs)), f"{int(rs[0]['games']):,} - {int(rs[-1]['games']):,}",
                        pct(max(float(r["eval_win"]) for r in rs)), pct(rs[-1]["eval_win"]))
    console.print(summary)
    console.print()


# ---- training ---------------------------------------------------------------------
def configure_training():
    """Ask for duration + settings. Returns a TrainConfig, or None if cancelled."""
    cfg = train.TrainConfig()

    status = None
    try:
        status = train.load_status()
    except SystemExit:
        console.print("[yellow]The saved checkpoint uses an older network layout, so this run starts fresh.[/]")
        cfg.fresh = True

    if status and not cfg.fresh:
        keep = ask(questionary.select(
            "Checkpoint found. What should this run do?",
            choices=[Choice(f"Continue training ({status['games']:,} games, depth {status['depth']})", False),
                     Choice("Start over from scratch (deletes checkpoint, best weights and history)", True)],
            style=STYLE))
        if keep is None:
            return None
        cfg.fresh = keep

    duration = ask(questionary.select("How long should Clyde train?", choices=DURATION_PRESETS, style=STYLE))
    if duration is None:
        return None
    if duration == "custom_time":
        text = ask(questionary.text("Duration (e.g. 45m, 2h, 1h30m):",
                                    validate=lambda t: parse_duration(t) is not None or "Try 45m, 2h or 1h30m",
                                    style=STYLE))
        if text is None:
            return None
        cfg.hours = parse_duration(text)
    elif duration == "custom_games":
        text = ask(questionary.text("Number of training games:", default="2000",
                                    validate=lambda t: t.isdigit() and int(t) > 0 or "Enter a positive whole number",
                                    style=STYLE))
        if text is None:
            return None
        cfg.games = int(text)
    elif duration[0] == "hours":
        cfg.hours = duration[1]

    advanced = ask(questionary.confirm("Adjust training settings? (defaults are sensible)", default=False, style=STYLE))
    if advanced is None:
        return None
    if advanced:
        answers = [
            ask(questionary.select("Value-head label depth (how deep Clyde learns to 'see'):",
                                   choices=TARGET_DEPTH_CHOICES, default=3, style=STYLE)),
            ask(questionary.select("Clyde's lookahead while playing:",
                                   choices=[Choice("1 ply   (the goal: deep knowledge, shallow search)", 1),
                                            Choice("2 plies", 2), Choice("3 plies", 3)], style=STYLE)),
            ask(questionary.select("Hardest smartBot depth to climb to:",
                                   choices=[Choice(str(d), d) for d in range(1, 7)],
                                   default=cfg.max_depth, style=STYLE)),
            ask(questionary.text("Training games between evaluations:", default=str(cfg.eval_every),
                                 validate=lambda t: t.isdigit() and int(t) > 0 or "Positive whole number",
                                 style=STYLE)),
            ask(questionary.text("Games per evaluation:", default=str(cfg.eval_games),
                                 validate=lambda t: t.isdigit() and int(t) > 0 or "Positive whole number",
                                 style=STYLE)),
            ask(questionary.text("Win rate needed to move up a depth (0-1):", default=str(cfg.promote),
                                 validate=lambda t: _is_rate(t) or "A number between 0 and 1", style=STYLE)),
        ]
        if any(a is None for a in answers):
            return None
        cfg.target_depth, cfg.search_depth, cfg.max_depth = answers[0], answers[1], answers[2]
        cfg.eval_every, cfg.eval_games, cfg.promote = int(answers[3]), int(answers[4]), float(answers[5])

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("Mode", "[red]start over[/]" if cfg.fresh else "continue" if status else "new")
    summary.add_row("Budget", fmt_hours(cfg.hours) if cfg.hours else f"{cfg.games:,} games" if cfg.games
                    else "until Ctrl+C")
    summary.add_row("Label depth", f"{cfg.target_depth} plies")
    summary.add_row("Play lookahead", f"{cfg.search_depth} ply")
    summary.add_row("Curriculum", f"up to smartBot depth {cfg.max_depth}, promote at {pct(cfg.promote)}")
    summary.add_row("Evaluation", f"{cfg.eval_games} games every {cfg.eval_every} training games")
    console.print(Panel(summary, title="Training plan", border_style="#62a0ea", box=box.ROUNDED))

    go = ask(questionary.confirm("Start training?", default=True, style=STYLE))
    return cfg if go else None


def _is_rate(t):
    try:
        return 0 < float(t) <= 1
    except ValueError:
        return False


def run_training(cfg):
    """Train with a live dashboard. Ctrl+C stops after the current game and saves."""
    if cfg.hours:
        total, unit = cfg.hours * 3600, "time"
    elif cfg.games:
        total, unit = cfg.games, "games"
    else:
        total, unit = None, "forever"

    progress = Progress(TextColumn("[bold]{task.description}"), BarColumn(bar_width=None),
                        TextColumn("{task.percentage:>3.0f}%"), TimeElapsedColumn(), TextColumn("left"),
                        TimeRemainingColumn(), expand=True)
    task = progress.add_task("Training", total=total)
    evals, state = [], {"line": Text("loading checkpoint...")}

    def render():
        parts = [progress, state["line"]]
        if evals:
            parts.append(eval_table(evals[-8:], title="Evaluations this run"))
        parts.append(Text("Ctrl+C stops after the current game and saves.", style="grey50"))
        return Panel(Group(*parts), title="[bold]Training Clyde[/]", border_style="#e5a50a", box=box.ROUNDED)

    with Live(render(), console=console, refresh_per_second=4, transient=False) as live:
        def on_start(meta, agent):
            state["line"] = Text(f"{'Resuming at' if meta['games'] else 'Starting from'} {meta['games']:,} games, "
                                 f"smartBot depth {meta['depth']}, epsilon {agent.epsilon:.3f}")
            state["agent"] = agent
            live.update(render())

        def on_game(played, meta, winner, elapsed):
            progress.update(task, completed=min(elapsed, total) if unit == "time" else played)
            wins = state.setdefault("wins", [])
            wins.append(winner == "Clyde")
            recent = wins[-100:]
            state["line"] = Text.assemble(
                ("games ", "grey50"), f"{played:,} this run / {meta['games']:,} total   ",
                ("depth ", "grey50"), f"{meta['depth']}   ",
                ("eps ", "grey50"), f"{state['agent'].epsilon:.3f}   ",
                ("last 100 ", "grey50"), win_bar(sum(recent) / len(recent), 10), f" {sum(recent) / len(recent):.0%}")
            live.update(render())

        def on_eval(row, promoted):
            evals.append(row)
            if promoted:
                live.console.print(f"[bold green]Promoted[/] to smartBot depth {row['depth'] + 1} "
                                   f"after {pct(row['eval_win'])} wins at depth {row['depth']}")
            live.update(render())

        played, meta = train.train(cfg, on_start=on_start, on_game=on_game, on_eval=on_eval)

    console.print(f"\n[bold green]Saved[/] clyde/dqn.pt - {played:,} games this run, "
                  f"{meta['games']:,} total, now at smartBot depth {meta['depth']}.\n")


# ---- evaluation -------------------------------------------------------------------
def configure_eval():
    if not os.path.exists(MODEL_PATH):
        console.print("[yellow]No checkpoint yet - train Clyde first.[/]\n")
        return None
    weights = ask(questionary.select("Which weights?", choices=[
        Choice("Latest checkpoint (clyde/dqn.pt)", MODEL_PATH),
        Choice("Best so far (clyde/dqn_best.pt)", train.BEST_PATH),
    ], style=STYLE)) if os.path.exists(train.BEST_PATH) else MODEL_PATH
    if weights is None:
        return None
    depth = ask(questionary.select("smartBot depth:", choices=[Choice(f"{d}{'  (random moves)' if d == 0 else ''}", d)
                                                                for d in range(0, 7)],
                                   default=4, style=STYLE))
    if depth is None:
        return None
    search = ask(questionary.select("Clyde's lookahead:", choices=[Choice("1 ply", 1), Choice("2 plies", 2),
                                                                    Choice("3 plies", 3), Choice("4 plies", 4)],
                                    style=STYLE))
    if search is None:
        return None
    games = ask(questionary.text("Number of games:", default="100",
                                 validate=lambda t: t.isdigit() and int(t) > 0 or "Positive whole number",
                                 style=STYLE))
    if games is None:
        return None
    return weights, depth, search, int(games)


def run_eval(weights, depth, search, games):
    agent = DQNAgent(path=weights, search_depth=search)
    try:
        agent.load()
    except SystemExit as e:
        console.print(f"[red]{e}[/]\n")
        return
    tally = {"Clyde": 0, "Draw": 0, "smartBot": 0}
    start = time.time()
    with Progress(TextColumn("[bold]Evaluating"), BarColumn(), TextColumn("{task.completed}/{task.total}"),
                  TextColumn("{task.fields[score]}"), TimeRemainingColumn(), console=console) as progress:
        task = progress.add_task("eval", total=games, score="")
        for _ in range(games):
            tally[play_game(agent, depth=depth, training=False, log=False)] += 1
            progress.update(task, advance=1,
                            score=f"W {tally['Clyde']}  D {tally['Draw']}  L {tally['smartBot']}")

    t = Table(box=box.SIMPLE_HEAVY, title=f"Clyde ({search}-ply) vs smartBot depth {depth} - {games} games",
              title_style="bold")
    for col in ("result", "games", "rate", ""):
        t.add_column(col, justify="right" if col in ("games", "rate") else "left")
    for label, key in (("Clyde wins", "Clyde"), ("Draws", "Draw"), ("smartBot wins", "smartBot")):
        t.add_row(label, str(tally[key]), pct(tally[key] / games), win_bar(tally[key] / games))
    console.print(t)
    console.print(f"[grey50]{(time.time() - start) / games * 1000:.0f} ms per game[/]\n")


# ---- main menu --------------------------------------------------------------------
def reset():
    if not any(os.path.exists(p) for p in (MODEL_PATH, train.BEST_PATH, train.LOG_PATH)):
        console.print("[yellow]Nothing to delete.[/]\n")
        return
    sure = ask(questionary.confirm("Delete the checkpoint, best weights and training history?", default=False,
                                   style=STYLE))
    if sure:
        for p in (MODEL_PATH, train.BEST_PATH, train.LOG_PATH):
            if os.path.exists(p):
                os.remove(p)
        console.print("[green]Deleted. The next training run starts from scratch.[/]\n")


def main():
    while True:
        console.print(status_panel())
        action = ask(questionary.select("What do you want to do?", choices=[
            Choice("Train Clyde", "train"),
            Choice("Evaluate against smartBot", "eval"),
            Choice("Training history", "history"),
            Choice("Start over (delete checkpoint)", "reset"),
            Choice("Quit", "quit"),
        ], style=STYLE))
        if action in (None, "quit"):
            return
        if action == "train":
            cfg = configure_training()
            if cfg:
                run_training(cfg)
        elif action == "eval":
            choice = configure_eval()
            if choice:
                run_eval(*choice)
        elif action == "history":
            show_history()
        elif action == "reset":
            reset()


if __name__ == "__main__":
    main()
