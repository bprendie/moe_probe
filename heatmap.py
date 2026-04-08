"""
MoE Expert Activation Heatmap
------------------------------
Drop-in replacement for the post-run scatterplot in demo_avionics.py.

Public API:
    render_heatmap(routing_history, console, num_experts=40)

routing_history: list of [expert_a, expert_b] per token (top-2 routing)
console:         rich.console.Console instance
"""

import numpy as np
from rich.console import Console
from rich.text import Text

PLOT_WIDTH = 60   # token buckets (each bucket = N tokens wide)
CELL = 2          # chars per cell — makes blocks square-ish in most fonts


def _shade(intensity: float) -> tuple[str, str]:
    """Map a normalized intensity [0,1] to (char*CELL, rich_style)."""
    if intensity == 0:
        return "· " * CELL, "dim"
    elif intensity < 0.20:
        return "░" * CELL, "dim green"
    elif intensity < 0.45:
        return "▒" * CELL, "green"
    elif intensity < 0.70:
        return "▓" * CELL, "yellow"
    elif intensity < 0.88:
        return "█" * CELL, "bold yellow"
    else:
        return "█" * CELL, "bold red"


def render_heatmap(routing_history: list, console: Console, num_experts: int = 40):
    if not routing_history:
        return

    num_tokens = len(routing_history)

    # ── Build frequency matrix [expert × bucket] ──────────────────────────
    freq = np.zeros((num_experts, PLOT_WIDTH), dtype=np.float32)
    for col in range(PLOT_WIDTH):
        t_start = int(col * num_tokens / PLOT_WIDTH)
        t_end   = int((col + 1) * num_tokens / PLOT_WIDTH)
        if t_end == t_start:
            t_end = t_start + 1
        for t in range(t_start, min(t_end, num_tokens)):
            for expert in routing_history[t]:
                if expert < num_experts:
                    freq[expert, col] += 1

    # Normalize globally so backbone experts glow and cold ones stay dark
    global_max = freq.max() or 1.0
    norm = freq / global_max

    # ── Render ─────────────────────────────────────────────────────────────
    console.print()
    console.print("[bold magenta]=== MoE Expert Activation Heatmap · Layer 20 ===[/bold magenta]")
    console.print(
        f"[dim]X: token timeline (0 → {num_tokens})  "
        f"Y: expert  "
        f"color: activation frequency (dim=cold → red=hot)[/dim]"
    )

    tick_label = f"0{'':>{PLOT_WIDTH * CELL - len(str(num_tokens)) - 1}}{num_tokens}"
    prefix_width = 6   # "  E00 "

    for group_start in [0, 20]:
        group_end = group_start + 20
        console.print()

        # Top time axis
        axis = Text(" " * prefix_width + tick_label, style="dim")
        console.print(axis)

        for expert in range(group_start, min(group_end, num_experts)):
            line = Text(f"  E{expert:02d} ", style="bold cyan")
            for col in range(PLOT_WIDTH):
                char, style = _shade(norm[expert, col])
                line.append(char, style=style)
            console.print(line)

        # Bottom time axis (mirrors top)
        console.print(Text(" " * prefix_width + tick_label, style="dim"))

    console.print()


# ── Standalone test with synthetic data ───────────────────────────────────
if __name__ == "__main__":
    import random

    rng = random.Random(42)
    num_tokens = 800

    def mock_history(n):
        history = []
        for t in range(n):
            # Expert 38 and 11: backbone — almost always active
            a = 38 if rng.random() > 0.02 else rng.randint(0, 39)
            b = 11 if rng.random() > 0.05 else rng.randint(0, 39)

            # Expert 10, 18, 32: highly active
            if b == a:
                b = rng.choice([10, 18, 32])

            # Rhythm experts: fire every ~8 tokens
            if t % 8 == 0:
                a = rng.choice([7, 17, 27])

            # Flash-and-die: only first 12 tokens
            if t < 12:
                b = rng.choice([5, 12, 26])

            # Expert 0: delayed entry after token 20
            if t > 20 and rng.random() > 0.55:
                a = 0

            history.append([a, b])
        return history

    c = Console()
    render_heatmap(mock_history(num_tokens), c)
