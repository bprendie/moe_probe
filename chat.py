import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import time
import sys
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.live import Live
from rich.layout import Layout
from rich.console import Group
from rich.text import Text

console = Console(highlight=False)

# ==========================================
# Load Model
# ==========================================
console.print(Rule(style="dim"))
console.print(Panel.fit(
    "[bold white]Granite 3.0 MoE[/bold white]  [dim]·[/dim]  [cyan]ibm-granite/granite-3.0-3b-a800m-instruct[/cyan]\n"
    "[dim]Loading model into VRAM...[/dim]",
    border_style="cyan",
    padding=(0, 2),
))

model_id = "ibm-granite/granite-3.0-3b-a800m-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="cuda"
)

num_experts = getattr(model.config, "num_local_experts", 40)

# ==========================================
# Routing Hook
# ==========================================
routing_queue = []

def router_hook(module, args, output):
    logits = output[0] if isinstance(output, tuple) else output
    probs = torch.nn.functional.softmax(logits, dim=-1)
    routing_queue.append(probs[-1, :].detach().float().cpu().numpy())

target_layer = "layers.20.block_sparse_moe.router"
hook_attached = False

for name, module in model.named_modules():
    if target_layer in name and isinstance(module, torch.nn.Linear):
        if module.out_features == num_experts:
            module.register_forward_hook(router_hook)
            hook_attached = True
            break

console.print(Panel.fit(
    "[bold white]Granite 3.0 MoE[/bold white]  [dim]·[/dim]  [cyan]ibm-granite/granite-3.0-3b-a800m-instruct[/cyan]\n"
    f"[dim]Ready  ·  hook {'attached' if hook_attached else 'NOT found'}  ·  type [/dim][bold]exit[/bold][dim] to quit[/dim]",
    border_style="cyan",
    padding=(0, 2),
))
console.print(Rule(style="dim"))

# ==========================================
# VU Meter
# ==========================================
def vu_color(level):
    if level >= 4: return "bold red"
    if level == 3: return "bold yellow"
    return "bold green"

def make_vu_panel(weights, peak_lvls, gen_done):
    max_w = max(weights) if max(weights) > 1e-9 else 1e-9

    current_levels = []
    for w in weights:
        norm = w / max_w
        log_norm = np.log10(1 + norm * 9)
        current_levels.append(min(5, int(round(log_norm * 5))))

    for i, lvl in enumerate(current_levels):
        if lvl >= peak_lvls[i]:
            peak_lvls[i] = lvl
        elif not gen_done:
            peak_lvls[i] = max(0, peak_lvls[i] - 1)

    lines = []
    for group_start in [0, 20]:
        for row_level in range(5, 0, -1):
            line = Text()
            for i in range(group_start, group_start + 20):
                lvl = current_levels[i]
                pk  = peak_lvls[i]
                color = vu_color(row_level)
                if lvl >= row_level:
                    line.append("██", style=color)
                elif pk == row_level:
                    line.append("▀▀", style=color)
                else:
                    line.append("··", style="dim")
                line.append(" ")
            lines.append(line)

        lbl = Text()
        for i in range(group_start, group_start + 20):
            lbl.append(f"{i:02d} ", style="dim cyan")
        lines.append(lbl)

        if group_start == 0:
            lines.append(Text(""))

    return Panel(
        Group(*lines),
        title="[bold blue]Gating Network · Layer 20[/bold blue]",
        border_style="blue"
    )

# ==========================================
# Chat Loop
# ==========================================
messages = []

while True:
    try:
        console.print()
        user_input = console.input("[bold green] You ❯ [/bold green] ")

        if user_input.lower() in ["exit", "quit"]:
            console.print()
            console.print(Rule("[dim]session ended[/dim]", style="dim"))
            break
        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})

        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to("cuda")

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=1024,
            repetition_penalty=1.15,
        )

        layout = Layout()
        layout.split_row(
            Layout(name="experts", ratio=2),
            Layout(name="text",    ratio=3),
        )

        peak_levels = [0] * num_experts
        last_weights = np.ones(num_experts) / num_experts
        last_expert_update = 0

        layout["experts"].update(make_vu_panel(last_weights, peak_levels, False))
        layout["text"].update(Panel(
            "",
            title="[bold magenta] ◆ Granite[/bold magenta]",
            border_style="magenta"
        ))

        routing_queue.clear()
        thread = Thread(target=model.generate, kwargs=gen_kwargs)

        start_time = time.time()
        first_token_time = None
        response_text = ""

        thread.start()

        with Live(layout, refresh_per_second=10, console=console) as live:
            for chunk in streamer:
                if first_token_time is None:
                    first_token_time = time.time()

                response_text += chunk

                display_lines = response_text.split("\n")
                layout["text"].update(Panel(
                    "\n".join(display_lines[-40:]),
                    title="[bold magenta] ◆ Granite[/bold magenta]",
                    border_style="magenta"
                ))

                now = time.time()
                if routing_queue and (now - last_expert_update) >= 0.35:
                    last_expert_update = now
                    last_weights = routing_queue[-1]
                    routing_queue.clear()
                    layout["experts"].update(make_vu_panel(last_weights, peak_levels, False))

            layout["experts"].update(make_vu_panel(last_weights, peak_levels, True))
            time.sleep(1.5)

        thread.join()
        end_time = time.time()

        messages.append({"role": "assistant", "content": response_text})

        # ── Metrics
        ttft       = (first_token_time - start_time) if first_token_time else 0.0
        total_time = end_time - start_time
        tokens_out = len(tokenizer.encode(response_text, add_special_tokens=False))
        tps        = tokens_out / total_time if total_time > 0 else 0.0

        metrics = (
            f"[dim]TTFT [/dim][bold white]{ttft:.2f}s[/bold white]"
            f"[dim]   ·   TPS [/dim][bold white]{tps:.1f}[/bold white]"
            f"[dim]   ·   tokens [/dim][bold white]{tokens_out}[/bold white]"
        )
        console.print(Rule(metrics, style="dim", align="right"))

    except KeyboardInterrupt:
        console.print()
        console.print(Rule("[dim]interrupted[/dim]", style="dim"))
        break
