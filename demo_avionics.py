import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import time
import numpy as np
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console, Group
from rich.text import Text

console = Console()

# ==========================================
# 1. Setup Model & Hardware
# ==========================================
console.print("[bold cyan][System][/bold cyan] Loading Granite 3.0 MoE (3B-A800M) into VRAM...")
model_id = "ibm-granite/granite-3.0-3b-a800m-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="cuda"
)

num_experts = getattr(model.config, "num_local_experts", 40)

# ==========================================
# 2. The Vector-Level Telemetry Hook
# ==========================================
routing_queue = []
routing_history = []

def router_hook(module, args, output):
    logits = output[0] if isinstance(output, tuple) else output
    probs = torch.nn.functional.softmax(logits, dim=-1)

    current_token_probs = probs[-1, :].detach().float().cpu().numpy()
    routing_queue.append(current_token_probs)

    top_2 = current_token_probs.argsort()[-2:][::-1]
    routing_history.append(top_2.tolist())

# Deep layer isolation (Layer 20)
target_layer = "layers.20.block_sparse_moe.router"
hook_attached = False

for name, module in model.named_modules():
    if target_layer in name and isinstance(module, torch.nn.Linear):
        if module.out_features == num_experts:
            module.register_forward_hook(router_hook)
            console.print(f"[bold cyan][System][/bold cyan] Attached telemetry hook to tensor: {name}")
            hook_attached = True
            break

if not hook_attached:
    console.print(f"[bold red][System Error][/bold red] Could not find target layer: {target_layer}")

console.print("\n[bold green][System][/bold green] Initialization complete. Inference engine ready.")
time.sleep(0.5)

# ==========================================
# VU Meter Rendering
# ==========================================
def vu_color(level):
    """Color by level: 1-2 green, 3 yellow, 4-5 red."""
    if level >= 4: return "bold red"
    if level == 3: return "bold yellow"
    return "bold green"

def make_vu_panel(weights, peak_lvls, gen_done):
    """
    Render 40 experts as two rows of 20 VU meters, 5 levels tall.
    Logarithmically normalized so the hottest expert always pegs the top.
    peak_lvls is mutated in place.
    """
    max_w = max(weights) if max(weights) > 1e-9 else 1e-9

    current_levels = []
    for w in weights:
        norm = w / max_w
        log_norm = np.log10(1 + norm * 9)   # log10 scale: [0,1] -> [0,1]
        current_levels.append(min(5, int(round(log_norm * 5))))

    # Update peak holds
    for i, lvl in enumerate(current_levels):
        if lvl >= peak_lvls[i]:
            peak_lvls[i] = lvl
        elif not gen_done:
            peak_lvls[i] = max(0, peak_lvls[i] - 1)

    lines = []
    for group_start in [0, 20]:
        # Render top-down: level 5 is the top row
        for row_level in range(5, 0, -1):
            line = Text()
            for i in range(group_start, group_start + 20):
                lvl = current_levels[i]
                pk = peak_lvls[i]
                color = vu_color(row_level)
                if lvl >= row_level:
                    line.append("██", style=color)
                elif pk == row_level:
                    line.append("▀▀", style=color)   # peak hold marker
                else:
                    line.append("··", style="dim")
                line.append(" ")
            lines.append(line)

        # Expert number labels
        lbl = Text()
        for i in range(group_start, group_start + 20):
            lbl.append(f"{i:02d} ", style="dim cyan")
        lines.append(lbl)

        if group_start == 0:
            lines.append(Text(""))   # blank row between the two groups

    return Panel(
        Group(*lines),
        title="[bold blue]Live Gating Network · Layer 20[/bold blue]",
        border_style="blue"
    )

# ==========================================
# 3. The Interactive Pulse Loop
# ==========================================
while True:
    try:
        console.print("\n" + "="*80)
        user_prompt = console.input("[bold magenta]Enter Audience Prompt (or type 'exit'): [/bold magenta]")

        if user_prompt.lower() in ['exit', 'quit']:
            console.print("[bold red]Shutting down inference engine.[/bold red]")
            break
        if not user_prompt.strip():
            continue

        messages = [{"role": "user", "content": user_prompt}]
        formatted_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_chat, return_tensors="pt").to("cuda")

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=250,
            repetition_penalty=1.3,
        )

        layout = Layout()
        layout.split_row(
            Layout(name="experts", ratio=3),
            Layout(name="text",    ratio=1)
        )

        generated_text = ""
        start_time = time.time()
        token_count = 0
        routing_queue.clear()
        routing_history.clear()
        last_expert_update = 0
        peak_levels = [0] * 40
        last_weights = np.ones(num_experts) / num_experts

        # Seed both panels before generation starts
        layout["experts"].update(make_vu_panel(last_weights, peak_levels, False))
        layout["text"].update(Panel("", title="[bold green]Output[/bold green]", border_style="green"))

        with Live(layout, refresh_per_second=10) as live:
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            start_time = time.time()

            for new_text in streamer:
                generated_text += new_text
                token_count += 1

                elapsed = time.time() - start_time
                tps = token_count / elapsed if elapsed > 0 else 0

                # Right panel: narrow streaming text
                display_lines = generated_text.split("\n")
                clamped_text = "\n".join(display_lines[-30:])
                layout["text"].update(Panel(
                    clamped_text,
                    title=f"[bold green]Output[/bold green] [yellow]{tps:.1f} TPS[/yellow]",
                    border_style="green"
                ))

                # Left panel: VU meters, throttled
                now = time.time()
                if routing_queue and (now - last_expert_update) >= 0.35:
                    last_expert_update = now
                    last_weights = routing_queue[-1]
                    routing_queue.clear()
                    layout["experts"].update(make_vu_panel(last_weights, peak_levels, False))

                # Stop consuming once the text panel is full — content doesn't matter
                if len(display_lines) > 28:
                    break

            # Generation complete — freeze peaks and hold for audience
            layout["experts"].update(make_vu_panel(last_weights, peak_levels, True))
            time.sleep(2.0)

        thread.join()

        # ==========================================
        # 4. Post-Run Telemetry Scatterplot
        # ==========================================
        if routing_history:
            active_experts = sorted(list(set(e for step in routing_history for e in step)))
            plot_width = 80
            num_tokens = len(routing_history)

            console.print("\n[bold magenta]=== Deep Layer MoE Telemetry Timeline (Layer 20) ===[/bold magenta]")
            console.print(f"X-Axis: Tokens (0 -> {num_tokens}) | Y-Axis: Active Experts")
            console.print("-" * 100)

            for expert in active_experts:
                row_chars = []
                for i in range(plot_width):
                    start_idx = int(i * num_tokens / plot_width)
                    end_idx = int((i + 1) * num_tokens / plot_width)
                    if end_idx == start_idx: end_idx = start_idx + 1

                    window_active = any(expert in routing_history[t] for t in range(start_idx, end_idx) if t < num_tokens)
                    row_chars.append("█" if window_active else "·")

                row_str = "".join(row_chars)
                console.print(f" Expert {expert:02d} | [cyan]{row_str}[/cyan]")
            console.print("-" * 100)

    except KeyboardInterrupt:
        console.print("\n[bold red]Operation cancelled by user. Shutting down.[/bold red]")
        break
