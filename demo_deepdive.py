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
# VU Meter Rendering (live display)
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
                pk = peak_lvls[i]
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
        title="[bold blue]Live Gating Network · Layer 20[/bold blue]",
        border_style="blue"
    )

# ==========================================
# Heatmap Rendering (post-run)
# ==========================================
PLOT_WIDTH = 60

def _shade(intensity: float) -> tuple:
    if intensity == 0:
        return "··", "dim"
    elif intensity < 0.20:
        return "░░", "dim green"
    elif intensity < 0.45:
        return "▒▒", "green"
    elif intensity < 0.70:
        return "▓▓", "yellow"
    elif intensity < 0.88:
        return "██", "bold yellow"
    else:
        return "██", "bold red"

def render_heatmap(routing_history, num_experts=40):
    num_tokens = len(routing_history)

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

    global_max = freq.max() or 1.0
    norm = freq / global_max

    console.print()
    console.print("[bold magenta]=== MoE Expert Activation Heatmap · Layer 20 ===[/bold magenta]")
    console.print(
        f"[dim]X: token timeline (0 → {num_tokens})  "
        f"Y: expert  "
        f"color: activation frequency  (dim=cold  ░▒▓█=hot)[/dim]"
    )

    prefix = "  E00 "
    tick   = f"0{'':{PLOT_WIDTH * 2 - len(str(num_tokens)) - 1}}{num_tokens}"

    for group_start in [0, 20]:
        console.print()
        console.print(Text(" " * len(prefix) + tick, style="dim"))

        for expert in range(group_start, min(group_start + 20, num_experts)):
            line = Text(f"  E{expert:02d} ", style="bold cyan")
            for col in range(PLOT_WIDTH):
                char, style = _shade(norm[expert, col])
                line.append(char, style=style)
            console.print(line)

        console.print(Text(" " * len(prefix) + tick, style="dim"))

    console.print()

# ==========================================
# Deep Dive: build stats + prompt
# ==========================================
def build_deepdive_prompt(user_prompt, routing_history, num_experts=40):
    num_tokens = len(routing_history)
    early_end  = num_tokens // 4
    late_start = 3 * num_tokens // 4

    counts       = [0] * num_experts
    early_counts = [0] * num_experts
    mid_counts   = [0] * num_experts
    late_counts  = [0] * num_experts

    for t, experts in enumerate(routing_history):
        for e in experts:
            if e < num_experts:
                counts[e] += 1
                if t < early_end:
                    early_counts[e] += 1
                elif t >= late_start:
                    late_counts[e] += 1
                else:
                    mid_counts[e] += 1

    early_tokens = max(early_end, 1)
    mid_tokens   = max(late_start - early_end, 1)
    late_tokens  = max(num_tokens - late_start, 1)

    # Top 15 most active experts
    ranked = sorted(range(num_experts), key=lambda i: counts[i], reverse=True)
    active = [i for i in ranked if counts[i] > 0][:15]

    header = f"{'Expert':<8} {'Total%':>7} {'Early%':>7} {'Mid%':>7} {'Late%':>7}  Pattern"
    divider = "-" * 55
    rows = [header, divider]

    for e in active:
        total_rate = counts[e] / num_tokens * 100
        early_rate = early_counts[e] / early_tokens * 100
        mid_rate   = mid_counts[e]   / mid_tokens   * 100
        late_rate  = late_counts[e]  / late_tokens  * 100

        rates = [early_rate, mid_rate, late_rate]
        spread = max(rates) - min(rates)
        if spread < 8:
            pattern = "uniform"
        elif early_rate == max(rates) and late_rate == min(rates):
            pattern = "front-loaded"
        elif late_rate == max(rates) and early_rate == min(rates):
            pattern = "back-loaded"
        elif mid_rate == max(rates):
            pattern = "mid-peak"
        elif early_rate == min(rates):
            pattern = "delayed-onset"
        else:
            pattern = "variable"

        rows.append(
            f"  E{e:02d}   {total_rate:>7.1f} {early_rate:>7.1f} {mid_rate:>7.1f} {late_rate:>7.1f}  {pattern}"
        )

    stats_block = "\n".join(rows)

    return f"""You are a Mixture of Experts language model (IBM Granite 3.0 MoE, 40 experts per layer). \
The table below shows activation statistics for your own Layer 20 gating network while you generated \
a response. Each token generation selects exactly 2 experts. The table shows how often each of the top \
15 most-activated experts fired, broken into three equal phases of the generation run.

Task you were responding to:
"{user_prompt}"

Layer 20 expert activation statistics ({num_tokens} tokens total):
{stats_block}

Analyze this routing data in 2-3 tight paragraphs. Identify which experts look like universal \
backbone experts vs task-specific specialists. Note any interesting phase patterns — did activation \
shift as the response moved through different sections of the task? Given the multi-part nature of \
the task (it had distinct sections requiring different knowledge domains), speculate on what \
computational roles the most distinctive experts might be serving. Be specific."""


def render_deepdive(user_prompt, routing_history, num_experts=40):
    console.print("\n[bold cyan]━━━ Routing Deep Dive · Granite Self-Analysis ━━━[/bold cyan]")
    console.print("[dim]Querying model for self-analysis of its own routing behavior...[/dim]\n")

    prompt = build_deepdive_prompt(user_prompt, routing_history, num_experts)
    dd_messages   = [{"role": "user", "content": prompt}]
    dd_formatted  = tokenizer.apply_chat_template(dd_messages, tokenize=False, add_generation_prompt=True)
    dd_inputs     = tokenizer(dd_formatted, return_tensors="pt").to("cuda")
    dd_streamer   = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    dd_kwargs     = dict(**dd_inputs, streamer=dd_streamer, max_new_tokens=450, repetition_penalty=1.3)

    routing_queue.clear()

    thread = Thread(target=model.generate, kwargs=dd_kwargs)
    thread.start()

    analysis = ""
    with Live(console=console, refresh_per_second=10) as live:
        for chunk in dd_streamer:
            analysis += chunk
            live.update(Panel(
                analysis,
                title="[bold cyan]Expert Routing Analysis[/bold cyan]",
                border_style="cyan"
            ))

    thread.join()
    console.print()

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

                display_lines = generated_text.split("\n")
                clamped_text = "\n".join(display_lines[-30:])
                layout["text"].update(Panel(
                    clamped_text,
                    title=f"[bold green]Output[/bold green] [yellow]{tps:.1f} TPS[/yellow]",
                    border_style="green"
                ))

                now = time.time()
                if routing_queue and (now - last_expert_update) >= 0.35:
                    last_expert_update = now
                    last_weights = routing_queue[-1]
                    routing_queue.clear()
                    layout["experts"].update(make_vu_panel(last_weights, peak_levels, False))

                # Stop consuming once the text panel is full — content doesn't matter
                if len(display_lines) > 28:
                    break

            # Freeze peaks and hold for audience
            layout["experts"].update(make_vu_panel(last_weights, peak_levels, True))
            time.sleep(2.0)

        thread.join()  # ensure generation is fully finished before next prompt

        # ==========================================
        # 4. Post-Run Heatmap
        # ==========================================
        # Snapshot routing_history before deepdive pollutes it
        history_snapshot = [row[:] for row in routing_history]

        if history_snapshot:
            render_heatmap(history_snapshot, num_experts=num_experts)

        # ==========================================
        # 5. Deep Dive Self-Analysis (optional)
        # ==========================================
        if history_snapshot:
            answer = console.input("\n[bold magenta]Shall we analyze this heatmap together? (y/n): [/bold magenta]").strip().lower()
            if answer == "y":
                render_deepdive(user_prompt, history_snapshot, num_experts=num_experts)

    except KeyboardInterrupt:
        console.print("\n[bold red]Operation cancelled by user. Shutting down.[/bold red]")
        break
