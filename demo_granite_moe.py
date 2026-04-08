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
model_id = "ibm-granite/granite-4.0-tiny-preview"
console.print(f"[bold cyan][System][/bold cyan] Loading Granite 4.0 Tiny MoE ({model_id}) into VRAM...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True
)

num_experts = 62  # As discovered
top_k = 6         # As discovered

# ==========================================
# 2. The Vector-Level Telemetry Hook
# ==========================================
routing_queue = []
routing_history = []

def router_hook(module, args, output):
    # For Granite 4.0 Hybrid, the output structure might be slightly different 
    # than 3.0, but usually the first element is the logits.
    logits = output[0] if isinstance(output, tuple) else output
    probs = torch.nn.functional.softmax(logits, dim=-1)

    current_token_probs = probs[-1, :].detach().float().cpu().numpy()
    routing_queue.append(current_token_probs)

    # Top-6 for Granite 4.0
    active = current_token_probs.argsort()[-top_k:][::-1]
    routing_history.append(active.tolist())

# Target layer discovered via inspect script
target_layer = "layers.20.block_sparse_moe.router.layer"
hook_attached = False

for name, module in model.named_modules():
    if target_layer in name:
        module.register_forward_hook(router_hook)
        console.print(f"[bold cyan][System][/bold cyan] Attached telemetry hook to Granite 4.0 router: {name}")
        hook_attached = True
        break

if not hook_attached:
    console.print(f"[bold red][System Error][/bold red] Could not find target layer: {target_layer}")

console.print("\n[bold green][System][/bold green] Initialization complete. Granite 4.0 engine ready.")
time.sleep(0.5)

# ==========================================
# VU Meter Rendering (Updated for 62 experts)
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
        if i >= len(peak_lvls): break
        if lvl >= peak_lvls[i]:
            peak_lvls[i] = lvl
        elif not gen_done:
            peak_lvls[i] = max(0, peak_lvls[i] - 1)

    lines = []
    # 62 experts: 2 rows of 31
    for group_start in [0, 31]:
        for row_level in range(5, 0, -1):
            line = Text()
            for i in range(group_start, group_start + 31):
                if i >= num_experts: break
                lvl = current_levels[i]
                pk = peak_lvls[i]
                color = vu_color(row_level)
                if lvl >= row_level:
                    line.append("█", style=color)
                elif pk == row_level:
                    line.append("▀", style=color)
                else:
                    line.append("·", style="dim")
            lines.append(line)

        lbl = Text()
        for i in range(group_start, group_start + 31):
            if i >= num_experts: break
            # Single digit label for density
            lbl.append(f"{i % 10}", style="dim cyan")
        lines.append(lbl)

        if group_start == 0:
            lines.append(Text(""))

    return Panel(
        Group(*lines),
        title=f"[bold blue]Granite 4.0 Hybrid MoE · 62 Experts (Top-{top_k})[/bold blue]",
        border_style="blue"
    )

# ==========================================
# 3. Interactive Loop
# ==========================================
while True:
    try:
        console.print("\n" + "="*80)
        user_prompt = console.input("[bold magenta]Enter Audience Prompt (or type 'exit'): [/bold magenta]")

        if user_prompt.lower() in ['exit', 'quit']:
            console.print("[bold red]Shutting down Granite 4.0 engine.[/bold red]")
            break
        if not user_prompt.strip():
            continue

        messages = [{"role": "user", "content": user_prompt}]
        formatted_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_chat, return_tensors="pt").to("cuda")

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=250, repetition_penalty=1.2)

        layout = Layout()
        layout.split_row(Layout(name="experts", ratio=3), Layout(name="text", ratio=1))

        generated_text = ""
        routing_queue.clear()
        routing_history.clear()
        peak_levels = [0] * num_experts
        last_weights = np.ones(num_experts) / num_experts

        with Live(layout, refresh_per_second=10) as live:
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            start_time = time.time()
            token_count = 0

            for new_text in streamer:
                generated_text += new_text
                token_count += 1
                elapsed = time.time() - start_time
                tps = token_count / elapsed if elapsed > 0 else 0

                clamped_text = "\n".join(generated_text.split("\n")[-30:])
                layout["text"].update(Panel(clamped_text, title=f"[yellow]{tps:.1f} TPS[/yellow]", border_style="green"))

                if routing_queue:
                    last_weights = routing_queue[-1]
                    routing_queue.clear()
                    layout["experts"].update(make_vu_panel(last_weights, peak_levels, False))

                if len(generated_text.split("\n")) > 28: break

            layout["experts"].update(make_vu_panel(last_weights, peak_levels, True))
            time.sleep(2.0)

        thread.join()

    except KeyboardInterrupt:
        console.print("\n[bold red]Operation cancelled.[/bold red]")
        break
