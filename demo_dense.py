import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import time
import numpy as np
import random
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console, Group
from rich.text import Text

console = Console()

# ==========================================
# 1. Setup Model & Hardware
# ==========================================
# Qwen 2.5 3B is roughly 3.09B parameters.
# In bfloat16, this takes ~6.2GB of VRAM.
model_id = "Qwen/Qwen2.5-3B-Instruct"

console.print(f"[bold cyan][System][/bold cyan] Loading Dense Model ({model_id}) into VRAM...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda"
    )
except Exception as e:
    console.print(f"[bold red][Error][/bold red] Failed to load model: {e}")
    exit(1)

# Logic: 3.09B parameters / 200M per meter = ~15.5 meters, rounding to 16.
num_meters = 16
total_params = "3.09B"
active_params = "3.09B (100%)"

console.print("\n[bold green][System][/bold green] Initialization complete. Dense Inference engine ready.")
time.sleep(0.5)

# ==========================================
# VU Meter Rendering (Simulated for Dense)
# ==========================================
def vu_color(level):
    if level >= 4: return "bold red"
    if level == 3: return "bold yellow"
    return "bold green"

def make_vu_panel(is_generating, peak_lvls):
    """
    Render 16 parameter blocks as VU meters.
    In a dense model, they ALL fire at 100% during every forward pass.
    """
    current_levels = []
    for i in range(num_meters):
        if is_generating:
            # Jitter near the top to show "activity"
            lvl = random.randint(4, 5)
        else:
            lvl = 0
        current_levels.append(lvl)

    # Update peak holds
    for i, lvl in enumerate(current_levels):
        if lvl >= peak_lvls[i]:
            peak_lvls[i] = lvl
        elif not is_generating:
            peak_lvls[i] = max(0, peak_lvls[i] - 1)

    lines = []
    # Render top-down: level 5 is the top row
    for row_level in range(5, 0, -1):
        line = Text()
        for i in range(num_meters):
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

    # Labeling blocks by parameter count
    lbl = Text()
    for i in range(num_meters):
        lbl.append(f"{(i+1)*0.2:.1f}G", style="dim cyan" if i % 2 == 0 else "dim blue")
        lbl.append(" ")
    lines.append(lbl)

    stats = Text.assemble(
        ("\nArchitecture: ", "bold white"), ("Dense Transformer\n", "white"),
        ("Total Params: ", "bold white"), (f"{total_params}\n", "yellow"),
        ("Active Params: ", "bold white"), (f"{active_params}", "bold red"),
        (" (Uniform Compute)", "dim red")
    )
    lines.append(Text(""))
    lines.append(stats)

    return Panel(
        Group(*lines),
        title="[bold red]Dense Parameter Utilization · All Layers[/bold red]",
        border_style="red"
    )

# ==========================================
# 3. The Interactive Pulse Loop
# ==========================================
while True:
    try:
        console.print("\n" + "="*80)
        user_prompt = console.input("[bold magenta]Enter Audience Prompt (or type 'exit'): [/bold magenta]")

        if user_prompt.lower() in ['exit', 'quit']:
            console.print("[bold red]Shutting down dense inference engine.[/bold red]")
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
            Layout(name="parameters", ratio=2),
            Layout(name="text",    ratio=1)
        )

        generated_text = ""
        token_count = 0
        peak_levels = [0] * num_meters
        is_generating = True

        layout["parameters"].update(make_vu_panel(True, peak_levels))
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

                # Keep meters pinned while generating
                layout["parameters"].update(make_vu_panel(True, peak_levels))

                if len(display_lines) > 28:
                    break

            is_generating = False
            # Hold the red for a moment
            layout["parameters"].update(make_vu_panel(True, peak_levels))
            time.sleep(2.0)

        thread.join()

    except KeyboardInterrupt:
        console.print("\n[bold red]Operation cancelled by user.[/bold red]")
        break
