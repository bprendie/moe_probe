# Demo Sample Queries — Granite 3.0 MoE · Layer 20 Telemetry

Use these in `demo_deepdive.py`. Each section explains *why* the query is interesting for the
visualization before giving the prompt to paste.

---

## Category 1: Queries That Benefit From MoE Architecture

These are multi-domain prompts that naturally require the model to draw on distinct bodies of
knowledge within a single response. Watch the heatmap: you should see *different* experts dominating
each section rather than one expert carrying the whole run.

---

### 1a — Cross-Domain Analogy Bridge
The model has to hold three completely different knowledge domains simultaneously and find the
connection between them. Good for showing that MoE isn't just about "different topics" — it's about
routing *within* a single coherent argument.

```
Compare the CAP theorem in distributed systems to the Heisenberg uncertainty principle in quantum
mechanics. Then unify both concepts using an analogy drawn from jazz improvisation. Be technical
and precise in all three sections.
```

---

### 1b — Formal Mathematics → Working Code → Plain English
Forces a register shift from symbolic reasoning to executable logic to natural language explanation.
Each section should light up a different cluster of experts. The plain English section at the end
often shows a "cool down" pattern — watch for experts that only fire during the prose section.

```
Derive the closed-form solution to the Black-Scholes partial differential equation for a European
call option, showing all steps. Then implement the final pricing formula as a standalone Python
function with type hints. Finally, explain what the output number actually means to someone who
has never studied finance.
```

---

### 1c — Multi-Stack Technical Deep Dive
Three distinct engineering domains, all at expert level. Networking, cryptography, and systems
programming rarely share experts in this model. Good for demonstrating that a 3B parameter model
with sparse routing can cover this breadth at all.

```
Explain how TLS 1.3 session resumption works at the packet level, including the specific handshake
messages involved. Then describe how a hardware security module (HSM) would store and protect the
session ticket key. Finally, write a minimal Rust struct that represents a TLS session ticket with
appropriate field types and a derive macro for serialization.
```

---

## Category 2: "Jolt Factor" Queries

These are specifically designed to force the model to *pivot hard* mid-response — language changes,
domain expertise swaps, register shifts. These are your money shots for the VU meter display. The
transition moments between sections are when you'll see the meters spike into red as the gating
network re-routes.

The reference prompt from the live demo that maxed out the meters:

```
Explain the latency challenges of remote GPU rendering over a WAN, but write it entirely in
professional German. Then, write an x86 assembly routine that calculates an inverse square root.
Finally, write a three-line Haiku about a crashed server, written entirely in Japanese Kanji.
```

---

### 2a — Formal Proof → Street Register → Assembly
The tonal whiplash between academic mathematics and colloquial English is almost as jarring to the
router as a language change. The assembly section at the end should cause a visible re-routing event.

```
Write a formal mathematical proof that the square root of 2 is irrational, using standard proof
by contradiction notation. Then re-explain the same proof in the most casual street slang you can
manage, as if texting a friend. Finally, write an x86 NASM routine that checks whether an integer
input is a perfect square.
```

---

### 2b — Medical Latin → Nursery Rhyme → CUDA Kernel
Three sections with maximally different token distributions. Medical terminology, children's verse,
and GPU compute code have almost no vocabulary overlap — the routing network has to fully re-engage
for each one. The CUDA section in particular tends to fire a very specific cluster.

```
Describe the pathophysiology of septic shock using formal medical Latin terminology where possible.
Then restate the core concept as a simple nursery rhyme suitable for young children. Finally, write
a CUDA kernel in C++ that simulates a simplified agent-based infection spread model on a 1D grid.
```

---

### 2c — Legal English → Medieval English → Binary Encoding
This one is particularly fun for audiences: the "Medieval English" section is a real domain shift
that the model handles surprisingly well, and the transition into raw binary is abrupt enough that
you can usually see a physical VU meter spike on the display right as it starts.

```
Write a binding software license clause, in formal legal English, that prohibits reverse
engineering. Translate that same clause into Middle English as Chaucer might have written it.
Finally, encode the ASCII bytes of the first sentence of your legal clause as raw binary, outputting
only the 1s and 0s, space-separated by byte, nothing else.
```

---

### 2d — Academic Spanish → Python → Haiku in Arabic
Three-language pivot. Spanish and Arabic use different Unicode blocks entirely — the tokenizer
representation changes dramatically, which tends to cause strong routing events. Good closer for a
demo because the Arabic calligraphy in the terminal is visually striking.

```
Explain the concept of entropy in information theory in formal academic Spanish. Then implement
Shannon entropy calculation as a Python function. Finally, write a three-line Haiku about
information loss, composed entirely in Arabic script.
```

---

## Category 3: Queries Designed to Use as FEW Experts as Possible

These are monotonous, single-domain, structurally repetitive tasks. The goal is to show the
*other side* of the story: when the task is simple and homogeneous, the gating network settles into
a narrow, stable routing pattern. The heatmap should look like 2-3 hot rows and 37 cold ones.

Use these as a contrast demo — run one of these *after* a Category 2 prompt so the audience can see
the difference in expert utilization side by side.

---

### 3a — Pure Numeric Sequence (the floor case)
No vocabulary, no domain knowledge, just counting. Likely to engage the minimum possible number of
experts. Good baseline to show what "minimal routing" looks like.

```
Output the integers from 1 to 200, one per line. No punctuation, no labels, no explanation.
Begin immediately with the number 1.
```

---

### 3b — Repetitive Structured Data
Highly repetitive token patterns. The model locks into a rhythm quickly and stays there. Watch the
heatmap stabilize into a nearly static pattern after the first few rows.

```
Generate 40 rows of raw CSV data. Columns: ID, TIMESTAMP_UNIX, SENSOR_VALUE_FLOAT. Use sequential
integer IDs starting at 1000, realistic unix timestamps starting at 1700000000 incrementing by 60,
and random float sensor readings between 0.0 and 100.0 with 4 decimal places. Output only the CSV
rows, no header, no explanation, starting immediately.
```

---

### 3c — Single-Domain Monotone Explanation
One topic, one language, one register, no pivots. Compare its heatmap to the Category 2 prompts
and the audience will immediately see why MoE shines on diverse tasks.

```
Explain how a B-tree index works in a relational database. Cover insertion, deletion, and search.
Write at a technical level appropriate for a senior software engineer. Use only plain English prose,
no code, no lists, no headers.
```

---

### 3d — Templated Identity Generation
The most repetitive possible output: the model is essentially a template engine. Expect very low
expert diversity and a heatmap that looks almost like a barcode — the same 2-3 experts firing on
every single token.

```
Generate 30 fake but realistic-looking user records in this exact format, one per line, no header,
no explanation:
FIRSTNAME LASTNAME | email@domain.com | +1-XXX-XXX-XXXX | CITY, ST

Start immediately.
```
