# Talk Track — MoE Live Demo
## AI Tinkerers NH

> **Room read:** Smart crowd, high bravado, lots of buzzword fluency. Goal is not to impress them —
> it's to give them a mental model they'll actually be able to *use* to explain this to someone else
> tomorrow. The demo does the heavy lifting. Your job is to narrate what they're already watching.
>
> No pitches. No products. Pure architecture.

---

## 1. The Opening Hook (~2 min)

*Before you touch the keyboard.*

"Everyone in this room has said the words 'Mixture of Experts' in the last six months. Maybe in a
meeting, maybe in a Discord. And you were probably right to use it. But I want to ask a different
question tonight — not *what* is a MoE model, but *what is it actually doing right now, in
real-time, on this machine, for this specific sentence I'm about to type.*

Because there's a difference between knowing the name of something and understanding it well enough
to make an architectural decision with it. Tonight we're going to close that gap."

---

## 2. What a Transformer Layer Actually Is (~5 min)

*Do not skip this. Most of the room thinks they know this. Most of them don't — not at this level.*

"Before we can talk about MoE, we need to agree on what a transformer layer actually looks like
under the hood. Because MoE is not a new model type — it's a surgical replacement of one specific
component inside a standard transformer.

A transformer is a stack of repeated layers. Each layer has two major components:

**Component 1: Multi-Head Self-Attention.**
This is the part everyone talks about. Attention lets every token look at every other token in the
context and figure out which ones are relevant to it. It's how the model knows that 'it' in a
sentence refers to the noun three words earlier. Attention is inherently relational — it's about
*where to look.*

**Component 2: The Feed-Forward Network (FFN).**
After attention, each token goes through a feed-forward block independently. No token looks at any
other token here — it's just a two-layer MLP applied position-wise. This block is larger than it
looks: in most models it's 4x the width of the model's hidden dimension. In a 7B parameter dense
model, roughly two-thirds of the parameters live in these FFN blocks.

This is also where the model's *factual knowledge* is believed to live. Researchers have found they
can probe specific factual associations directly from FFN weights. The attention mechanism finds
*where* to look — the FFN block is *what the model knows.*

In a dense model, every token passes through both of these components, fully, every single layer,
every single forward pass. Every parameter fires. Every time."

---

## 3. The Dense Model Problem — Every Neuron, Every Token (~3 min)

"Let's think about what that means computationally.

GPT-4 is estimated at roughly 1.8 trillion parameters. When you hit send on a message, every one of
those parameters participates in computing your response — even if your message is 'write a haiku.'
Even if the model's knowledge of 15th century Ottoman trade policy, advanced organic chemistry, or
Swahili grammar is completely irrelevant to the task.

This is the core inefficiency of dense models: **uniform compute regardless of task complexity.**
You are paying the full inference cost every time, for every token, regardless of how much of the
model's capacity is actually useful for what you asked.

At the scale of production API traffic, this is brutal. It's also why inference is estimated to
account for roughly 90% of a model's total lifetime compute cost — not training. Training is a
one-time event. Inference runs forever."

---

## 4. What MoE Does — The Surgical Swap (~5 min)

*This is the core technical section. Slow down here.*

"Mixture of Experts fixes this by replacing the FFN block with a more interesting construction.
Instead of one single FFN that every token goes through, you build **N independent FFNs** —
these are your experts. In the model we're running tonight, Granite 3.0 from IBM, that's **40
experts per MoE layer.**

Then you add one new component that doesn't exist in a standard transformer: **the gating network,
also called the router.**

Here's what the router does for every single token:

1. It takes the token's current hidden state vector — the representation built up so far by the
   attention mechanism.
2. It multiplies that vector by a learned weight matrix to produce a set of **logits** — one score
   per expert.
3. It runs a **softmax** over those 40 logits to get a probability distribution.
4. It selects the **top-2** experts by probability weight.
5. It runs the token through *only those two experts*, weights their outputs by their softmax
   scores, and sums the results.

The other 38 experts are not called. They do not execute. Their parameters participate in
*loading* into VRAM, but not in *computing* this token.

This is **sparse activation.** The model has 3 billion parameters, but for any given token, it's
only actively computing through a fraction of them.

The critical implication: you get the *capacity* of a large model — 40 different learned
representations of knowledge — at the *compute cost* of running 2 of them. That's not a
marketing claim. That's arithmetic."

### The Parameter Paradox

"There's a subtlety here worth understanding. Mixtral 8x7B — one of the best-known MoE models —
has 56 billion total parameters. But it runs at the inference speed of a 12-13B dense model.

Why? Because the non-expert layers — attention, embeddings, layer norms — are shared and identical
to a standard transformer. And per token, only 2 of the 8 expert FFNs activate. So the *active
parameter count* per forward pass is much smaller than the *total parameter count.*

The cost is VRAM: you still have to load all 56B parameters onto your GPU(s) even though you're
only using a fraction of them at any moment. MoE trades **memory** for **compute efficiency.** On
hardware with enough VRAM, it's a compelling trade."

### What the Experts Actually Learn

"Here's the part that surprised researchers: the experts aren't manually assigned domains. There's
no 'this expert handles German' or 'this expert handles math.' The specialization emerges entirely
from training.

The router learns which experts to call through gradient descent, the same way everything else in
the model is learned. And through that process, the experts tend to develop soft specializations —
certain syntactic patterns, certain knowledge domains, certain output formats seem to consistently
route to the same subset of experts.

Which is exactly what we're about to watch happen."

---

## 5. The Demo — Live Narration (~10-15 min)

*Open the terminal. Run `demo_deepdive.py`.*

### On the VU Meters:

"I've attached a PyTorch forward hook directly to the router's linear layer in Layer 20. Every
time a token is generated, I capture the full 40-dimensional softmax output before the top-2
selection happens — the raw routing weights. That's what you're seeing on these VU meters.

The height and color of each meter corresponds to how much probability weight the router assigned
to that expert for the current token. Green is low weight, yellow is medium, red means the router
is heavily favoring this expert. The peak hold markers show the highest weight that expert has
reached during this run.

We're watching the gating network make decisions in real time, one token at a time.

Watch what happens at the transition points when I run this prompt."

*[Fire a Category 2 / jolt-factor prompt.]*

"There — did you see the routing pattern shift? When we crossed from one domain into another, a
different cluster of experts lit up. The router detected the change in context and re-weighted.
That's the specialization emerging live. The model doesn't have a manual switch — the gating
network just learned that certain patterns of hidden state activate certain experts, and you're
watching it execute that learned routing."

### On the Peak Hold:

"The `▀▀` marker is the peak hold — same concept as a VU meter on a mixing console. A transient
hit on an expert will leave that marker floating at the peak position, drifting down slowly over
subsequent tokens. When generation finishes, I freeze all of them. The frozen state is a snapshot
of which experts were most heavily used across the entire run."

---

## 6. The Heatmap (~3 min)

*After generation ends and the heatmap renders.*

"This is the full run as a 2D heatmap. X-axis is the token timeline — left is token 0, right is
the last token. Y-axis is all 40 experts. Color intensity is activation frequency within each
time window: dark is cold, red is hot.

**Hot rows across the whole timeline** — these are backbone experts. They fire on nearly every
token regardless of what the content is. They're likely handling the universal mechanics of
language generation: next-token probability, basic grammatical structure, output formatting.

**Rhythmically patterned rows** — regular spacing, almost metronomic. This is the *structure* of
the output bleeding into the routing. If we generated CSV data, experts that handle delimiter
tokens fire on every comma and newline. The router has learned that certain token types
consistently need certain experts.

**Flash-and-die at the left edge** — experts that are bright at token 0-10 and then go completely
dark. These fired while the model was orienting to the task — reading the prompt type, setting up
the output format — and were never needed again once generation was underway.

**Delayed onset** — experts that are cold early and then warm up. Often corresponds to a phase
transition: the model finishing its 'setup' generation and shifting into the specific content the
task requires.

Run a monotonous prompt — pure CSV, or counting integers — and this heatmap collapses to 2-3
bright rows and 37 dark ones. Run a multi-domain jolt prompt and you'll see a much richer pattern.
The heatmap is a fingerprint of the task's internal structure."

---

## 7. The Introspection — Optional (~2 min intro)

*Pause. Ask the room.*

"Now here's where it gets a little meta. I've computed per-expert activation statistics from that
run — total firing rate, and a breakdown across three phases of the generation: early, mid, and
late. I'm going to feed that data back to Granite, along with the original prompt, and ask it to
analyze its own routing behavior.

It doesn't have direct introspective access to its weights. It's pattern-matching statistical
evidence the same way we are. But it's often surprisingly coherent — it can correctly identify
which experts look like backbone nodes vs. task-specific specialists, and sometimes correctly
infers *why* certain experts surged in specific phases.

Worth noting: the deepdive generation itself will also go through the router and show up on the
meters. You're watching the model think about its own thinking, and that process has its own
routing signature."

*[Run if the room wants it. Ask: "Shall we analyze this heatmap together?"]*

---

## 8. The Takeaway (~2 min)

*Close it out.*

"What I want you to leave with tonight is a concrete mental model, not a slogan.

A transformer has two core components per layer: **attention**, which figures out what to look at,
and the **FFN**, which applies what the model knows. In a dense model, both fire fully for every
token. In a MoE model, the FFN is replaced by 40 specialized sub-networks and a learned router
that picks 2 per token.

The result is a model that can hold the capacity of a much larger system while spending the compute
of a much smaller one — and the routing pattern it generates is a live, observable signature of
what the model understands itself to be doing.

That's not an abstraction. You just watched it happen.

The routing pattern is a fingerprint of the task. Every token, 40 options, pick 2. And now you
can see exactly how it chooses."

---

## Appendix: Prompt Cheat Sheet

See `sample_queries.md` for the full list. Quick picks for the room:

| Goal | Prompt category |
|---|---|
| Show MoE breadth | Cat 1 — Cross-domain analogy or Black-Scholes |
| Max VU meter drama | Cat 2 — German/Assembly/Japanese (the reference) |
| Show contrast / calm run | Cat 3 — Integer sequence or CSV generation |
| Trigger the introspection | Any Cat 2 prompt, then `y` at the prompt |
| Side-by-side comparison | Cat 3 first, then Cat 2 — heatmap contrast is striking |
