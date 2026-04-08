# Reading the MoE Telemetry Scatterplot

## The "backbone" experts — 38 and 11
Solid bars, wall-to-wall. These fire on nearly every token regardless of content. They're doing the universal heavy lifting — basic syntax prediction, token probability, the "always-on" substrate of the model. Think of them as the master bus on your mixing console: everything runs through them.

## The "rhythm" experts — 07, 09, 17, 27, 30
Notice the regular, almost metronomic spacing. This is the CSV structure itself — the commas, the newlines, the column separators. The model has internalized the repeating grammar of the format and certain experts are literally taking turns on the beat, one per structural delimiter.

## The "flash and die" specialists — 05, 12, 26
Active only in the first ~10 tokens, then completely silent for the rest of the 800-token run. These fired during header row generation (`HOST_UUID, UPTIME_SECONDS...`) — a syntactically different task from generating numeric data rows — and were never needed again. The model routed differently for that one moment, then settled into a new stable pattern.

## Expert 00's delayed entry
Silent at first, then switches on around token 20 and stays hot. That's almost certainly the model transitioning from "parsing the prompt / writing the header" into "generating repetitive numeric data." A different cognitive mode, a different expert.

---

The headline for the audience: **you're not watching a language model predict text — you're watching it route computation through specialized subnetworks in real time, and the routing pattern is a fingerprint of the task's internal structure.**
