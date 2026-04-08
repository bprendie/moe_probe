## Capability Demo: Five Business Tasks

The goal isn't to match a cloud model output-for-output. The goal is to show that a 3B model running on local silicon handles real business tasks — the kind an IT or ops team encounters daily — accurately and instantly. No API call. No round-trip. No data leaving the building.

Each prompt is a real scenario. The output speaks for itself.

---

### **Task 1: IT Support Ticket Triage**
*Capability: Classification and structured extraction*

A support desk receives hundreds of tickets. Routing them manually is slow. This shows the model reading natural language and producing a structured, actionable JSON response — the kind a ticketing system could ingest directly.

> **Prompt:**
> "A user submitted the following IT support ticket: 'My VDI session keeps disconnecting every 20 minutes. I'm on Windows 11, connected via Citrix. Started happening after the update last Tuesday. I'm losing unsaved work each time.' Classify it as JSON with fields: `category` (string), `priority` (integer, 1=critical to 5=low), `suggested_team` (string), and `one_line_summary` (string). Output ONLY the JSON."

---

### **Task 2: SQL Session Report**
*Capability: Data querying*

Confirmed working in testing. Simple aggregation — no dialect traps, no complex set logic. The kind of query a team lead asks for on a Monday morning.

> **Prompt:**
> "Write a PostgreSQL query for a table `vdi_sessions` with columns `user_id` (int), `login_time` (timestamp), and `logout_time` (timestamp). Return the `user_id` and the average session duration in minutes, aliased as `avg_min`. Filter for users with more than 5 sessions in the last 30 days using `INTERVAL '30 days'`. Output ONLY the SQL."

---

### **Task 3: jq Filter for API Output**
*Capability: Data transformation*

Confirmed working in testing. Ops engineers pipe JSON through jq constantly — monitoring APIs, log exporters, service health endpoints. Single-line output, sub-second response.

> **Prompt:**
> "Write a jq filter expression that takes an array of objects with fields `user_id` (string), `status` (string), and `duration_minutes` (number). Return only objects where `status` equals `\"active\"` and `duration_minutes` is greater than 30. Output ONLY the jq filter string, no explanation, no code block."

---

### **Task 4: systemd Service Unit**
*Capability: Configuration generation*

Confirmed working in testing. Every Linux service needs a unit file. Asking a model to generate one from a spec is faster and less error-prone than writing it by hand. The model gets every field right and adds sensible defaults.

> **Prompt:**
> "Write a systemd unit file for a VDI session agent with these exact values: Description=VDI Session Agent, ExecStart=/usr/local/bin/vdi-agent --config /etc/vdi/agent.conf, Restart=on-failure, RestartSec=5, User=vdi, WantedBy=multi-user.target. Include [Unit], [Service], and [Install] sections. Output ONLY the unit file contents."

---

### **Task 5: On-Call Handoff Note**
*Capability: Business communication under constraints*

Nobody writes good handoff notes at 2am. This shows the model taking raw incident facts and producing a structured, useful summary in a fixed format — the kind a real on-call rotation would actually use. Three bullets, no padding, no hedging.

> **Prompt:**
> "You are the outgoing on-call engineer. Write a handoff note for the incoming engineer based on this incident: VDI broker at 10.0.1.45:8443 became unreachable at 09:14 UTC. 47 active user sessions were disconnected. The service was restarted at 09:31 UTC and sessions recovered. A memory leak in vdi-broker v2.3.1 is suspected. Format each bullet exactly as '- [LABEL]: detail' using these three labels: STATUS, HYPOTHESIS, NEXT ACTION. Output ONLY the three bullets."

---

### **What the demo shows**

Five different capability categories: classification, querying, transformation, generation, and communication. Each is a task someone in an IT or ops org actually performs. The model handles all five locally, privately, with sub-300ms time-to-first-token.

The question isn't whether a small model can beat a large one. The question is whether it's good enough for the task in front of it — and fast enough that it doesn't slow anyone down. These five say yes.
