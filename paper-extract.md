Here’s a tight “agent-readme” you can drop into your vibe coding agent.

MEMORYCODE at a glance
	•	What it is: a synthetic multi-session mentor↔mentee coding benchmark where simple code rules (e.g., “prefix function names with gn_”) are taught across sessions, updated later, and buried among irrelevant “filler” office chatter. The agent must retrieve the latest rules and compose them when writing code. (See fig. 1 and §3.)  ￼
	•	Dataset shape: 360 dialogue histories spanning 1–100 sessions each (12 sizes). “Short” <15 sessions vs “Long” ≥16. Longest history ≈63k tokens; typical token counts: short ~3.2k (±2.7k), long ~26.2k (±15.5k) (Table 2, p. 5). Rules: 51 instructions (16 updatable, up to 8 times); 80 fillers (two types). (§3; Tables 5–6, pp. 16–21.)
	•	Tasks / inputs: three setups: INSTRUCTION (single rule), SESSION (one full session: many rules + filler), HISTORY (entire multi-session history). Output is Python code only. (§4, p. 5–6.)

How scoring works (wire this into your agent harness)
	•	Each instruction has a regex-based test; the run scores 1 only if (a) all relevant Python objects comply and (b) the code has no syntax errors; macro-average across items. (§4, p. 6.)
	•	Evaluation prompting: models were evaluated with temperature 0; outputs must be code-only, no prose (Appendix F prompt templates on pp. 34). For dataset generation they used temperature 0.9/top-p 0.9, but for evaluation keep T=0 (§C, p. 13; §4.1, p. 6).

Results your agent should expect (headline stats)
	•	Easy in isolation: near-perfect on INSTRUCTION for large models (e.g., DeepSeek-V3 100%; GPT-4o 94.5%).
	•	Multi-turn is OK; multi-session is hard: on Short History (<15 sessions) GPT-4o 79.6%, DeepSeek-R1 85.9%; on Long History (16–100) everyone collapses: GPT-4o 30.5%, DeepSeek-R1 41.3%, Llama-3.1-405B 20.9% (Table 4, p. 15; Fig. 3, p. 7).
	•	Why models fail: not just retrieval—compositional application of multiple, updated rules is the bottleneck (INSTRUCTIONS-CHAIN curve mirrors HISTORY; Fig. 5, p. 8). RAG gives marginal gains on short, none on long (Appendix D, Fig. 7, p. 13).
	•	Difficulty varies by rule: common practices (docstrings, annotations) score higher than odd patterns (digits in names); see per-rule bars (Fig. 8–9, pp. 14–15).

“How to use the repo” — practical notes for a new model
	1.	Data & splits: load the provided dialogues and the three evaluation inputs (INSTRUCTION / SESSION / HISTORY). For HISTORY, feed full concatenated sessions in order. (§4, p. 5–6; prompts in Appendix F, pp. 34.)
	2.	Prompts (copy exactly):
	•	History/Session system prompt: “You are the mentee… Only generate Python code and nothing else… Don’t ask for more information.”
	•	History/Session user prompt: include the entire dialogue or session text followed by the coding request (examples on pp. 34).
	•	Instruction prompt: single rule embedded in the style guide; again code-only output (Table 14–16, p. 34).
	3.	Decoding: temperature=0, greedy; truncate model chatter if present (reject-sample until code-only). (§C, p. 13; Appendix F.)
	4.	Evaluation harness: run the repo’s regex tests per item; fail any sample with syntax error; macro-average accuracy. (§4, p. 6.)
	5.	Repro knobs: you can optionally reproduce their RAG baseline by retrieving top-k prior sessions that contain pivots; expect little/no win on long histories (Appendix D, p. 13).
	6.	Compute sanity checks: verify your context window covers up to ~63k-token histories (or implement chunk-and-reason strategy if your model can’t hold the full history). (Table 2, p. 5.)

Agent tactics that materially improve odds (based on findings)
	•	Prospective-memory scratchpad: during ingestion, extract rules per object type (function names, method names, args, attributes, imports, decorators, comments) and apply updates “last-write-wins.” Reprint the consolidated rule set to the model just before code generation. (§§3, 6; Fig. 5, p. 8.)
	•	Composition check before emit: have the agent self-audit: “list all active rules and tick them off” → then generate code. (The paper shows composition—not single updates—is the main failure mode; §6.2, p. 7–8.)
	•	Static compliance pass: after generation, run the same regex tests locally; if any fail, auto-repair and re-test before submitting. (§4, p. 6; Table 5 rules, pp. 16–17.)

If you want, I can turn this into a markdown README for your agent, including ready-to-paste prompt blocks and a tiny Python harness that wraps your model and runs the regex tests.
