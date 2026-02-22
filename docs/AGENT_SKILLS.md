# Central agent skills — Claude, Cursor, Codex

**Purpose:** One shared reference so each agent contributes to the **betterment** of Tessera without overwriting the others. Use this file in Claude project knowledge, Cursor rules, or Codex skills.

**Initiator and design authority:** Kirk Maddocks initiated the Tessera concept. The agents (Claude, Cursor, Codex) implement and extend under that vision.

---

## 1. What Tessera is

- **Tessera** = open protocol for **AI-to-AI knowledge transfer**.
- Knowledge is encoded as **activation-level representations**, moved through a **Universal Hub Space (UHS)**, and packaged as **Tessera tokens** (TBF).
- Enables one model to teach another across different architectures (no weight sharing required).
- **Swarm** (v1): many small models contribute tokens → central operator aggregates, updates a large model, broadcasts back; contributors earn quality-weighted credits.

**Key code areas:** `tessera/` (token, uhs, transfer, fingerprint, drift, privacy, gates, binary, swarm, policy, credits), `docs/` (specs, use cases), `tests/`.

---

## 2. Cross-agent principle: contribute, don’t overwrite

- **Extend and improve** existing behavior; avoid replacing or contradicting working, committed code unless the user explicitly asks for a rewrite.
- **Respect boundaries:** if another agent owns an area (see §3), add hooks, options, or docs rather than reimplementing that area.
- **Single source of truth:** specs and contracts live in `docs/` (e.g. `docs/swarm-roundtrip-architecture.md`). When changing behavior, update the relevant doc and keep code and docs in sync.
- **Tests:** add or update tests for your changes; don’t remove or weaken tests that guard others’ behavior without explicit user approval.

---

## 3. Agent roles (suggested focus, not hard borders)

All three agents work under the same design authority (see top of doc). Roles below are about how each contributes, not who “owns” the project.

| Agent   | Suggested focus | Avoid |
|--------|------------------|--------|
| **Claude** | Research, long-form docs, use-case narratives, protocol semantics, ML pipeline design, aggregation/training logic. | Rewriting Cursor’s protocol/CLI or Codex’s automation without merging. |
| **Cursor** | Implementation in this repo: Python APIs, CLI, tests, refactors, protocol layer (submit/aggregate/score/credits), tooling. | Reimplementing Claude’s ML/aggregation design in a conflicting way; delete-and-replace of large blocks. |
| **Codex** | Automation, scripts, CI, packaging, multi-repo or external workflows, orchestration that calls Tessera. | Changing core Tessera APIs or token semantics; overwriting implementation details that Cursor maintains. |

When in doubt: **add a small, additive change or document the proposed change** (e.g. in a comment or `docs/`) and let the user decide before large refactors.

---

## 4. Where to look before changing

- **Swarm / round-trip:** `docs/swarm-roundtrip-architecture.md`, `tessera/swarm.py`, `tessera/policy.py`, `tessera/credits.py`.
- **Token / TBF:** `tessera/token.py`, `tessera/binary.py`; swarm metadata in `custom_metadata` (see swarm spec).
- **Transfer / UHS:** `tessera/transfer.py`, `tessera/uhs.py`; `SwarmAggregator` in `tessera/swarm.py` for central-model training.
- **Use cases:** `docs/use-cases/*.md` (agriculture-mining, healthcare-medical-imaging, etc.).
- **Contributing:** `CONTRIBUTING.md` (branching, tests, style).

---

## 5. How to use this file per platform

- **Claude:** Include this file (or a short summary + link) in project knowledge / custom instructions so Claude sees roles and “contribute, don’t overwrite” before editing.
- **Cursor:** Add a rule in `.cursor/rules/` that references this file (e.g. “When editing Tessera, follow docs/AGENT_SKILLS.md for cross-agent contribution rules”).
- **Codex:** Add a skill or prompt that points at this file (or its path in the repo) so Codex runs automation and scripts in line with these roles and principles.

---

## 6. Keeping this file current

- Prefer **appending or refining** sections (e.g. new agent focus, new “where to look” paths) over deleting existing guidance.
- If an agent’s role or a principle changes, update §2 and §3 and add a short note at the top with the date of the last substantive change.

*Last substantive update: 2025-02-21 — initiator set to Kirk Maddocks; roles clarified as contribution focus under one vision.*
