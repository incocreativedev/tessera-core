# Swarm Round-Trip Architecture Spec (Tessera v1)

**Status:** v1 decisions locked  
**Pilot domain:** Ag/Mining edge fleets  
**Governance:** Central operator  
**Rewards:** Quality-weighted free-usage credits  

---

## 1. Summary

Bidirectional training system: many contributor-side small models train locally, send Tessera tokens to a central operator, and receive periodic "collective update" tokens from the large model. Contributors earn quality-weighted free-usage credits.

**Two implementation layers (merged):**
- **Protocol layer** (`tessera.swarm`): `submit()`, `aggregate_tokens()`, `broadcast()`, `score_token()`, `compute_credits()`; no model training. Used for coordination, CLI, and policy/credits gating.
- **ML engine** (`SwarmAggregator`): Fingerprints the central model, trains UHS on aggregator activations, aggregates contributor tokens, fine-tunes the central model on decoded hub targets, re-encodes updated activations as a broadcast token. Uses privacy √N composition. Gate inputs with `policy.accept_token()` and `score_token()` before passing tokens to the aggregator.

---

## 2. Public Interface

### CLI (tessera swarm)

| Command | Purpose |
|--------|--------|
| `tessera swarm submit --token <path> --contributor-id <id>` | Submit a contributor token to central ingress |
| `tessera swarm aggregate --round <id>` | Aggregate accepted tokens for a round (hub vector) |
| `tessera swarm broadcast --round <id>` | Emit broadcast token for round (large model → contributors) |
| `tessera swarm score --round <id>` | Score tokens for a round (utility) |
| `tessera swarm credits --contributor-id <id>` | Show credits / free-usage tier for contributor |

### Token metadata (custom_metadata)

Swarm-specific fields stored in `TesseraToken.custom_metadata`:

| Field | Type | Description |
|-------|------|-------------|
| `swarm_round_id` | str | Round this token belongs to |
| `contributor_id` | str | Contributor identity |
| `local_data_fingerprint` | str | Non-PII hash of local data (e.g. dataset hash) |
| `quality_signals` | dict | `{drift, recon_error, novelty, freshness}` |
| `aggregation_weight` | float | Weight used in robust weighted mean |
| `utility_score` | float | Computed utility for credits |
| `broadcast_version` | str | Version of broadcast token (when applicable) |
| `lineage_parent_rounds` | list[str] | Prior round IDs this update depends on |

### APIs

- `aggregate_tokens(tokens, method="robust_weighted_mean") -> hub_vector`
- `score_token(token, round_context) -> utility_score`
- `compute_credits(contributor_id, utility_scores) -> credits`

---

## 3. System Design

### Contributor loop (small model side)

1. Train locally on contributor data.
2. Encode activations to UHS.
3. Package TBF token with swarm metadata and provenance.
4. Submit token to central ingress.
5. Receive broadcast token for latest accepted round.
6. Decode broadcast into local architecture and run local distillation update.

### Central loop (large model side)

1. Ingest tokens; run validation and policy checks.
2. Score each token for utility.
3. Aggregate accepted hub vectors with robust weighting.
4. Decode aggregate into large-model target space.
5. Fine-tune large model on decoded targets.
6. Re-encode updated large-model behavior as broadcast token.
7. Emit contributor credits and updated free-usage quota.

### Aggregation algorithms

- **v1 default:** `robust_weighted_mean` — base weight = normalized utility score; Huber-style clipping of outliers by cosine-distance to median; final vector = weighted mean of clipped vectors.
- **Other strategies** (L2-renormalised): `mean`, `weighted`, `trimmed` (drop farthest by cosine), `median` (coordinate-wise). See `AggregationStrategy` in `tessera.swarm`.
- **Minimum accepted contributors per round:** 5.
- **Round cadence:** every 24 hours.
- **Broadcast tokens** from the central model use `KnowledgeType.SWARM`.

---

## 4. Quality-Weighted Credits Policy (v1)

**Utility score (per accepted token):**

```
utility = 0.35*quality + 0.25*novelty + 0.20*freshness + 0.20*reliability
```

- **quality:** from inverse drift / reconstruction error.
- **novelty:** distance to prior accepted round centroid.
- **freshness:** recency window decay (full credit ≤24h).
- **reliability:** contributor historical acceptance ratio.

**Credits:**

- Daily credits per contributor = sum of accepted utility scores for the round.
- **Abuse guard:** per-contributor max 15% of total round weight.
- Free-usage tier maps to rolling 30-day credits.

---

## 5. New Components

| File | Purpose |
|------|---------|
| `tessera/swarm.py` | Protocol: submit, validate, aggregate_tokens, score_token, compute_credits, broadcast, score; ML engine: `SwarmAggregator` (aggregate, aggregate_and_broadcast), `AggregationStrategy` enum |
| `tessera/policy.py` | Governance and acceptance rules for central operator |
| `tessera/credits.py` | Utility scoring and credit ledger |
| `docs/swarm-roundtrip-architecture.md` | This spec, lifecycle, threat model |
| `tests/test_swarm.py` | Round-trip orchestration and aggregation tests |
| `tests/test_credits.py` | Scoring and credit assignment tests |
| `tests/test_swarm_integration.py` | Full-cycle integration: submit → validate → score → aggregate → broadcast; lineage and credits |

---

## 6. Failure Modes and Safeguards

| Risk | Safeguard |
|------|-----------|
| Poisoning / outlier contributors | Robust clipping + contributor weight cap + acceptance gating |
| Stale or replayed tokens | Enforce round IDs, timestamp windows, lineage checks |
| Privacy drift | Require privacy metadata; reject missing policy fields |
| Collapse from low diversity | Halt aggregation if contributor diversity threshold fails |
| Large-model regression | Rollback to prior broadcast version if validation metrics fail |

---

## 7. Testing and Acceptance Criteria

**Unit tests:** token scoring determinism; aggregation correctness under outliers; credit calculation and caps; metadata validation and rejection.

**Integration tests:** simulate 20 contributors, mixed architectures; full cycle submit → aggregate → large update → broadcast → local decode; lineage and version propagation.

**Benchmark (Ag/Mining pilot):** baseline vs swarm-updated large model; baseline vs swarm-updated small models after broadcast; task accuracy/F1, drift stability, token rejection rate, cost per improvement point, central GPU-hours saved.

**v1 acceptance gates:**

- ≥3% relative improvement on pilot task over no-swarm baseline.
- ≤5% regression risk in worst-site performance.
- ≥25% central training GPU-hour reduction vs centralized-only schedule.
- Zero raw-data transfer in protocol logs.

---

## 8. Rollout Plan

| Phase | Description |
|-------|-------------|
| A | Offline simulation with synthetic/public Ag/Mining-like datasets |
| B | Shadow rounds with real contributors; no production credit impact |
| C | Production credits enabled; capped usage tiers |
| D | Broadcast updates enabled for contributor local model refresh |
| E | Policy tuning after 30-day telemetry review |

---

## 9. Assumptions and Defaults

- Existing Tessera UHS and TBF primitives unchanged at core serialization level.
- Swarm-specific fields live in `custom_metadata` for backward compatibility.
- Central operator is trusted for policy and rewards in v1.
- Contributors accept non-bit-identical local updates (behavioral transfer).
- Daily round cadence for pilot; no real-time sync in v1.
