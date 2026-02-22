# Comparison: Your Use Case Brief (PDF) vs Research-Backed Hypothesis Doc

**Compared:**  
- **A.** `Tessera_Mining_Agriculture_Use_Case.pdf` (Kirk Maddocks | Inco Creative | Feb 2026)  
- **B.** `docs/use-cases/agriculture-mining-use-case.md` (research summary + hypothesis test)

**Current PDF on file:** 19 pages. Full brief with §5 Digital Twin / Sim-to-Real, §6 Token Marketplace, §7 Validation Framework, §8 Shared Challenges, §9 Technical Mapping, §10–11 Worked Examples, §12 Getting Started, §13 Appendix. This is the **full** version (scored 8.1/10 in the comparison below).

---

## Updates in PDF v2 (19 pages, Feb 2026)

The updated PDF adds four major elements that align it with the research doc and the comparison feedback:

| Addition | What it does |
|----------|----------------|
| **§5 Digital Twin / Sim-to-Real Transfer** | Dedicated section on sim-to-real: Fortescue, Rio Tinto, Mineral, Blue River; Tessera as bridge (sim encodes → real decodes, drift quantifies gap, SWAP for bidirectional). Addresses the “sim→real” benefit area from the research doc. |
| **§6 The Token Marketplace** | Registry concept with example table (ore-waste-discriminator, wet-road-speed-policy, amaranth-resistant, frost-damage-detector, drill-pattern-granite, soybean-yield-predictor; providers BHP, Rio Tinto, Iowa Corn Coop, CSIRO, Epiroc, AGCO). “Why tokens not data”: IP, privacy, architecture independence, auditability, cooperatives (Agri-Gaia, NAPDC). Positions Tessera as “open protocol, commercial marketplace” (Linux/Kubernetes playbook). |
| **§7 Validation Framework: What Has to Be True** | Assumption table with Mining evidence, Agriculture evidence, and Status (Strong / Needed / Plausible). Assumptions: cross-arch bottleneck, “behaviour” as shareable artefact, compact tokens beat full weight (Strong); in-domain pilots, fits workflows (Needed); token marketplace adoption (Plausible). **Recommended pilots:** Mining = sim-to-real haul trucks (transformer in twin → CNN on-vehicle); Agriculture = ViT→CNN weed detection (ViT cloud → MobileNetV2 on sprayer). **Verdict:** “The hypothesis holds. The main gap is in-domain proof — one mining pilot and one agriculture pilot would make the benefit concrete.” |
| **Executive summary** | Now states the doc “explores sim-to-real transfer and the token marketplace concept, and provides a rigorous validation framework assessing what has to be true for Tessera to deliver real value.” |

**Effect:** The PDF now includes explicit caveats, a “what has to be true” framework, recommended pilots, and a verdict that match the research doc. The “caveats / rigor” gap is largely closed; the brief remains industry-facing while being more defensible.

---

## 1. Side-by-Side Comparison

| Dimension | Your PDF (A) | Research doc (B) |
|----------|--------------|------------------|
| **Purpose** | Industry use case brief: persuade and guide industrial teams to evaluate Tessera. | Research validation: test whether ag/mining are plausible use cases and under what conditions. |
| **Use cases** | **8 named, narrative scenarios** — Mining: fleet-to-fleet (Cat/Komatsu), drill-to-drill, predictive maintenance across equipment classes, site-to-site (AU/BR/CA). Ag: tractor-to-tractor (9RX→8R), drone-to-ground, crop-specific (corn→soy), sprayer/weed (See & Spray, new species). | **6 thematic benefit areas** — edge vs cloud, heterogeneous hardware, sim→real, region/site transfer, fleet updates, auditability. No named OEMs or scenarios. |
| **Concreteness** | **Very high:** OEM names (Cat, Komatsu, Deere, CNH, AGCO, DJI), market sizes ($1.6B→$12.6B mining, $7.4B→$24.3B ag), connectivity (200–800 ms, 256 kbps), token sizes (3.2 KB vs 450 MB, 4.8 KB), drift thresholds (e.g. &lt;0.10), timelines (5 min fleet-wide, AgIN 2026, Deere 2030). | **Medium:** Benefit categories plus research citations (MDPI, Nature, arXiv). No code, no token sizes, no OEM-specific scenarios. |
| **Protocol scope** | **Full protocol:** Mode A/B/C/D/W, SWAP projection, correlation maps, HMAC, DP, INT8. Positions Tessera as a full multi-mode protocol, not just activation transfer. | **Current implementation:** Focus on Mode A (activation-based), TBF, UHS, drift, DP. No mention of Mode B/C/D/W (not yet in codebase). |
| **Evidence** | Industry trends: fleet counts (~1,500 trucks, 30+ mines), AgIN 2026, Deere autonomy target, Komatsu/Applied Intuition, Vale Carajas. No academic citations. | Research citations: peer-reviewed and preprint (SciDirect, MDPI, Nature, arXiv, IEEE). Hypothesis-test table and explicit “what has to be true.” |
| **Worked examples** | **Two full worked examples** with pseudo-code and numbers: (1) Pilbara mining OTA (30 trucks, encode→distribute→decode→verify→activate), (2) Iowa drone swarm→tractor (weed discovery, mesh, 5 min). | None. |
| **Actionability** | **High:** Install, benchmark, CLI (`tessera inspect`, `validate`, `list-anchors`), 4-week evaluation path, contact and resources. | **Medium:** Recommendations only (document, pilot, engage cooperatives/OEM-agnostic initiatives). |
| **Caveats / rigor** | **Current (19-page):** §7 Validation Framework with assumption table, evidence columns, status (Strong/Needed/Plausible), recommended pilots, verdict. | Explicit “What has to be true” (4 conditions), hypothesis-test table, verdict, comparison to SDC. |
| **Format** | **Current (19-page):** PDF, contents, §5 Sim-to-Real, §6 Marketplace, §7 Validation, worked examples, appendix. | ~2-page markdown; structured sections 1–7; reference list. |

---

## 2. What Each Does Better

**Your PDF (A)**  
- **Narrative and buy-in:** Tells a clear story (knowledge trapped → tokens set it free) with specific actors (Cat, Komatsu, Deere, drones).  
- **Concrete scenarios:** A product manager or OEM can see themselves in “fleet-to-fleet,” “drone-to-ground,” “site-to-site.”  
- **Technical mapping:** Mode W/C/D/B and components (UHS, drift, HMAC, INT8, SWAP) mapped to mining and ag applications — good for aligning protocol roadmap with verticals.  
- **Actionability:** 4-week path and CLI commands give a clear next step.  
- **Worked examples:** Pilbara and Iowa examples with token sizes and drift thresholds make the value proposition tangible.  
- **If using 19-page v2:** Adds §5 Sim-to-Real, §6 Token Marketplace, §7 Validation Framework (assumptions, evidence, pilots, verdict) for stronger rigor in-house.

**Research doc (B)**  
- **Rigor and defensibility:** Hypothesis stated and tested; “what has to be true” and research citations make the claim falsifiable and citable.  
- **Alignment with current code:** Stays within Mode A and implemented features, avoiding overclaiming unimplemented modes.  
- **Evidence base:** Links to literature (architecture diversity, edge/cloud, transfer learning, cooperatives, OEM-agnostic mining) so external reviewers can verify.  
- **Completeness of “why it could fail”:** Explicit conditions (cross-arch as bottleneck, behavior as artifact, in-domain validation) — useful for prioritising pilots and partnerships.

---

## 3. Gaps in Each

**Your PDF (A)**  
- **Protocol vs implementation:** Mode B/C/D/W and some concepts (e.g. “encode(cloud_model, anchor_id=…)”) are not yet in tessera-core; a technical reader might try to run the snippet and find API mismatches.  
- **Evidence:** No academic citations; industry numbers are compelling but not sourced. (In 19-page v2, §7 Validation Framework adds structured evidence and status.)  
- **Caveats (15-page):** “Why Tessera fits” table only; no explicit “what has to be true” or verdict. (In 19-page v2, §7 addresses this.)

**Research doc (B)**  
- **No named scenarios:** Doesn’t give a single “Cat 797F + Komatsu 930E” or “9RX → 8R” style story; weaker for sales or partner conversations.  
- **No worked examples or numbers:** No token sizes, bandwidth, or drift thresholds; harder to visualise operational benefit.  
- **No getting-started path:** Doesn’t tell an industrial team exactly what to run in week 1–4.  
- **No full-protocol vision:** Stays with Mode A; doesn’t lay out how Mode B/C/D/W would map to ag/mining (useful for roadmap).

---

## 4. Scores (1–10)

| Criterion | Your PDF 15-page (current) | Your PDF 19-page v2 | Research doc (B) |
|-----------|---------------------------|---------------------|------------------|
| **Concreteness / specificity** | 9 | 9 | 6 |
| **Evidence / rigor** | 6 | 7 | 8 |
| **Persuasiveness for industry** | 9 | 9 | 6 |
| **Technical accuracy vs current codebase** | 7 | 7 | 9 |
| **Completeness (use cases + protocol + actions)** | 9 | 10 | 7 |
| **Actionability (next steps, CLI, path)** | 9 | 9 | 6 |
| **Caveats / “what must be true”** | 5 | 8 | 9 |
| **Fit for external / academic citation** | 5 | 6 | 8 |
| **Overall (average)** | **7.4** | **8.1** | **7.4** |

**Interpretation:**  
- **Current PDF (19 pages):** Scores **8.1/10** — full brief with §5 Sim-to-Real, §6 Token Marketplace, §7 Validation Framework (assumptions, evidence, pilots, verdict).  
- **Alternative 15-page:** A shorter version (use cases + Shared Challenges + Technical Mapping + examples + Getting Started only) would score **7.4/10** — stronger for a one-shot pitch; full 19-page is stronger for defensibility and validation story.

---

## 5. Recommended Use of Each

- **Your PDF:** Primary **industry-facing** asset: partner conversations, website, grant/applications where you need a polished, scenario-rich story and a clear evaluation path. Optionally add one short “Current implementation” note (e.g. “Mode A and TBF are available today; Modes B/C/D/W are protocol roadmap”) and a “What we’re validating” line that points to the research doc.  
- **Research doc:** **Internal and technical** reference: hypothesis test, citations, and “what has to be true.” Use it in technical whitepapers, academic-style write-ups, and to decide which pilot (e.g. one ag + one mining) to run first.  
- **Merge idea:** Add a “Research and validation” subsection to your brief that references the hypothesis and conditions (and link to the research doc). In the research doc, add 1–2 short “Example scenarios” (e.g. condensed Pilbara + Iowa) and a “Getting started” pointer to your PDF or README. That way both documents reinforce each other.

---

## 6. Summary

| Document | Best for | Strongest on | Weakest on |
|----------|----------|--------------|------------|
| **Your PDF (15-page current)** | Industry brief, partners, evaluation path | Concreteness, narrative, worked examples, actionability | Protocol vs implementation, explicit caveats, citations |
| **Your PDF (19-page v2)** | Same + validation story | Same + validation framework, sim-to-real, token marketplace | Protocol vs implementation, academic citations |
| **Research doc** | Hypothesis testing, citations, prioritising pilots | Rigor, “what has to be true,” alignment with code, references | Named scenarios, numbers, getting-started path |

**Scores:** **PDF 15-page: 7.4/10** | **PDF 19-page v2: 8.1/10** | **Research doc: 7.4/10**. The 15-page brief is the concise pitch; the 19-page v2 adds rigor and marketplace/sim-to-real. Research doc remains the citation and code-aligned reference.
