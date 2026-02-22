# Comparison: Healthcare PDF Brief vs Research-Backed Hypothesis Doc

**Compared:**  
- **A.** `Tessera_Healthcare_Medical_Imaging_Use_Case.pdf` (Kirk Maddocks | Inco Creative | Feb 2026) — **25 pages**  
- **B.** `docs/use-cases/healthcare-medical-imaging-use-case.md` (research summary + hypothesis test)

**Current PDF (25 pages):** Includes **§7 Healthcare AI Consortia Alignment** (TRAIN, COMAI, MedPerf, COFE; Tessera’s fit; lightweight models) and **§8 Cross-Vertical Comparison: SDC / Ag-Mining / Healthcare** (table + “why healthcare” paragraph). Validation Framework is §9, Technical Mapping §10, Worked Examples 11–12, When Not the Right Fit §13, Getting Started §14, Appendix §15, Sources & References §16.

---

## 1. Side-by-Side Comparison

| Dimension | Your PDF (A) | Research doc (B) |
|----------|--------------|------------------|
| **Purpose** | Industry use case brief for healthcare teams, regulators, and partners; citeable. | Hypothesis validation and research summary; conditions and evidence. |
| **Use cases** | **7 named scenarios** — Radiology: cross-scanner portability, hospital-to-hospital, rural specialist AI, cross-vendor stroke. Pathology: cross-scanner WSI, foundation→edge, multi-site tumour classification. | **6 thematic benefit areas** — edge vs cloud, cross-vendor/scanner, site-to-site, FDA/PCCP, research→production, auditability. No named scenarios. |
| **Concreteness** | **Very high:** Vendor names (GE Edison, Siemens syngo.via, Philips, Canon; Aperio, Hamamatsu, Paige, PathAI); market sizes with citations [1],[5],[6]; AUC 0.74→0.60 [3]; 5.2% FL deployment [4]; 28% rural broadband [10]; token sizes (5.1 KB, 2.8 KB); pilot status table (Seeking partner, Q4 2026). | **Medium:** Benefit categories and research summary tables; no OEM/scanner names, no token sizes, no pilot status. |
| **Implementation vs protocol** | **Explicit table:** Mode A, UHS, TBF, drift, DP, CLI+MCP = IMPLEMENTED; Mode W/B/C/D, SWAP = SPECIFIED. Worked examples use Mode A only; notes where use case references specified-only modes. | Focus on Mode A and current implementation; no explicit IMPL vs SPEC table. |
| **Evidence / citations** | **20 numbered references** in §14 (Markets and Markets, FDA, arxiv, PMC, Mordor, Paige, Aidoc, Viz.ai, FDA PCCP, Innolitics, etc.). In-text [1]–[20]. Cite-as line for the document. | Short reference list by theme (Springer, Nature, FDA, etc.); no numbered in-text citations. |
| **Regulatory** | **Dedicated §6:** PCCP draft guidance, how Tessera aligns (drift as change validation, provenance as audit trail, bounded updates, pre/post metrics). Caveat: token integration would require regulatory review. Appendix includes PCCP alignment. | PCCP and lifecycle mentioned in “Where Tessera could help” and “What has to be true”; no dedicated regulatory section. |
| **Validation framework** | **§7:** Assumption table with Evidence for, Evidence against/gaps, Status. Recommended pilots (radiology cross-vendor chest X-ray; pathology ViT→CNN). **Pilot status table** (scope, status, timeline). Verdict: “strong but clinical validation essential.” | “What has to be true” (4 bullets); hypothesis-test table; verdict. No pilot status table, no “evidence against” column. |
| **Limitations** | **§11 When Tessera Is Not the Right Fit:** same-vendor/same-scanner, sufficient local data, real-time pipelines, bit-exact regulatory, single-arch FL, no clinical validation yet. | No dedicated limitations section. |
| **Federated learning** | **§5 Beyond Federated Learning:** FL vs Tessera comparison table (architecture, bandwidth, rounds, adoption, IP, regulatory, privacy). “Complementary, not replacement.” | FL mentioned in research summary and consortia; no FL-vs-Tessera section. |
| **Worked examples** | **Two full examples** with **tessera-core-aligned API:** ModeATransfer(transmitter=, receiver=), transfer.execute(), TBFSerializer.save/load, QuantType. Chest X-ray (5.1 KB, Site A/B/C, drift, AUC); Pathology ViT→EfficientNet (2.8 KB, Gleason). | None. |
| **Getting started** | Prerequisites (Python, PyTorch, tessera-core, tested platforms); install; benchmark; CLI; 4-week path with IRB note for clinical; resources. | Recommendations only; no step-by-step or prerequisites. |
| **Cross-vertical comparison** | **§8 Cross-Vertical Comparison: SDC / Ag-Mining / Healthcare** — table (architecture diversity, edge vs cloud, regulation, data privacy, sharing/consortia, domain shift, token value proposition) + “why healthcare” paragraph. | §7 table + “Why Tessera fits healthcare relative to other verticals” paragraph. |
| **Consortia alignment** | **§7 Healthcare AI Consortia Alignment** — TRAIN, COMAI, MedPerf, COFE; common challenge; Tessera’s fit; lightweight models (MobileViT, BitMedViT, etc.). | Consortia in research summary and comparison; no dedicated section. |
| **Format** | **25-page** PDF; contents; implementation table; §7 Consortia, §8 Cross-vertical; appendix; **Sources & References** (§16); cite-as. | ~4-page markdown; sections 1–10; reference summary. |

---

## 2. What Each Does Better

**Your PDF (A)**  
- **Citations and defensibility:** 20 numbered sources; in-text [1]–[20]; market and regulatory facts traceable. Document itself is citeable.  
- **Implementation transparency:** IMPLEMENTED vs SPECIFIED table and “worked examples use Mode A” note avoid overclaiming; technical accuracy vs codebase is high.  
- **Regulatory narrative:** §6 PCCP and “How Tessera aligns” give partners and regulators a clear story; caveat on regulatory review is appropriate.  
- **Concrete scenarios:** Cross-scanner chest X-ray, rural specialist AI, cross-vendor stroke, pathology foundation→edge, multi-site Gleason—each with vendor/scanner names and key advantage.  
- **FL positioning:** §5 “Beyond Federated Learning” table and “complementary” framing clarify when to use Tessera vs FL.  
- **Limitations:** §11 “When Tessera Is Not the Right Fit” plus “no clinical validation yet” show balance and reduce overclaim risk.  
- **Pilot status:** Table (Seeking partner, Q4 2026) makes next steps concrete.  
- **API-correct worked examples:** Snippets match tessera-core (ModeATransfer, execute, TBFSerializer).  

**Research doc (B)**  
- **Hypothesis in one place:** Single paragraph hypothesis; easy to copy or cite for papers/proposals.  
- **Cross-use-case comparison:** Table comparing healthcare to autonomous driving and ag/mining (regulation, privacy, domain shift) in one view.  
- **Concise “what has to be true”:** Four bullets; quick scan for partners.  
- **Reference summary by theme:** Architectures, edge/cloud, transfer, domain shift, regulation, consortia—useful for deep dives without reading the PDF.  

---

## 3. Gaps in Each

**Your PDF (A)**  
- **Cross-use-case comparison:** Does not compare healthcare to SDC or ag/mining in one table (useful for “why Tessera in healthcare” relative to other verticals).  
- **Hypothesis as single statement:** The value proposition is clear but not distilled into one copy-paste “hypothesis” paragraph for papers/grants.  

**Research doc (B)**  
- **No citations list:** References are summary only; no numbered refs for fact-checking.  
- **No “When not to use”:** Missing explicit limitations (same-vendor, real-time, bit-exact, etc.).  
- **No worked examples or API:** No code or token sizes; harder to visualize operational benefit.  
- **No implementation status:** Doesn’t spell out IMPL vs SPEC for healthcare readers.  
- **No pilot status:** No table or timeline for next steps.  
- **No FL vs Tessera:** Doesn’t spell out when FL is better vs when Tessera is better.  

---

## 4. Scores (1–10)

| Criterion | Your PDF (A) | Research doc (B) |
|-----------|--------------|------------------|
| **Concreteness / specificity** | 9 | 6 |
| **Evidence / rigor (citations)** | 9 | 6 |
| **Persuasiveness for industry** | 9 | 6 |
| **Technical accuracy vs codebase** | 9 | 8 |
| **Completeness** | 10 | 7 |
| **Actionability** | 9 | 6 |
| **Caveats / limitations** | 9 | 6 |
| **Fit for external citation** | 9 | 6 |
| **Regulatory alignment** | 9 | 7 |
| **Overall (average)** | **9.1** | **6.4** |

**Interpretation:** The healthcare PDF scores **9.1/10** — higher than the ag/mining brief (8.1) because it already incorporates: (1) **Sources & References** with numbered citations, (2) **Implementation status** table (IMPL vs SPEC), (3) **When Tessera Is Not the Right Fit**, (4) **API-correct** worked examples, (5) **Pilot status** table, and (6) **Regulatory** section with PCCP alignment and caveat. The research doc remains the short hypothesis + comparison-to-other-verticals reference; the PDF is the primary industry and regulatory asset.

---

## 5. Improvements Applied to the Research Doc

To close gaps in the research doc and align it with the PDF without duplicating it:

1. **Add “When Tessera is not the right fit”** — Short subsection (same-vendor/same-scanner, sufficient local data, real-time pipelines, bit-exact regulatory, single-arch FL, no clinical validation yet) so the hypothesis doc states limitations explicitly.  
2. **Add “Industry brief” pointer** — Note that a detailed, citeable industry brief (PDF) exists with use cases, regulatory section, FL comparison, worked examples, and references.  
3. **Add key citations** — Pull 3–5 key source labels from the PDF (e.g. cross-vendor AUC drop [3], FL deployment rate [4], FDA PCCP [19]) into the research doc’s reference section for traceability.  
4. **Add cross-vertical comparison** — Keep the existing comparison table (SDC, ag/mining, healthcare); no change needed.  
5. **Optional: one short “Example scenarios” line** — e.g. “Example scenarios: cross-vendor chest X-ray (GE→Siemens/Canon), pathology ViT→edge CNN (Paige-style foundation→community lab); see industry brief for full narrative and code.”  

These edits are applied in `healthcare-medical-imaging-use-case.md` below.

---

## 6. Summary

| Document | Best for | Strongest on | Weakest on |
|----------|----------|--------------|------------|
| **Your PDF** | Partners, regulators, grants, clinical teams | Concreteness, citations, implementation table, regulatory, FL comparison, limitations, worked examples, pilot status | Cross-vertical comparison (SDC, ag/mining) not in one table |
| **Research doc** | Hypothesis statement, cross-vertical comparison, quick “what has to be true” | Hypothesis paragraph, comparison table, concise conditions | Citations, limitations, examples, pilot status, regulatory narrative |

**Scores:** **PDF: 9.1/10** | **Research doc: 6.4/10**. The healthcare brief is already close to “path to 10”: it has sources, implementation clarity, limitations, correct API, **§7 Consortia Alignment**, and **§8 Cross-Vertical Comparison** (SDC / ag-mining / healthcare table + token value proposition row). Keep in-text numbers traceable to refs (e.g. 5.2% FL → [4]) as the PDF evolves.

**Keeping the PDF current:** (1) **Citations** — When new market data, FDA guidance (PCCP final, lifecycle), or peer-reviewed evidence appears, update §16 Sources & References and in-text [1]–[n]. (2) **Implementation table** — When tessera-core or the protocol changes (e.g. a mode moves from SPECIFIED to IMPLEMENTED), update the “Implementation status” table so it matches the current codebase and spec.
