# Tessera Use Case: Healthcare & Medical Imaging

**Status:** Hypothesis and research summary  
**Date:** 2025-02-21  
**Context:** Exploration of healthcare and medical imaging as a real-world use case for Tessera (AI-to-AI knowledge transfer), with research-backed hypothesis and feedback in the same style as autonomous driving and agriculture/mining.

---

## 1. Hypothesis

**Tessera can benefit healthcare and medical imaging by enabling cross-architecture, portable knowledge transfer between: (a) cloud/heavy and edge/light models (e.g. ViT or 3D CNN in the data center → lightweight CNN or quantized ViT on clinic/hospital edge), (b) different model families (CNN, Vision Transformer, hybrid ViT-CNN) deployed across heterogeneous vendors (scanner manufacturers, PACS, AI vendors), (c) models trained at one site or scanner and consumed at another without sharing raw PHI, and (d) sim or research models and production-cleared architectures—all via a standard token (TBF) with audit trail and optional differential privacy, supporting FDA-aligned updates (e.g. PCCP), multi-site generalization, and consortium or network sharing of “capabilities” rather than data.**

---

## 2. Research Summary

### Medical imaging AI: architectures and deployment

| Finding | Source / evidence |
|--------|--------------------|
| **Architecture diversity** | CNNs, Vision Transformers, and hybrid ViT-CNN architectures are all used; systematic reviews (2024) show transformer-based and hybrid models often outperforming conventional CNNs for segmentation, classification, and detection. U-Net evolution with transformer components is common. |
| **Edge vs cloud** | Edge deployment is driven by PHI/privacy (data stays local), latency, and connectivity. Cloud used for heavy training; model compression, quantization, and knowledge distillation used to get performant models onto edge (e.g. 3D CNN for lung screening on consumer hardware; BitMedViT, MobileViT on edge). Hybrid “train in cloud, run on edge” is standard. |
| **Lightweight vs heavy** | Lightweight transformers (MobileViT, EfficientViT, EdgeViTs, ternary-quantized ViT) and small CNNs run on devices; large ViTs and 3D CNNs run in cloud or on servers. Cross-architecture transfer (e.g. ViT → small CNN) is relevant where edge cannot run the teacher. |
| **Knowledge transfer in use** | Federated learning with knowledge distillation (e.g. cardiac CT across 8 hospitals, CNNs → transformer); multi-organ segmentation with distillation to avoid catastrophic forgetting; transfer learning and domain adaptation for cross-site generalization. |

### Heterogeneity and domain shift

| Finding | Source / evidence |
|--------|--------------------|
| **Scanner and vendor heterogeneity** | MRI scanner manufacturer (GE, Philips, Siemens) causes significant domain shift; performance drops when models trained on one manufacturer are tested on another. PACS and AI vendors vary (KLAS Imaging AI 2024); infrastructure and vendor compatibility are cited as implementation challenges. |
| **Multi-site generalization** | Distribution shift across sites (scanner, protocol, population, labels) is a major obstacle. Adding multi-hospital data can sometimes hurt worst-group accuracy (spurious hospital–disease correlations). FedMedICL and similar frameworks evaluate multi-site shifts; site-specific vs multi-institutional model choice depends on data size and variability. |
| **Regulatory and predicate lock-in** | FDA 510(k) and predicate networks can tie vendors to specific architectures; changing architecture may require new predicates. PCCP (Predetermined Change Control Plans) and lifecycle guidance support pre-authorized modifications with documented methodology and impact assessment—aligning with auditable, controlled updates (e.g. token-based capability updates). |

### Sharing, consortia, and audit

| Finding | Source / evidence |
|--------|--------------------|
| **Consortia and networks** | TRAIN (Trustworthy & Responsible AI Network), COMAI, MedPerf (federated benchmarking: “bring model to data”), MLCommons COFE. Focus on evaluation and responsible AI; sharing of “model” or “capability” without centralizing raw data is desired. |
| **Audit and provenance** | FDA PCCP and lifecycle guidance require planned modifications, validation methodology, and impact assessment. Audit trail and provenance for AI updates are regulatory expectations—token metadata (lineage, drift, source) fits. |
| **Privacy** | PHI and data locality are paramount; differential privacy and federated learning are used. Tokens that encode “behavior” without raw data support privacy-preserving capability sharing. |

---

## 3. Where Tessera Could Help (Healthcare & Medical Imaging)

1. **Edge vs cloud (clinic/hospital)**  
   Heavy model (ViT, 3D CNN) in cloud or central server; lightweight model (small CNN, quantized ViT) on edge device or PACS-integrated node. Architectures differ. Tessera allows transfer of “behavior” to the edge model and distribution as a **token**, avoiding full model push and preserving compatibility with existing edge runtimes.

2. **Cross-vendor and cross-scanner**  
   Different scanner manufacturers (GE, Philips, Siemens) and AI vendors imply different model backbones and training pipelines. A **standard token format** (TBF) would allow one validated “capability” (e.g. lesion detector, segmentation policy) to be consumed by different architectures and vendors, reducing lock-in and easing multi-site deployment.

3. **Site-to-site and multi-institution without raw data**  
   Models trained at site A (or on scanner type A) need to be used at site B without sharing PHI. Today: federated learning or transfer learning with careful domain adaptation. Tessera adds a **portable token** of “what the source model learned” that any compatible consumer architecture can ingest—supporting consortia (TRAIN, MedPerf-style evaluation, or capability sharing) without centralizing data.

4. **FDA-aligned updates (PCCP / lifecycle)**  
   PCCPs allow pre-authorized modifications with documented methodology and impact assessment. A **Tessera token** with provenance, drift metric, and optional DP could support “capability patch” updates (e.g. new lesion type, new scanner protocol) within a PCCP, with a clear audit trail and impact signal (drift) before activation.

5. **Research / sim → production**  
   Research or simulation models (e.g. large ViT, different architecture) may need to be translated into a production-cleared, edge-deployable form. Tessera can transfer knowledge from research/sim model to production architecture in a compact, auditable way—supporting “train in research, deploy as token” under a controlled change process.

6. **Auditability and lineage**  
   Tokens carry metadata, lineage, and optional DP. For FDA, hospital QA, and liability, “this device’s model was updated with this token, from this source, with this drift score” provides a clear story—similar to the benefit in autonomous driving and ag/mining.

**Industry brief:** A detailed, citeable industry brief (25-page PDF) is available with seven narrative use cases (radiology and pathology), regulatory section (FDA PCCP), “Beyond Federated Learning” comparison, **§7 Healthcare AI Consortia Alignment** (TRAIN, COMAI, MedPerf, COFE), **§8 Cross-Vertical Comparison** (SDC / ag-mining / healthcare), implementation status table (IMPL vs SPEC), worked examples with tessera-core API, “When Tessera Is Not the Right Fit,” pilot status table, and numbered Sources & References. See repository docs or `Tessera_Healthcare_Medical_Imaging_Use_Case.pdf`.

---

## 4. What Has to Be True for It to Be a Real Benefit

- **The bottleneck is cross-architecture transfer.** If most deployments use same-architecture distillation or single-vendor pipelines and that’s sufficient, Tessera’s extra value is smaller. Benefit is largest where organizations **need** to move capability across **different** model families or vendors (e.g. ViT ↔ CNN, cloud model → edge, vendor A → vendor B, site A → site B).

- **“Behavior” or “capability” as a shareable artifact.** Consortia and regulators already care about model evaluation and controlled updates. A **registry or marketplace of Tessera tokens** (e.g. “cardiac CT segmentation v2,” “scanner-agnostic lesion detector”) only takes off if the sector values exchanging “knowledge” in token form, with provenance, rather than only sharing data or full weights.

- **Validation in-domain.** Demonstrating transfer between, e.g., a ViT and a small CNN (or 3D CNN → 2D lightweight model) for a concrete medical imaging task (even at small scale or on public data) would make the healthcare benefit concrete—and support discussions with regulators (e.g. how a PCCP might reference token-based updates).

- **Alignment with regulatory expectations.** PCCP and lifecycle guidance emphasize methodology, impact assessment, and traceability. Tessera’s drift metric and token provenance should be framed as supporting (not replacing) existing QMS and FDA expectations.

---

## 5. When Tessera Is Not the Right Fit

Tessera adds value when architectures differ and privacy or bandwidth constrain direct model or data sharing. In these scenarios, simpler approaches are preferable:

- **Same-vendor, same-scanner deployments** — If all sites run identical equipment and software, standard vendor deployment is simpler; hub-space translation adds no value.
- **Sufficient data for local training** — If a site has enough local data to achieve acceptable performance on its own equipment, single-site training avoids cross-architecture transfer complexity.
- **Real-time inference pipelines** — Tessera transfers learnt knowledge, not real-time streams; applications requiring live cross-device data fusion need direct data pipelines.
- **Regulatory requirement for bit-exact models** — Where regulations demand the deployed model match the validated model bit-for-bit, full-weight deployment is required; tokens produce functionally equivalent but not bit-identical models.
- **Single-architecture federated learning** — If all sites run the same architecture and have bandwidth for gradient exchange, standard FL may be the simpler choice.
- **No clinical validation yet** — Tessera has not been validated in a clinical trial with patient outcomes; deployment must include appropriate human oversight and institutional validation until pilots are completed.

---

## 6. Testing the Hypothesis

| Criterion | Healthcare / Medical imaging |
|----------|------------------------------|
| **Multiple model architectures in use?** | Yes (CNN, ViT, hybrid ViT-CNN, U-Net variants; lightweight vs heavy). |
| **Edge vs cloud / heavy vs light?** | Yes (edge for PHI and latency; cloud for training; distillation/compression to edge). |
| **Cross-arch transfer need plausible?** | Yes (ViT/3D CNN in cloud → small CNN or quantized ViT on edge; different vendors; site-to-site). |
| **Data scarcity / generalization?** | Yes (multi-site domain shift, scanner/protocol differences; need for capability transfer without raw data). |
| **Sharing / consortium plausible?** | Yes (TRAIN, COMAI, MedPerf, COFE; federated benchmarking and responsible AI). |
| **Auditability / lineage valued?** | Yes (FDA PCCP, lifecycle guidance, hospital QA, liability). |
| **Privacy / PHI constraints?** | Yes (tokens encode behavior, not raw data; DP optional)—strong fit. |

**Verdict:** The hypothesis holds under current research. Healthcare and medical imaging exhibit: (1) architecture and deployment diversity, (2) edge/cloud and heavy/light splits, (3) existing use of knowledge distillation and federated learning, (4) severe multi-site and cross-vendor domain shift, (5) strong regulatory and audit expectations, and (6) consortia focused on evaluation and responsible sharing. There is no fundamental reason Tessera would not apply; the main gap is **in-domain proof** (e.g. one pilot on a public or synthetic medical imaging task with measurable transfer quality and a short “regulatory alignment” note).

**Example scenarios (detailed in industry brief):** Cross-vendor chest X-ray (GE→Siemens/Canon), hospital-to-hospital lung nodule sharing, rural specialist AI (INT8 tokens over limited bandwidth), cross-vendor stroke detection; pathology cross-scanner whole-slide analysis, foundation model→edge (e.g. ViT-Large→EfficientNet), multi-site tumour classification (e.g. Gleason) with SWAP.

---

## 7. Cross-Vertical Comparison: Healthcare vs Autonomous Driving and Agriculture/Mining

Tessera’s use cases span autonomous driving (SDC), agriculture/mining, and healthcare. The table below shows why Tessera fits healthcare **relative to other verticals**: same need for cross-architecture transfer and portable tokens, with **stronger** regulation and privacy and **more severe** multi-site/domain shift.

| Dimension | Autonomous driving | Agriculture & mining | Healthcare & medical imaging |
|----------|--------------------|----------------------|-----------------------------|
| **Architecture diversity** | High (perception, prediction, planning; many vendors). | High (CNN, ViT, hybrid; edge vs cloud). | High (CNN, ViT, hybrid; edge vs cloud; scanner/vendor variance). |
| **Edge vs cloud** | Yes (car vs data center). | Yes (tractor/drone/LHD vs cloud). | Yes (clinic/hospital edge vs cloud; PHI drives edge). |
| **Regulation / audit** | Strong (safety-critical). | Present (safety in mining; compliance in ag). | **Very strong** (FDA, PCCP, lifecycle; audit trail expected). |
| **Data privacy** | Important. | Important (cooperatives, site data). | **Critical** (PHI; tokens as “no raw data” fit). |
| **Sharing / consortium** | Emerging (OEMs, suppliers). | Emerging (cooperatives, Agri-Gaia, OEM-agnostic). | Active (TRAIN, COMAI, MedPerf, COFE). |
| **Domain shift / multi-site** | Present (OEM, geography). | Present (region, site, equipment). | **Severe** (scanner, protocol, hospital, population). |

**Why Tessera fits healthcare relative to other verticals:** Healthcare has the same core drivers as SDC and ag/mining—heterogeneous vendors, edge/cloud split, and need to share “capability” without sharing raw data—plus **higher** regulatory and audit expectations (FDA, PCCP, QMS) and **critical** PHI constraints. Multi-site and cross-scanner domain shift in medical imaging are especially severe (e.g. AUC drops across GE vs Philips vs Siemens), making portable, architecture-agnostic tokens a strong fit. Consortia (TRAIN, MedPerf, COFE) are already focused on evaluation and responsible sharing, so token-based capability exchange aligns with existing initiatives.

---

## 8. Recommendations

1. **Document** this use case in the repo (e.g. link from README or a “Use cases” index) alongside autonomous driving and agriculture/mining.
2. **Pilot** one medical imaging scenario (e.g. public dataset: chest X-ray or histopathology; transfer ViT → small CNN or 3D→2D lightweight) to validate transfer quality and document drift; add a short “Regulatory alignment” subsection (PCCP, lifecycle, provenance) for stakeholder discussions.
3. **Engage** with a healthcare AI consortium (e.g. TRAIN, MedPerf, or a clinical partner) to see if token-based capability sharing or federated evaluation could align with their roadmap.
4. **Regulatory** (medium term): Draft a 1–2 page “Tessera and FDA PCCP / lifecycle” note describing how token-based updates could fit within predetermined change control and impact assessment—for use in partner or regulator conversations.

---

## 9. References (summary)

- **Architectures:** Springer (ViT vs CNN systematic review), medRxiv/PMC (hybrid ViT-CNN), MDPI (U-Net and transformers), Monash survey (transformers in medical imaging).
- **Edge/cloud:** SPIE/PMC (browser-based edge, serverless), Springer (edge deep learning survey), MDPI (Kvasir-Capsule edge ViT), arXiv (BitMedViT, EdgeViTs).
- **Transfer/distillation:** Nature (federated cardiac CT, knowledge-distilled transformer), Stanford (federated multi-organ, distillation), arXiv (multimodal SSL, domain adaptation).
- **Domain shift / multi-site:** Nature (MRI scanner manufacturers), FedMedICL/MICCAI (distribution shifts), Nature (multi-hospital chest X-ray), Frontiers (generalization vs site-specific).
- **Regulation:** FDA (PCCP final guidance, lifecycle draft), Lancet Digital Health (510(k) predicate networks), RSNA (FDA review process), PMC (radiology AI FDA review).
- **Consortia:** TRAIN (Microsoft/news), COMAI (NYU), Nature (MedPerf), MLCommons (COFE).
- **Vendors/heterogeneity:** KLAS Imaging AI 2024, Springer (vendor survey), Nature (scanner manufacturer effects).

**Key evidence (see industry brief for full citations):** Cross-vendor AUC drop (e.g. 0.74→0.60 Siemens vs Philips) [e.g. arxiv 2407.18060]; federated learning real-world deployment &lt;6% of studies [PMC10897620]; FDA PCCP draft guidance (2025) for AI-enabled device software; 1,250+ FDA-authorised AI/ML devices (2025).

---

## 10. Keeping This Document Current

- **Citations:** Update the reference list and any in-text numbers when new market data, regulatory guidance (e.g. FDA PCCP final vs draft, lifecycle guidance), or peer-reviewed evidence (e.g. cross-vendor studies, FL deployment rates) become available. The industry brief (PDF) maintains a numbered Sources & References section—keep that list and in-text [1]–[n] in sync with this summary.
- **Implementation table:** If the healthcare brief includes an “Implementation status” table (IMPL vs SPEC), update it when the tessera-core codebase or protocol changes (e.g. a new mode or component is implemented or specified). The table should reflect the current state of `tessera-core` and the protocol spec so readers and partners are not misled.
- **FDA guidance:** FDA guidance on AI-enabled devices (PCCP, lifecycle, SaMD) is evolving. When the agency issues final guidance or new drafts, update the regulatory section of the industry brief and any “PCCP alignment” or “regulatory pathway” claims in this doc so they remain accurate.
