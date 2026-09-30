"""
Rich Industry Sample Corpus for Document Intelligence & RAG System.
Provides 4 domain-specific documents: Finance 10-K, Clinical Protocol, SaaS Legal MSA, and AI Whitepaper.
"""

SAMPLE_DOCUMENTS = [
    {
        "filename": "AlphaTech_Cloud_10K_Annual_Report.md",
        "title": "AlphaTech Cloud Enterprise - Form 10-K Annual Report (FY 2025)",
        "content": """# AlphaTech Cloud Enterprise - Form 10-K Annual Report (FY 2025)

## 1. Executive Financial Summary & Overview
AlphaTech Cloud Enterprise Inc. (NASDAQ: ATCE) is a global provider of enterprise cloud infrastructure, AI-accelerated compute clusters, and document intelligence platforms. For the fiscal year ended December 31, 2025, the company reported total consolidated revenue of $1,420,000,000 ($1.42B), representing a 38.2% year-over-year growth compared to $1,027,500,000 in FY 2024.

Gross profit for the fiscal year reached $1,028,080,000, achieving a GAAP gross margin of 72.4%, up from 69.8% in the preceding fiscal period. Adjusted EBITDA was $310,400,000, reflecting an Adjusted EBITDA margin of 21.9%.

| Financial Metric | FY 2024 (USD) | FY 2025 (USD) | YoY Growth (%) |
| :--- | :--- | :--- | :--- |
| **Total Revenue** | $1,027,500,000 | $1,420,000,000 | +38.2% |
| **Subscription Cloud Revenue** | $865,000,000 | $1,245,000,000 | +43.9% |
| **Professional Services** | $162,500,000 | $175,000,000 | +7.7% |
| **GAAP Gross Margin** | 69.8% | 72.4% | +260 bps |
| **Adjusted EBITDA** | $195,200,000 | $310,400,000 | +59.0% |
| **Free Cash Flow** | $142,800,000 | $238,500,000 | +67.0% |

## 2. Segment Revenue & Customer Retention
Subscription revenue comprised 87.7% of total revenue in FY 2025. Annual Recurring Revenue (ARR) reached $1.34B at year-end.
- **Enterprise Cohort (> $100k ARR)**: 1,842 enterprise customers, an increase of 412 customers (+28.8%).
- **Dollar-Based Net Retention Rate (NRR)**: 124.5% as of Q4 2025, driven by enterprise expansion into Vector Intelligence APIs and GPU Inference Pipelines.
- **Geographic Breakdown**: North America accounted for 64% ($908.8M), EMEA accounted for 24% ($340.8M), and APAC accounted for 12% ($170.4M).

## 3. Balance Sheet, Liquidity & Debt Covenants
As of December 31, 2025, AlphaTech held $645,000,000 in cash, cash equivalents, and short-term marketable securities.
- **Senior Secured Term Loan**: Outstanding principal balance of $350,000,000 due November 2029 with interest at SOFR + 2.25%.
- **Financial Covenants**: Under the Credit Agreement, AlphaTech must maintain a maximum Total Net Leverage Ratio of 3.25:1.00 and a minimum Interest Coverage Ratio of 4.00:1.00. As of December 31, 2025, the Total Net Leverage Ratio was 0.88:1.00 and the Interest Coverage Ratio was 9.42:1.00, in full compliance with all credit terms.

## 4. Key Risk Factors & Market Outlook
1. **GPU Hardware Supply Constraints**: Dependence on advanced semiconductor foundries and GPU interconnect suppliers. Allocation delays could impact customer onboarding timelines.
2. **Regulatory & AI Data Governance**: Compliance with European Union AI Act (Regulation EU 2024/1689), US Executive Orders on AI Safety, and GDPR cross-border data transfer mechanisms.
3. **Cybersecurity & Infrastructure Resilience**: Ongoing capital expenditures in zero-trust architecture, automated SOC-2 Type II audits, and FedRAMP High certification pathways.
"""
    },
    {
        "filename": "Biocura_Oncology_Clinical_Protocol_Phase3.md",
        "title": "Biocura Therapeutics - Phase III Oncology Protocol (BC-809)",
        "content": """# Biocura Therapeutics - Phase III Oncology Protocol (BC-809)

## 1. Study Overview & Therapeutic Rationale
Protocol Number: BCT-ONC-302  
Study Title: A Multi-Center, Randomized, Double-Blind, Placebo-Controlled Phase III Trial of BC-809 (Veloctamab) in Combination with Standard Chemotherapy in Patients with Advanced Metastatic Non-Small Cell Lung Cancer (NSCLC) Harboring KRAS G12C Mutations.

BC-809 is a high-affinity, bispecific monoclonal antibody designed to selectively target oncogenic signaling while enhancing tumor-infiltrating lymphocyte (TIL) proliferation.

## 2. Primary and Secondary Endpoints
- **Primary Endpoint**: Progression-Free Survival (PFS) assessed by Blinded Independent Central Review (BICR) using RECIST v1.1 criteria.
  - *Result / Target*: Median PFS of 14.2 months in the BC-809 arm compared to 8.4 months in the control arm (Hazard Ratio [HR] = 0.58; 95% CI: 0.46–0.73; p < 0.001).
- **Secondary Endpoints**:
  - **Overall Survival (OS)**: Median OS was 22.8 months for the BC-809 group versus 16.1 months for standard therapy (HR = 0.66; p = 0.002).
  - **Objective Response Rate (ORR)**: Confirmed ORR of 64.2% in the investigational group versus 38.6% in the placebo group.
  - **Duration of Response (DoR)**: Median DoR was 15.6 months versus 7.8 months.

## 3. Patient Eligibility & Stratification Criteria
### Inclusion Criteria:
1. Histologically or cytologically confirmed metastatic or locally advanced Stage IV NSCLC with documented KRAS G12C mutation confirmed by next-generation sequencing (NGS).
2. Age $\ge 18$ years with an Eastern Cooperative Oncology Group (ECOG) Performance Status score of 0 or 1.
3. At least one measurable lesion according to RECIST v1.1 guidelines.
4. Adequate bone marrow and organ function: Absolute Neutrophil Count (ANC) $\ge 1.5 \times 10^9/\text{L}$, Platelets $\ge 100 \times 10^9/\text{L}$, and Serum Creatinine $\le 1.5 \times \text{ULN}$.

### Exclusion Criteria:
1. Active, untreated central nervous system (CNS) metastases or leptomeningeal disease.
2. Prior treatment with KRAS G12C covalent inhibitors or immune checkpoint inhibitors within 28 days of randomization.
3. History of autoimmune pneumonitis or interstitial lung disease requiring systemic steroids.

## 4. Dosing Schedule & Safety Profile
- **Investigational Arm**: BC-809 administered at 250 mg orally twice daily (BID) continuously in 21-day treatment cycles in combination with intravenous Carboplatin (AUC 5) and Pemetrexed (500 mg/m²).
- **Safety & Adverse Event Rates**:
  - Most common Treatment-Emergent Adverse Events (TEAEs): Grade 1/2 fatigue (42.1%), mild nausea (36.4%), and transient maculopapular rash (24.8%).
  - Grade 3/4 TEAEs: Neutropenia observed in 18.5% of patients (managed via G-CSF support), and elevated AST/ALT in 5.2% of patients (resolved following temporary dose reduction to 150 mg BID).
  - Treatment discontinuation due to adverse events was 4.1% in the BC-809 cohort compared to 3.8% in the control group.
"""
    },
    {
        "filename": "Enterprise_SaaS_Master_Services_Agreement_SLA.md",
        "title": "CloudCore Systems - Enterprise SaaS Master Services Agreement & SLA",
        "content": """# CloudCore Systems - Enterprise SaaS Master Services Agreement & SLA

## 1. Services & Service Level Agreement (SLA) Commitments
This Master Services Agreement ("Agreement") is executed between CloudCore Systems Inc. ("Provider") and Enterprise Customer ("Customer"). Provider shall guarantee a Monthly Uptime Percentage of at least **99.9%** ("Uptime Commitment") for all core production APIs, vector query services, and ingestion pipelines, calculated over each calendar month, 24 hours per day, 7 days per week.

### Service Credit Schedule:
| Monthly Uptime Percentage | Service Credit Percentage (% of Monthly Recurring Fee) |
| :--- | :--- |
| **99.0% to < 99.9%** | 10% Service Credit |
| **95.0% to < 99.0%** | 25% Service Credit |
| **< 95.0%** | 50% Service Credit |

Service credits must be claimed by Customer within thirty (30) days following the end of the affected calendar month and shall be credited against future invoices.

## 2. Maintenance Windows & Exclusions
The following shall be excluded from the calculation of Monthly Uptime Percentage:
1. **Scheduled Maintenance**: Planned maintenance windows conducted between 00:00 and 04:00 UTC on Sundays, provided Provider gives at least forty-eight (48) hours prior written electronic notice.
2. **Emergency Maintenance**: Critical security patches or infrastructure mitigation, limited to no more than two (2) hours per month.
3. **Customer Force Majeure**: Downtime resulting from Customer's network failures, unauthorized modifications, or third-party DNS provider outages outside Provider's direct control.

## 3. Limitation of Liability & Financial Cap
TO THE MAXIMUM EXTENT PERMITTED BY LAW:
- **Aggregate Liability Cap**: In no event shall either party's aggregate cumulative liability arising out of or related to this Agreement exceed the total fees actually paid by Customer to Provider in the twelve (12) months preceding the incident giving rise to liability.
- **Consequential Damages Waiver**: Neither party shall be liable for any lost profits, loss of business, indirect, special, incidental, punitive, or consequential damages.
- **Uncapped Liabilities**: The liability limitations above shall NOT apply to: (a) breach of confidentiality obligations under Section 7; (b) gross negligence or willful misconduct; or (c) indemnification obligations for third-party intellectual property infringement under Section 8.

## 4. Data Security, Privacy & Compliance (SOC-2 & GDPR)
Provider shall maintain physical, administrative, and technical safeguards conforming to SOC-2 Type II standards and ISO 27001 certifications.
- All customer data at rest shall be encrypted using AES-256 GCM.
- All data in transit across public networks shall be encrypted using TLS 1.3.
- In the event of a confirmed Security Incident involving unauthorized access to Customer Personal Data, Provider shall notify Customer in writing within twenty-four (24) hours of verification.

## 5. Term, Termination & Cure Period
- **Term**: Initial term of thirty-six (36) months, automatically renewing for successive twelve (12) month periods unless either party provides ninety (90) days prior written notice of non-renewal.
- **Termination for Cause**: Either party may terminate this Agreement immediately if the other party breaches any material provision and fails to cure such breach within thirty (30) days of receiving formal written notice.
"""
    },
    {
        "filename": "Autonomous_Agents_and_Graph_RAG_Architecture.md",
        "title": "Next-Gen Document Intelligence: Hybrid RAG & Graph Vector Architecture",
        "content": """# Next-Gen Document Intelligence: Hybrid RAG & Graph Vector Architecture

## 1. Abstract & System Architecture
Modern Document Intelligence systems require bridging unstructured document hierarchies with dense semantic retrieval and lexical keyword precision. Pure dense vector search frequently fails on domain-specific numerical queries (e.g., balance sheet line items, clinical dosage numbers, contract clauses). To resolve this, our architecture unifies:
1. **Structure-Aware Document Decomposition**: Preserving markdown headers, tables, and bounding entities.
2. **Dual-Path Hybrid Indexing**: High-dimensional dense embeddings paired with an inverted Okapi BM25 sparse index.
3. **Reciprocal Rank Fusion (RRF)**: Combining multi-retriever rankings with smoothing parameter $k=60$.
4. **Contextual Cross-Encoder Re-Ranking**: Dynamically evaluating token overlap, exact phrase match, and query intent.
5. **Grounded Provenance Generation**: Ensuring every factual claim in the synthesized answer is explicitly mapped to verifiable citation tags.

```
[Raw Document] --> [Structure Parser] --> [Semantic Chunking]
                                                 |
         +---------------------------------------+---------------------------------------+
         |                                                                               |
         v                                                                               v
[Dense Vector Embeddings]                                                      [Okapi BM25 Index]
(Sublinear TF-IDF / Cosine Sim)                                                (Sparse Lexical Search)
         |                                                                               |
         +---------------------------------------+---------------------------------------+
                                                 |
                                                 v
                                   [Reciprocal Rank Fusion (RRF)]
                                                 |
                                                 v
                                [Cross-Encoder Context Reranker]
                                                 |
                                                 v
                                 [Grounded LLM Prompt Assembly]
                                                 |
                                                 v
                                  [SSE Stream + Provenance Tags]
```

## 2. Reciprocal Rank Fusion (RRF) Formulation
Reciprocal Rank Fusion computes a robust consensus ranking across disparate retrieval engines without requiring manual score normalization. For any document or chunk $d$:

$$RRF(d) = \sum_{m \in M} \frac{w_m}{k + r_m(d)}$$

Where:
- $M = \{\text{Dense Vector}, \text{BM25 Lexical}\}$ is the set of retrieval models.
- $w_m$ is the assigned retriever weight (typically $w_{dense} = 0.50, w_{bm25} = 0.50$).
- $k = 60$ is the rank smoothing constant preventing top-ranked single-engine anomalies from dominating.
- $r_m(d)$ is the 1-based rank position of chunk $d$ in retriever $m$.

## 3. Memory Hierarchy in Multi-Agent Autonomous RAG
To support multi-turn conversational reasoning, the system maintains a 3-tier memory hierarchy:
- **L1 Working Memory**: In-context sliding window containing recent user dialogue turns and active citations.
- **L2 Episodic Memory**: Dynamic vector cache of previous question-answer pairs and user preferences with exponential decay.
- **L3 Semantic Ground Truth**: Persistent hybrid index of the verified document corpus.

## 4. Grounded Citation Verification & Hallucination Mitigation
To eliminate hallucination, the generation pipeline enforces strict verification:
1. Every claim must directly reference `[Source X]` where $X$ corresponds to an indexed chunk provided in the grounding context.
2. Chunks with confidence scores below the calibrated threshold ($\tau = 0.20$) are discarded before prompt assembly.
3. If the retrieved context does not contain sufficient factual evidence to answer the query, the model explicitly acknowledges the absence of information rather than extrapolating.
"""
    }
]
