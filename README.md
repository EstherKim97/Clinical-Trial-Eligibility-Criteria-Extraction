# Clinical Trial Eligibility Criteria Extraction Pipeline

Automated NLP pipeline for extracting and analyzing eligibility criteria from ClinicalTrials.gov. Demonstrates clinical domain expertise and NLP capabilities for pharmaceutical data science roles.

## 🎯 Project Overview

**Problem:** Clinical development teams spend weeks manually reviewing competitor trial protocols to benchmark eligibility criteria—biomarker requirements, lab test thresholds, and prior therapy restrictions.

**Solution:** Automated extraction pipeline using pattern-based NLP to extract structured medical entities from unstructured eligibility text.

**Impact:** Reduces protocol benchmarking time from 2 weeks → 5 minutes.

---

## 🔬 Key Features

- **Strategic trial selection**: Disease + treatment combinations (immunotherapy, targeted therapy, advanced disease)
- **Multi-category extraction**: Diseases, biomarkers, drugs, lab tests, procedures
- **Quantitative criteria capture**: Numerical thresholds (e.g., "serum creatinine ≤1.5", "ANC ≥1,500/μL")
- **Biomarker intelligence**: Actionable mutations (EGFR sensitizing, BRAF V600E, PD-L1 expression variants)
- **Treatment classification**: Immunotherapy vs. targeted therapy vs. chemotherapy
- **Clinical assay recognition**: PD-L1 SP-142, tumor proportion score

---

## 📊 Results Summary

**Analysis of 40 oncology clinical trials:**

### Entities Extracted:
- **~200+ total medical entities** across 5 categories
- **18 unique biomarkers** including:
  - EGFR sensitizing mutations
  - BRAF V600E mutation  
  - PD-L1 expression (6 variations: expression, status, TPS, SP-142 assay)
  - ALK translocation
  - HER2 status
  - MSI-high/dMMR status
  - Targetable genomic aberrations

### Key Drugs Identified:
- Checkpoint inhibitors: Pembrolizumab, Nivolumab
- Treatment regimens: Platinum-based chemotherapy
- Treatment classes: Targeted therapy, radiation therapy

### Lab Test Benchmarks:
- Absolute neutrophil count (appears in 10+ trials)
- Creatinine clearance (standard: ≥60 mL/min)
- Serum creatinine ≤1.5 (quantitative threshold)
- Total bilirubin, platelet count
- ECOG performance status (standard: 0-1)

### Disease Specificity:
- Non-small cell lung cancer (NSCLC)
- Small cell lung cancer (SCLC)  
- Metastatic disease
- Renal cell carcinoma

---

## 🛠️ Technical Stack

**Core Technologies:**
- **Language**: Python 3.12
- **Data Source**: ClinicalTrials.gov API v2
- **NLP Approach**: Regex-based pattern matching
- **Libraries**: pandas, requests, matplotlib, seaborn, json

**Skills Demonstrated:**
- API integration and data collection
- Text preprocessing and cleaning
- Medical terminology and clinical trial knowledge
- Pattern-based entity extraction
- Data visualization and insight generation

---

## 📁 Project Structure

```
clinical-trial-nlp/
├── collect_trials.py          # Data collection from ClinicalTrials.gov API
├── preprocess.py                 # Text cleaning & criteria structuring  
├── extract_entities_r.py     # Entity extraction with regex patterns
├── requirements.txt              # Python dependencies
├── data/
│   ├── raw/
│   │   └── api_trials.csv       # Raw API responses
│   └── processed/
│       └── processed_trials.csv  # Cleaned & structured criteria
└── results/
    ├── extraction_results.json   # Structured entity extraction output
    └── visualizations/           # Generated charts
        ├── entity_frequency.png
        ├── entity_types.png
        ├── entity_distribution.png
        ├── biomarker_landscape.png
        ├── treatment_classes.png

```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/clinical-trial-nlp.git
cd clinical-trial-nlp

# Install dependencies
pip install -r requirements.txt
```

### Run Pipeline

```bash
# Step 1: Collect trials from ClinicalTrials.gov (~2-3 minutes)
python collect_trials.py

# Step 2: Preprocess eligibility criteria text
python preprocess.py

# Step 3: Extract medical entities and get visualization
python extract_entities_r.py
```

### Expected Output

```
results/
├── extraction_results.json       # Structured entity data
└── visualizations/
    ├── entity_frequency.png
    ├── entity_types.png
    ├── biomarker_landscape.png   # Color-coded biomarker requirements
    ├── treatment_classes.png     # Immunotherapy vs targeted vs chemo

```

---

## 💡 Use Cases

### For Clinical Trial Designers (Primary Use Case)
- **Benchmark biomarker cutoffs**: "What HER2 thresholds do competitors use?"
- **Compare lab requirements**: "Is creatinine clearance ≥60 or ≥50 standard?"
- **Identify ECOG standards**: "Do most trials require 0-1 or allow 0-2?"

### For Medical Writers
- **Standard eligibility language**: Evidence-based protocol writing
- **Organ function definitions**: AST/ALT ≤2.5× ULN vs ≤3× ULN

### For Regulatory Affairs
- **Competitive intelligence**: Are our criteria too restrictive?
- **Industry alignment**: Do approved trials allow treated brain metastases?

---

## 🧠 Technical Design Decisions

### Why Regex Instead of Transformer Models?

**Initial Approach**: Attempted SciBERT and spaCy biomedical NER models

**Challenges Encountered**:
- Python 3.12 compatibility issues with spaCy (C++ compilation errors)
- SciBERT extracted generic tokens ("the", "or") instead of medical concepts
- Generic biomedical NER models produced useless labels (LABEL_0, LABEL_1)

**Final Solution**: Regex-based pattern matching

**Advantages**:
- ✅ No dependency installation issues
- ✅ Predictable, controllable extraction
- ✅ Fast performance (~5 min for 40 trials)
- ✅ Easy customization for clinical terminology
- ✅ Captures complete multi-word phrases ("non-small cell lung cancer")
- ✅ Extracts quantitative thresholds ("ANC ≥1,500/μL")

**Key Learning**: Pragmatic problem-solving—when sophisticated ML fails, simple solutions that work are more valuable than complex solutions that don't.

### Search Strategy: Method 3 (Disease + Treatment Combinations)

**Why not search "lung cancer" or "EGFR mutation"?**

Chose **disease + treatment combinations** because:
- More realistic (mimics how clinicians search)
- Captures diverse biomarkers naturally
- Gets complete clinical picture (disease + biomarker + treatment + labs)
- One search returns multiple entity types

**Example searches**:
- "immunotherapy lung cancer" → captures PD-L1, pembrolizumab, prior platinum therapy
- "HER2 targeted therapy" → captures HER2-positive, trastuzumab, LVEF requirements
- "triple negative breast cancer treatment" → captures ER/PR/HER2-negative status

---
## 💼 Example Use Case

### Scenario: Benchmarking Eligibility Criteria for Advanced Cancer Trials

**Research Question:**  
*"What are the standard eligibility criteria for advanced cancer trials using immunotherapy or targeted therapy?"*

**Keywords Used:**
```python
conditions = [
    # Immunotherapy trials
    'immunotherapy lung cancer',
    'immunotherapy melanoma',
    'checkpoint inhibitor NSCLC',
    'PD-1 therapy breast cancer',
    
    # Targeted therapy trials
    'targeted therapy breast cancer',
    'targeted therapy lung cancer',
    'HER2 targeted therapy',
    'EGFR inhibitor lung cancer',
    
    # Advanced/metastatic disease
    'chemotherapy advanced breast cancer',
    'hormone therapy metastatic breast cancer',
    'combination therapy melanoma',
    'metastatic colorectal cancer treatment',
    'advanced renal cell carcinoma therapy',
    'recurrent ovarian cancer treatment',
    
    # Disease subtypes
    'triple negative breast cancer treatment',
    'BRAF mutant melanoma',
    'MSI-high colorectal cancer'
]
```

**Results Interpretation:**

From analysis of 40 trials, the following patterns emerged:

**1. Most Common Eligibility Criteria (Top Medical Entities)**

Looking at the entity frequency chart:
- **Disease specificity**: Trials distinguish between NSCLC and SCLC (not just "lung cancer")
- **Lab safety tests**: Absolute neutrophil count, creatinine clearance, serum creatinine, total bilirubin, platelet count appear frequently
- **Specific drugs**: Checkpoint inhibitors (pembrolizumab, nivolumab) are explicitly mentioned
- **Quantitative thresholds**: "Serum creatinine ≤1.5" captured as a specific eligibility cutoff
- **Treatment context**: "Prior systemic therapy," "platinum-based chemotherapy," and "radiation therapy" indicate treatment history requirements

**Insight:** Eligibility criteria are highly specific—trials don't just say "cancer patients," they specify disease subtypes, exact drug names, and numerical lab thresholds.

**2. Entity Category Distribution**

The entity type distribution shows:
- **Lab tests** (~77 entities): Most common category, confirming safety labs are universal
- **Diseases** (~68 entities): High frequency reflects disease-specific trial design
- **Drugs** (~46 entities): Treatment history and drug restrictions are important
- **Biomarkers** (~18 entities): Present but less frequent than lab tests
- **Procedures** (~8 entities): Biopsies and prior treatments mentioned

**Insight:** Lab safety requirements dominate eligibility criteria. While biomarker testing is present (18 unique markers), it's less ubiquitous than basic safety labs, suggesting not all trials require biomarker selection.

**3. Biomarker Landscape by Therapeutic Area**

The color-coded biomarker chart reveals:
- **PD-L1 biomarkers** (blue - immunotherapy): Multiple variations extracted (expression, status, TPS, SP-142 assay)
- **Driver mutations** (coral - targeted therapy): EGFR sensitizing mutations, BRAF V600E, ALK translocation
- **HER2 markers** (green): HER2 status, HER2-targeting
- **MSI/MMR** (gold): Microsatellite instability-high, mismatch repair deficient

**Insight:** Biomarker requirements align with therapeutic modality—immunotherapy trials test PD-L1, targeted therapy trials test driver mutations (EGFR, BRAF, ALK). This demonstrates precision medicine approach where patient selection depends on molecular characteristics.

**4. Treatment Modality Breakdown**

The treatment class distribution shows:
- **Immunotherapy mentions**: Checkpoint inhibitors, PD-1/PD-L1 therapies
- **Targeted therapy mentions**: HER2-targeted, EGFR inhibitors, specific molecular targets
- **Chemotherapy mentions**: Platinum-based regimens, traditional cytotoxic agents

**Insight:** The dataset reflects modern oncology's shift toward immunotherapy and targeted therapy, while chemotherapy remains relevant (often as prior therapy requirement or combination treatment).

**Actionable Conclusions:**

For someone designing a similar trial, this analysis provides:
- ✅ **Common lab requirements**: ANC, creatinine, bilirubin, platelets are standard
- ✅ **Disease specificity standard**: Use precise disease subtypes (NSCLC vs SCLC), not generic terms
- ✅ **Biomarker-treatment alignment**: Match biomarker testing to treatment mechanism (PD-L1 for immunotherapy, driver mutations for targeted therapy)
- ✅ **Treatment landscape context**: Understanding which therapies are commonly studied helps position new trials competitively

**Limitations:**

- Dataset is lung cancer-heavy due to keyword selection
- For disease-specific benchmarking (e.g., breast cancer trials only), re-run with targeted keywords
- Biomarker count (18) suggests many trials don't specify biomarkers in eligibility text, or use generic terms like "adequate organ function"

---

## 📈 Sample Visualizations

### Top Medical Entities
<img width="1800" height="1200" alt="entity_frequency" src="https://github.com/user-attachments/assets/e521abd9-db16-401f-95cb-56c037700fbf" />
*Most frequently mentioned medical entities across 40 clinical trials, showing specific disease subtypes (NSCLC, SCLC), checkpoint inhibitors (pembrolizumab, nivolumab), complete lab test names, and quantitative thresholds like "serum creatinine ≤1.5"*

### Entity Type Distribution
<img width="1500" height="900" alt="entity_types" src="https://github.com/user-attachments/assets/9c72b02c-dda9-4746-9e40-364f8b37c078" />
*Distribution of extracted entities by category: Lab tests and diseases dominate eligibility criteria, with drugs, biomarkers, and procedures also captured*

### Biomarker Landscape
<img width="1800" height="1200" alt="biomarker_landscape" src="https://github.com/user-attachments/assets/d943052b-bf7f-42dc-a495-459016e54082" />
*Color-coded by therapeutic area: PD-L1 (immunotherapy), driver mutations (targeted therapy), HER2, MSI/MMR*

### Treatment Modality Distribution  
<img width="1500" height="900" alt="treatment_classes" src="https://github.com/user-attachments/assets/67971bdc-51e6-44d2-a69d-fcccb5d6857f" />
*Immunotherapy vs. Targeted Therapy vs. Chemotherapy mentions*


---

## 🔮 Future Enhancements

**Data Expansion:**
- Scale to 500+ trials for statistical significance
- Add temporal analysis (eligibility criteria trends 2020-2025)
- Include international trials (EudraCT, PMDA)

**Technical Improvements:**
- Integrate LLM-based summarization (GPT-4/Claude API)
- Build interactive dashboard (Streamlit/Plotly Dash)
- Add biomarker-disease association matrix
- Implement cross-trial patient matching

**Clinical Insights:**
- Enrollment restrictiveness scoring
- Geographic variation analysis (US vs EU criteria)
- Sponsor-specific patterns (pharma vs academic trials)

---

- **Data Source**: ClinicalTrials.gov (U.S. National Library of Medicine)
- **Clinical Expertise**: Informed by 4 years in pharmaceutical regulatory affairs
- **Inspiration**: Real-world protocol benchmarking workflows in clinical development teams
