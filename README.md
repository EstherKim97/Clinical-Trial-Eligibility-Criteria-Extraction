# Clinical Trial Eligibility Criteria Extraction Pipeline

Automated NLP pipeline for extracting and analyzing eligibility criteria from ClinicalTrials.gov. Built to demonstrate clinical domain expertise and NLP capabilities for pharmaceutical data science roles.

## 🎯 Overview

**Problem:** Clinical teams spend weeks manually benchmarking eligibility criteria across competitor trials.

**Solution:** Automated entity extraction pipeline using pattern-based NLP.

**Impact:** Reduces protocol benchmarking from 2 weeks → 5 minutes.

## 📊 Results (40 Trials Analyzed)

- **200+ medical entities** extracted across 5 categories
- **18 unique biomarkers**: EGFR mutations, PD-L1 expression, BRAF V600E, HER2 status, MSI-high
- **Specific drugs**: Pembrolizumab, nivolumab, platinum-based chemotherapy
- **Quantitative thresholds**: "Serum creatinine ≤1.5" captured
- **Disease specificity**: NSCLC vs SCLC differentiation

## 🛠️ Tech Stack

Python 3.12 • ClinicalTrials.gov API v2 • Regex NLP • pandas • matplotlib • seaborn

## 🚀 Quick Start
```bash
# Install and run
pip install -r requirements.txt
python collect_api_FIXED.py      # Collect trials
python preprocess.py              # Clean text
python extract_entities_regex.py # Extract entities
python create_visualizations.py  # Generate charts
```

## 📈 Visualizations

### Top Medical Entities
<img width="1800" height="1200" alt="entity_frequency" src="https://github.com/user-attachments/assets/e521abd9-db16-401f-95cb-56c037700fbf" />

### Entity Type Distribution
<img width="1500" height="900" alt="entity_types" src="https://github.com/user-attachments/assets/9c72b02c-dda9-4746-9e40-364f8b37c078" />

### Biomarker Landscape
<img width="1800" height="1200" alt="biomarker_landscape" src="https://github.com/user-attachments/assets/d943052b-bf7f-42dc-a495-459016e54082" />

### Treatment Modality Distribution  
<img width="1500" height="900" alt="treatment_classes" src="https://github.com/user-attachments/assets/67971bdc-51e6-44d2-a69d-fcccb5d6857f" />

## 💼 Example Use Case

**Question:** *"What are standard eligibility criteria for advanced cancer trials using immunotherapy or targeted therapy?"*

**Keywords Used:**
```python
# Disease + treatment combinations (17 searches)
'immunotherapy lung cancer'
'HER2 targeted therapy'
'EGFR inhibitor lung cancer'
'metastatic colorectal cancer treatment'
# ... etc
```

**Findings:**

**Lab Requirements (Universal)**
- Absolute neutrophil count, creatinine clearance, serum creatinine ≤1.5
- Total bilirubin, platelet count appear across most trials
- Insight: Lab safety criteria are foundational

**Biomarker Patterns (18 unique)**
- EGFR sensitizing mutations (lung cancer)
- PD-L1 expression variations (immunotherapy selection)
- BRAF V600E (melanoma), HER2 status (breast), MSI-high (colorectal)
- Insight: Biomarker requirements align with therapeutic modality

**Treatment Landscape**
- Checkpoint inhibitors: Pembrolizumab, nivolumab frequently mentioned
- Platinum-based chemotherapy: Common prior therapy requirement
- Insight: Reflects current oncology standard of care

**Actionable Conclusions:**
- ✅ Benchmark lab thresholds identified (serum creatinine ≤1.5)
- ✅ Disease-specific biomarker requirements mapped
- ✅ Standard treatment history patterns documented

## ⚠️ Limitations

**Manual Pattern Requirement**
- Requires explicit regex patterns for each entity (~60 biomarkers coded)
- Won't extract rare biomarkers (FGFR2, NRG1) unless patterns added
- *Solution*: Could use gene databases (OncoKB), NER models, or LLM APIs for automatic recognition

**Dataset Scope**
- Lung cancer-heavy due to keyword selection
- For disease-specific analysis, re-run with targeted keywords

**Quantitative Extraction**
- Many trials use "adequate organ function" without specific numbers
- Limits numerical threshold benchmarking

**Technical Trade-offs**
- Regex chosen after transformer models (SciBERT, spaCy) failed on Python 3.12
- Provides predictable extraction but requires pattern maintenance

## 🔮 Future Enhancements

- Automated entity recognition using UMLS or gene databases
- LLM integration (Claude/GPT API) for complex extractions
- Scale to 500+ trials for statistical significance
- Interactive dashboard (Streamlit)
