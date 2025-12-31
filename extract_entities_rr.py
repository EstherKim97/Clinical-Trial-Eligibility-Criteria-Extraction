"""
Medical Entity Extraction from Clinical Trial Eligibility Criteria

This script extracts structured medical entities from unstructured eligibility
criteria text using regex-based pattern matching.

Entity Categories:
- DISEASE: Cancer types, disease stages (e.g., "non-small cell lung cancer")
- DRUG: Specific drugs and treatment classes (e.g., "pembrolizumab", "checkpoint inhibitor")
- BIOMARKER: Molecular markers (e.g., "EGFR mutation", "PD-L1 expression ≥50%")
- LAB_TEST: Lab requirements with thresholds (e.g., "absolute neutrophil count ≥1,500/μL")
- PROCEDURE: Medical procedures (e.g., "tumor biopsy", "radiation therapy")

Design Decision:
Initially attempted transformer-based NER (SciBERT, spaCy) but encountered:
- Python 3.12 compatibility issues (C++ compilation errors)
- Poor extraction quality (generic tokens instead of medical concepts)
Regex approach provides:
- Predictable, controllable extraction
- No dependency issues
- Fast performance
- Easy customization for clinical terminology

Output: JSON file with extracted entities per trial + visualizations
"""

import pandas as pd
import json
import os
import re
from collections import Counter
import matplotlib.pyplot as plt
from transformers import pipeline

print("=" * 70)
print("Step 4: Medical Entity Extraction (Regex + BART)")
print("=" * 70)

# Load summarization model
print("\n⏳ Loading summarization model...")
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print("✓ Summarization model loaded (BART)")
except Exception as e:
    print(f"✗ Could not load summarizer: {e}")
    summarizer = None

print("\n✓ Ready!\n")

def extract_medical_entities_regex(text):
    """
    Extract medical entities from clinical trial eligibility criteria text.
    
    Uses regex patterns to identify complete medical phrases, not just single words.
    Patterns are designed to capture multi-word terms like "non-small cell lung cancer"
    rather than fragmenting into "non", "small", "cell", "lung", "cancer".
    
    Args:
        text (str): Raw eligibility criteria text (inclusion or exclusion)
        
    Returns:
        dict: Entity types mapped to lists of extracted entities
        Example: {
            'BIOMARKER': ['egfr sensitizing mutations', 'pd-l1 expression'],
            'DRUG': ['pembrolizumab', 'checkpoint inhibitor'],
            'LAB_TEST': ['absolute neutrophil count', 'serum creatinine ≤1.5']
        }
        
    Pattern Design Philosophy:
        - Match complete phrases: r'\b(non-small cell lung cancer)\b'
        - Allow flexible spacing/hyphens: r'\b(her2[- ]?positive)\b'
        - Capture numerical thresholds: r'\b(hemoglobin\s*≥?\s*\d+(?:\.\d+)?\s*g/dl)\b'
        
    Defensive Coding:
        - Handles NaN/None values
        - Normalizes whitespace
        - Filters incomplete extractions (e.g., "her2-" without rest of phrase)
    """
    # Handle NaN values
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return {}
    
    text = str(text)
    if len(text) < 20:
        return {}
    
    entities = {
        'DISEASE': set(),
        'DRUG': set(),
        'BIOMARKER': set(),
        'LAB_TEST': set(),
        'PROCEDURE': set()
    }
    
    # ========== DISEASE PATTERNS (Multi-word) ==========
    # Extract complete disease descriptions, not single words
    # Example: "metastatic breast cancer" NOT ["metastatic", "breast", "cancer"]
    cancer_patterns = [
        r'\b(non[- ]?small[- ]?cell lung cancer)\b',
        r'\b(small[- ]?cell lung cancer)\b',
        r'\b(triple[- ]?negative breast cancer)\b',
        r'\b(her2[- ]?positive breast cancer)\b',
        r'\b(metastatic breast cancer)\b',
        r'\b(advanced breast cancer)\b',
        r'\b(metastatic lung cancer)\b',
        r'\b(advanced lung cancer)\b',
        r'\b(metastatic (?:colorectal|colon|rectal) cancer)\b',
        r'\b(hepatocellular carcinoma)\b',
        r'\b(renal cell carcinoma)\b',
        r'\b(squamous cell carcinoma)\b',
        r'\b(acute myeloid leukemia)\b',
        r'\b(acute lymphoblastic leukemia)\b',
        r'\b(chronic lymphocytic leukemia)\b',
        r'\b(chronic myeloid leukemia)\b',
        r'\b(hodgkin[\'s]? lymphoma)\b',
        r'\b(non[- ]?hodgkin[\'s]? lymphoma)\b',
        r'\b(diffuse large b[- ]?cell lymphoma)\b',
        r'\b(multiple myeloma)\b',
    ]
    
    # Generic cancer patterns (2-3 word combinations)
    generic_cancer = [
        r'\b((?:lung|breast|colon|liver|pancreatic|kidney|ovarian|prostate|gastric|brain)\s+cancer)\b',
        r'\b((?:lung|breast|colon|liver|pancreatic|kidney|ovarian|prostate|gastric)\s+carcinoma)\b',
        r'\b(metastatic\s+(?:disease|cancer|carcinoma))\b',
        r'\b(advanced\s+(?:cancer|carcinoma|disease))\b',
        r'\b(recurrent\s+(?:cancer|carcinoma|disease))\b',
    ]
    
    # Disease states
    disease_states = [
        r'\b(stage\s+(?:I{1,3}V?|[1234])\s+(?:cancer|carcinoma|disease))\b',
        r'\b(locally advanced (?:cancer|carcinoma|disease))\b',
        r'\b(unresectable (?:cancer|carcinoma|disease|tumor))\b',
    ]
    
    # ========== DRUG PATTERNS ==========
    
    # Specific drug names (immunotherapy, targeted therapy)
    drug_names = [
        r'\b(pembrolizumab|keytruda)\b',
        r'\b(nivolumab|opdivo)\b',
        r'\b(atezolizumab|tecentriq)\b',
        r'\b(durvalumab|imfinzi)\b',
        r'\b(ipilimumab|yervoy)\b',
        r'\b(trastuzumab|herceptin)\b',
        r'\b(bevacizumab|avastin)\b',
        r'\b(rituximab|rituxan)\b',
        r'\b(cetuximab|erbitux)\b',
        r'\b(panitumumab|vectibix)\b',
    ]
    
    # Chemotherapy drugs
    chemo_drugs = [
        r'\b(cisplatin)\b',
        r'\b(carboplatin)\b',
        r'\b(oxaliplatin)\b',
        r'\b(paclitaxel|taxol)\b',
        r'\b(docetaxel|taxotere)\b',
        r'\b(gemcitabine|gemzar)\b',
        r'\b(pemetrexed|alimta)\b',
        r'\b(5[- ]?fluorouracil|5[- ]?fu)\b',
        r'\b(capecitabine|xeloda)\b',
        r'\b(doxorubicin|adriamycin)\b',
    ]
    
    # Treatment categories (2+ words)
    treatment_categories = [
        r'\b(platinum[- ]?based chemotherapy)\b',
        r'\b(prior (?:systemic |immune |chemo)?therapy)\b',
        r'\b(checkpoint inhibitor)\b',
        r'\b(pd[- ]?1 inhibitor)\b',
        r'\b(pd[- ]?l1 inhibitor)\b',
        r'\b(ctla[- ]?4 inhibitor)\b',
        r'\b(targeted therapy)\b',
        r'\b(systemic therapy)\b',
        r'\b(anti[- ]?cancer therapy)\b',
    ]
    
    # ========== BIOMARKER PATTERNS (EXPANDED V2) ==========
    
    biomarker_patterns = [
        # HER2 patterns - expanded
        r'\b(her2[- ]?positive|her2\+|her2 positive)\b',
        r'\b(her2[- ]?negative|her2\-|her2 negative)\b',
        r'\b(her2 amplification)\b',
        r'\b(her2 overexpression)\b',
        r'\b(her2 status)\b',
        r'\b(her2[- ]?targeting)\b',
        
        # ER/PR patterns
        r'\b(er[- ]?positive|er\+|estrogen receptor positive)\b',
        r'\b(er[- ]?negative|er\-|estrogen receptor negative)\b',
        r'\b(pr[- ]?positive|pr\+|progesterone receptor positive)\b',
        r'\b(pr[- ]?negative|pr\-|progesterone receptor negative)\b',
        
        # PD-L1 patterns - much more comprehensive
        r'\b(pd[- ]?l1 (?:positive|expression|status))\b',
        r'\b(pd[- ]?l1 (?:sp[- ]?142|sp[- ]?263|22c3|28[- ]?8))\b',  # Assay names
        r'\b(pd[- ]?l1 ic score)\b',
        r'\b(pd[- ]?l1 tps)\b',
        r'\b(pd[- ]?l1 tumor proportion score)\b',
        r'\b(pd[- ]?l1 (?:≥|>=)\s*\d+%)\b',
        r'\b(pdl1[- ]?positive)\b',
        
        # EGFR patterns - very comprehensive
        r'\b(egfr[- ]?positive)\b',
        r'\b(egfr sensitizing mutations?)\b',
        r'\b(egfr activating mutations?)\b',
        r'\b(egfr mutations?)\b',
        r'\b(egfr exon (?:18|19|20|21))\b',
        r'\b(egfr (?:exon 19 deletion|del19))\b',
        r'\b(egfr l858r)\b',
        r'\b(egfr t790m)\b',
        r'\b(egfr wild[- ]?type)\b',
        r'\b(egfr[- ]?mutant)\b',
        
        # ALK patterns
        r'\b(alk[- ]?positive)\b',
        r'\b(alk fusion)\b',
        r'\b(alk rearrangement)\b',
        r'\b(alk translocation)\b',
        
        # ROS1 patterns
        r'\b(ros1[- ]?positive)\b',
        r'\b(ros1 fusion)\b',
        r'\b(ros1 rearrangement)\b',
        
        # NTRK patterns (found in your data!)
        r'\b(ntrk fusion)\b',
        r'\b(ntrk rearrangement)\b',
        r'\b(ntrk[- ]?positive)\b',
        
        # KRAS patterns
        r'\b(kras mutations?)\b',
        r'\b(kras[- ]?mutant)\b',
        r'\b(kras wild[- ]?type)\b',
        
        # BRAF patterns - very specific
        r'\b(braf v600e)\b',
        r'\b(braf v600k)\b',
        r'\b(braf v600)\b',
        r'\b(braf mutations?)\b',
        r'\b(braf[- ]?mutant)\b',
        r'\b(braf wild[- ]?type)\b',
        
        # BRCA patterns
        r'\b(brca1?\/?\b2?\s+mutations?)\b',
        r'\b(brca[- ]?mutant)\b',
        
        # MSI/MMR patterns
        r'\b(msi[- ]?high)\b',
        r'\b(msi[- ]?h)\b',
        r'\b(microsatellite instability[- ]?high)\b',
        r'\b(mmr[- ]?deficient)\b',
        r'\b(mismatch repair[- ]?deficient)\b',
        
        # TMB patterns
        r'\b(tmb[- ]?high)\b',
        r'\b(tumor mutational burden[- ]?high)\b',
        r'\b(high tumor mutational burden)\b',
        
        # Generic useful patterns
        r'\b(targetable (?:genomic aberration|alteration|mutation)s?)\b',
        r'\b(actionable mutations?)\b',
        r'\b(driver mutations?)\b',
        r'\b(sensitizing mutations?)\b',
        r'\b(activating mutations?)\b',
    ]
    
    # ========== LAB TEST PATTERNS ==========
    
    # Lab tests WITH numerical thresholds (IMPROVED!)
    lab_tests = [
        r'\b(ecog performance status\s+[0-5]?(?:[- ]?[0-5])?)\b',
        r'\b(karnofsky performance status\s*≥?\s*\d+)\b',
        r'\b(absolute neutrophil count\s*≥?\s*\d+[\d,]*(?:/μl|/mm3)?)\b',
        r'\b(anc\s*≥?\s*\d+[\d,]*(?:/μl|/mm3)?)\b',
        r'\b(platelet count\s*≥?\s*\d+[\d,]*(?:/μl|/mm3)?)\b',
        r'\b(white blood cell count\s*≥?\s*\d+[\d,]*(?:/μl|/mm3)?)\b',
        r'\b(wbc\s*≥?\s*\d+[\d,]*)\b',
        r'\b(hemoglobin\s*≥?\s*\d+(?:\.\d+)?\s*(?:g/dl)?)\b',
        r'\b(creatinine clearance\s*≥?\s*\d+\s*(?:ml/min)?)\b',
        r'\b(estimated glomerular filtration rate\s*≥?\s*\d+)\b',
        r'\b(egfr\s*≥?\s*\d+\s*(?:ml/min)?)\b',
        r'\b(serum creatinine\s*≤?\s*\d+(?:\.\d+)?\s*(?:mg/dl)?)\b',
        r'\b(total bilirubin\s*≤?\s*\d+(?:\.\d+)?\s*(?:times\s+)?(?:uln|upper limit of normal)?)\b',
        r'\b(ast\s*≤?\s*\d+(?:\.\d+)?\s*(?:times\s+)?(?:uln|upper limit of normal)?)\b',
        r'\b(alt\s*≤?\s*\d+(?:\.\d+)?\s*(?:times\s+)?(?:uln|upper limit of normal)?)\b',
        r'\b(alkaline phosphatase\s*≤?\s*\d+(?:\.\d+)?\s*(?:times\s+)?(?:uln)?)\b',
        r'\b(serum albumin\s*≥?\s*\d+(?:\.\d+)?\s*(?:g/dl)?)\b',
        r'\b(international normalized ratio\s*≤?\s*\d+(?:\.\d+)?)\b',
        r'\b(inr\s*≤?\s*\d+(?:\.\d+)?)\b',
        r'\b(left ventricular ejection fraction\s*≥?\s*\d+%?)\b',
        r'\b(lvef\s*≥?\s*\d+%?)\b',
        # Also keep versions without numbers
        r'\b(absolute neutrophil count)\b',
        r'\b(creatinine clearance)\b',
        r'\b(total bilirubin)\b',
        r'\b(serum creatinine)\b',
        r'\b(left ventricular ejection fraction)\b',
        r'\b(platelet count)\b',
    ]
    
    # ========== PROCEDURE PATTERNS ==========
    
    procedures = [
        r'\b(tumor biopsy)\b',
        r'\b(core needle biopsy)\b',
        r'\b(surgical resection)\b',
        r'\b(definitive surgery)\b',
        r'\b(stem cell transplant(?:ation)?)\b',
        r'\b(bone marrow transplant(?:ation)?)\b',
        r'\b(radiation therapy)\b',
        r'\b(definitive radiotherapy)\b',
    ]
    
    # ========== EXTRACT ENTITIES ==========
    
    # Extract diseases
    for pattern in cancer_patterns + generic_cancer + disease_states:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entity = match.group(1).strip().lower()
            entities['DISEASE'].add(entity)
    
    # Extract drugs
    for pattern in drug_names + chemo_drugs + treatment_categories:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entity = match.group(1).strip().lower()
            # Skip overly generic terms
            if entity not in ['therapy', 'treatment']:
                entities['DRUG'].add(entity)
    
    # Extract biomarkers
    for pattern in biomarker_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entity = match.group(1).strip().lower()
            entities['BIOMARKER'].add(entity)
    
    # Extract lab tests
    for pattern in lab_tests:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entity = match.group(1).strip().lower()
            entities['LAB_TEST'].add(entity)
    
    # Extract procedures
    for pattern in procedures:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entity = match.group(1).strip().lower()
            entities['PROCEDURE'].add(entity)
    
    # Convert sets to sorted lists and filter out empty categories
    result = {}
    for entity_type, entity_set in entities.items():
        if entity_set:
            result[entity_type] = sorted(list(entity_set))
    
    return result

def summarize_criteria(text):
    """Summarize eligibility criteria using BART"""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    
    text = str(text)
    if len(text) < 100:
        return text
    
    if summarizer is None:
        return text[:100] + "..."
    
    try:
        summary = summarizer(
            text[:1000],
            max_length=60,
            min_length=20,
            do_sample=False
        )
        return summary[0]['summary_text']
    except Exception as e:
        print(f"      Summarization error: {e}")
        return text[:100] + "..."

def process_all_trials(input_csv='data/processed/processed_trials.csv',
                       output_json='results/extraction_results.json'):
    """
    Process all trials and extract entities from eligibility criteria.
    
    Reads preprocessed trial data, applies entity extraction to both
    inclusion and exclusion criteria, and compiles results into structured format.
    
    Returns:
        list: List of dictionaries, one per trial, containing:
            - Trial metadata (NCT ID, title, condition)
            - Extracted entities from inclusion criteria
            - Extracted entities from exclusion criteria
            
    Processing Flow:
        1. Load preprocessed trials from CSV
        2. For each trial:
           - Extract entities from inclusion criteria
           - Extract entities from exclusion criteria
           - Combine metadata with entity results
        3. Save results to JSON for downstream analysis
        
    Design Note:
        Extracts from both inclusion AND exclusion criteria because biomarkers
        often appear in exclusion (e.g., "EGFR mutation must be excluded").
        This ensures complete biomarker capture.
    """
    if not os.path.exists(input_csv):
        print(f"ERROR: {input_csv} not found!")
        return None
    
    df = pd.read_csv(input_csv)
    print(f"Processing {len(df)} trials...\n")
    
    results = []
    
    for idx, row in df.iterrows():
        nct_id = row['nct_id']
        print(f"[{idx+1:2d}/{len(df)}] {nct_id}", end='')
        
        # Extract entities
        inc_entities = extract_medical_entities_regex(
            row.get('inclusion_criteria', '')
        )
        exc_entities = extract_medical_entities_regex(
            row.get('exclusion_criteria', '')
        )
        
        # Summarize
        inc_summary = summarize_criteria(row.get('inclusion_criteria', ''))
        exc_summary = summarize_criteria(row.get('exclusion_criteria', ''))
        
        results.append({
            'nct_id': nct_id,
            'title': row.get('title', ''),
            'condition': row.get('condition', ''),
            'phase': row.get('phase', ''),
            'inclusion_entities': inc_entities,
            'exclusion_entities': exc_entities,
            'inclusion_summary': inc_summary,
            'exclusion_summary': exc_summary,
            'num_inclusion_entities': sum(len(v) for v in inc_entities.values()),
            'num_exclusion_entities': sum(len(v) for v in exc_entities.values())
        })
        
        total = results[-1]['num_inclusion_entities'] + results[-1]['num_exclusion_entities']
        print(f" → {total} entities ✓")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ Extraction complete!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_json}")
    
    return results

def visualize_entities(results):
    """Create visualizations"""
    print("\nGenerating visualizations...")
    
    # Collect all entities
    all_entities = []
    for result in results:
        for entity_list in result['inclusion_entities'].values():
            all_entities.extend(entity_list)
    
    if not all_entities:
        print("No entities found")
        return
    
    # Top entities
    entity_counts = Counter(all_entities)
    top_20 = entity_counts.most_common(20)
    
    if top_20:
        entities, counts = zip(*top_20)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(entities)), counts, color='steelblue')
        plt.yticks(range(len(entities)), entities)
        plt.xlabel('Frequency', fontsize=12)
        plt.title('Top 20 Medical Entities', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        os.makedirs('results/visualizations', exist_ok=True)
        plt.savefig('results/visualizations/entity_frequency.png', dpi=150)
        print(f"✓ Saved: results/visualizations/entity_frequency.png")
        plt.close()
    
    # Entity types
    type_counts = Counter()
    for result in results:
        for entity_type in result['inclusion_entities'].keys():
            type_counts[entity_type] += len(result['inclusion_entities'][entity_type])
    
    if type_counts:
        plt.figure(figsize=(10, 6))
        types = list(type_counts.keys())
        counts = [type_counts[t] for t in types]
        
        colors = ['steelblue', 'coral', 'mediumseagreen', 'gold'][:len(types)]
        plt.bar(types, counts, color=colors)
        plt.xlabel('Entity Type', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Entity Type Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig('results/visualizations/entity_types.png', dpi=150)
        print(f"✓ Saved: results/visualizations/entity_types.png")
        plt.close()

def print_summary_stats(results):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    
    total_trials = len(results)
    total_inc = sum(r['num_inclusion_entities'] for r in results)
    total_exc = sum(r['num_exclusion_entities'] for r in results)
    
    print(f"\nTrials processed: {total_trials}")
    print(f"Total inclusion entities: {total_inc}")
    print(f"Total exclusion entities: {total_exc}")
    print(f"Average entities per trial: {(total_inc + total_exc) / total_trials:.1f}")
    
    # Sample
    print("\n" + "="*70)
    print("Sample Extraction")
    print("="*70)
    
    if results:
        sample = results[0]
        print(f"\nTrial: {sample['nct_id']}")
        print(f"Title: {sample['title']}")
        print(f"\nInclusion Summary:")
        print(f"  {sample['inclusion_summary']}")
        print(f"\nExtracted Entities:")
        for entity_type, entities in sample['inclusion_entities'].items():
            print(f"  {entity_type}: {', '.join(entities[:5])}")

def visualize_biomarker_landscape(results):
    """Show which biomarkers are most commonly required"""
    
    biomarkers = []
    for result in results:
        if 'BIOMARKER' in result['inclusion_entities']:
            biomarkers.extend(result['inclusion_entities']['BIOMARKER'])
    
    if not biomarkers:
        return
    
    from collections import Counter
    biomarker_counts = Counter(biomarkers)
    
    plt.figure(figsize=(12, 8))
    markers, counts = zip(*biomarker_counts.most_common(15))
    
    # Color code by category
    colors = []
    for marker in markers:
        if 'pd-l1' in marker or 'pd l1' in marker:
            colors.append('steelblue')  # Immunotherapy
        elif 'egfr' in marker or 'alk' in marker or 'braf' in marker:
            colors.append('coral')  # Targeted therapy
        elif 'her2' in marker:
            colors.append('mediumseagreen')  # HER2-targeted
        elif 'msi' in marker or 'mmr' in marker:
            colors.append('gold')  # Immunotherapy
        else:
            colors.append('lightgray')
    
    plt.barh(range(len(markers)), counts, color=colors)
    plt.yticks(range(len(markers)), markers)
    plt.xlabel('Number of Trials', fontsize=12)
    plt.title('Biomarker Requirements Across Clinical Trials', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='PD-L1 (Immunotherapy)'),
        Patch(facecolor='coral', label='Driver Mutations (Targeted Therapy)'),
        Patch(facecolor='mediumseagreen', label='HER2 (Targeted Therapy)'),
        Patch(facecolor='gold', label='MSI/MMR (Immunotherapy)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('results/visualizations/biomarker_landscape.png', dpi=150)
    print("✓ Biomarker landscape saved")


def visualize_treatment_classes(results):
    """Categorize drugs by treatment class"""
    
    immunotherapy = []
    targeted = []
    chemo = []
    
    for result in results:
        if 'DRUG' in result['inclusion_entities']:
            for drug in result['inclusion_entities']['DRUG']:
                if any(x in drug for x in ['pembrolizumab', 'nivolumab', 'checkpoint', 'pd-1', 'pd-l1']):
                    immunotherapy.append(drug)
                elif any(x in drug for x in ['targeted', 'inhibitor', 'her2']):
                    targeted.append(drug)
                elif any(x in drug for x in ['platinum', 'chemo', 'cisplatin', 'carboplatin']):
                    chemo.append(drug)
    
    categories = ['Immunotherapy', 'Targeted Therapy', 'Chemotherapy']
    counts = [len(set(immunotherapy)), len(set(targeted)), len(set(chemo))]
    
    plt.figure(figsize=(10, 6))
    colors = ['steelblue', 'coral', 'mediumseagreen']
    plt.bar(categories, counts, color=colors, alpha=0.8)
    plt.ylabel('Number of Unique Drug/Treatment Mentions', fontsize=12)
    plt.title('Treatment Modality Distribution', fontsize=14, fontweight='bold')
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        plt.text(i, count + 0.5, str(count), ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/visualizations/treatment_classes.png', dpi=150)
    print("✓ Treatment classes saved")


def visualize_lab_thresholds(results):
    """Show common lab test thresholds - KEY INSIGHT!"""
    
    # Common lab tests
    anc_values = []
    creatinine_values = []
    bilirubin_values = []
    
    for result in results:
        criteria_text = result.get('inclusion_criteria', '') + ' ' + result.get('exclusion_criteria', '')
        
        # Extract ANC thresholds
        anc_match = re.search(r'neutrophil count\s*≥?\s*(\d+[\d,]*)', criteria_text, re.IGNORECASE)
        if anc_match:
            value = anc_match.group(1).replace(',', '')
            anc_values.append(int(value))
        
        # Extract creatinine clearance
        creat_match = re.search(r'creatinine clearance\s*≥?\s*(\d+)', criteria_text, re.IGNORECASE)
        if creat_match:
            creatinine_values.append(int(creat_match.group(1)))
    
    # Create threshold distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ANC thresholds
    if anc_values:
        from collections import Counter
        anc_counts = Counter(anc_values)
        axes[0].bar(anc_counts.keys(), anc_counts.values(), color='steelblue', alpha=0.8)
        axes[0].set_xlabel('ANC Threshold (cells/μL)', fontsize=11)
        axes[0].set_ylabel('Number of Trials', fontsize=11)
        axes[0].set_title('Absolute Neutrophil Count Requirements', fontsize=12, fontweight='bold')
        axes[0].axvline(x=1500, color='red', linestyle='--', alpha=0.5, label='Common: ≥1,500')
        axes[0].legend()
    
    # Creatinine clearance thresholds
    if creatinine_values:
        creat_counts = Counter(creatinine_values)
        axes[1].bar(creat_counts.keys(), creat_counts.values(), color='coral', alpha=0.8)
        axes[1].set_xlabel('Creatinine Clearance (mL/min)', fontsize=11)
        axes[1].set_ylabel('Number of Trials', fontsize=11)
        axes[1].set_title('Creatinine Clearance Requirements', fontsize=12, fontweight='bold')
        axes[1].axvline(x=60, color='red', linestyle='--', alpha=0.5, label='Common: ≥60')
        axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('results/visualizations/lab_thresholds.png', dpi=150)
    print("✓ Lab threshold benchmarks saved")


if __name__ == "__main__":
    results = process_all_trials()
    
    if results:
        print_summary_stats(results)
        visualize_entities(results)
        visualize_biomarker_landscape(results)
        visualize_treatment_classes(results)
        visualize_lab_thresholds(results)

        print("\n" + "="*70)
        print("✓ All done!")
        print("="*70)
        print("\nOutputs:")
        print("  - results/extraction_results.json")
        print("  - results/visualizations/entity_frequency.png")
        print("  - results/visualizations/entity_types.png")