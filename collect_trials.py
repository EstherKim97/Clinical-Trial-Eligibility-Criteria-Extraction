"""
Clinical Trial Data Collection from ClinicalTrials.gov

This script collects clinical trial data using the ClinicalTrials.gov API v2.
It searches for trials based on disease + treatment combinations and extracts
eligibility criteria text for downstream NLP analysis.

Key Design Decisions:
- Two-step API process: search for trial IDs, then fetch full details
- Deduplication to handle trials appearing in multiple searches
- Rate limiting (0.5s delay) to avoid overwhelming the API
- Defensive coding with .get() to handle missing JSON fields

Output: CSV file with trial metadata and eligibility criteria text
"""


import requests
import pandas as pd
import time
import os


def search_trials_v2(condition, max_results=20):
    """
    Search for clinical trials matching a specific condition.
    
    Uses ClinicalTrials.gov API v2 to find trial IDs (NCT numbers) matching
    the search query. This is step 1 of a two-step process.
    
    Args:
        condition (str): Search query (e.g., "immunotherapy lung cancer")
        max_results (int): Maximum number of trial IDs to return (default: 15)
        
    Returns:
        list: NCT IDs of matching trials, or empty list if search fails
        
    Design Note:
        API v2 separates search (get IDs) from fetch (get details).
        This function only returns IDs; use fetch_trial_details() to get full data.
    """

    base_url = "https://clinicaltrials.gov/api/v2/studies"
    
    # Correct parameter format for v2 API
    params = {
        'query.cond': condition,
        'pageSize': max_results,
        'format': 'json'
    }
    
    try:
        print(f"  Searching for '{condition}'...")
        response = requests.get(base_url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Debug: Print what we got
            print(f"  API Response keys: {list(data.keys())}")
            
            # Extract NCT IDs from response
            if 'studies' in data:
                nct_ids = []
                for study in data['studies']:
                    try:
                        nct_id = study['protocolSection']['identificationModule']['nctId']
                        nct_ids.append(nct_id)
                    except KeyError:
                        continue
                
                print(f"  Found {len(nct_ids)} trials")
                return nct_ids
            else:
                print(f"  Warning: No 'studies' key in response")
                print(f"  Response: {data}")
                return []
        else:
            print(f"  Error: Status code {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return []
            
    except Exception as e:
        print(f"  Error searching trials: {e}")
        return []

def get_trial_details_v2(nct_id):
    """
    Fetch complete trial details for a given NCT ID.
    
    Retrieves full trial information including eligibility criteria text.
    This is step 2 of the two-step API process.
    
    Args:
        nct_id (str): Clinical trial identifier (e.g., "NCT12345678")
        
    Returns:
        dict: Trial information including title, condition, phase, enrollment,
              inclusion/exclusion criteria text. Returns None if fetch fails.
              
    Defensive Coding:
        Uses .get() with default values throughout to handle missing fields
        gracefully. API responses can be inconsistent, so this prevents crashes.
    """
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Navigate the JSON structure
            protocol = data['protocolSection']
            identification = protocol['identificationModule']
            
            # Get eligibility
            eligibility = protocol.get('eligibilityModule', {})
            
            # Get conditions
            conditions = protocol.get('conditionsModule', {}).get('conditions', [])
            
            # Get design info
            design = protocol.get('designModule', {})
            
            return {
                'nct_id': nct_id,
                'title': identification.get('briefTitle', 'N/A'),
                'condition': ', '.join(conditions) if conditions else 'N/A',
                'phase': ', '.join(design.get('phases', ['N/A'])),
                'enrollment': design.get('enrollmentInfo', {}).get('count', 0),
                'criteria_text': eligibility.get('eligibilityCriteria', ''),
                'min_age': eligibility.get('minimumAge', 'N/A'),
                'max_age': eligibility.get('maximumAge', 'N/A'),
                'sex': eligibility.get('sex', 'ALL'),
                'healthy_volunteers': eligibility.get('healthyVolunteers', False),
                'data_source': 'api',
                'has_pdf': False
            }
        else:
            print(f"    Error fetching {nct_id}: Status {response.status_code}")
            return None
            
    except Exception as e:
        print(f"    Error fetching {nct_id}: {e}")
        return None

def collect_oncology_trials(max_total=40):
    """
    Main collection function
    """
    print("=" * 70)
    print("Collecting Clinical Trials from ClinicalTrials.gov API v2")
    print("=" * 70)
    
    # Strategic keyword searches using Method 3: Disease + Treatment combinations
    # This approach captures diverse biomarkers, drugs, and clinical criteria naturally
    # More effective than searching for diseases alone or biomarkers alone
    conditions = [
        # Category 1: Immunotherapy trials (4 searches)
        'immunotherapy lung cancer',
        'immunotherapy melanoma',
        'checkpoint inhibitor NSCLC',
        'PD-1 therapy breast cancer',
        
        # Category 2: Targeted therapy trials (4 searches)
        'targeted therapy breast cancer',
        'targeted therapy lung cancer',
        'HER2 targeted therapy',
        'EGFR inhibitor lung cancer',
        
        # Category 3: Advanced/metastatic (6 searches)
        'chemotherapy advanced breast cancer',
        'hormone therapy metastatic breast cancer',
        'combination therapy melanoma',
        'metastatic colorectal cancer treatment',
        'advanced renal cell carcinoma therapy',
        'recurrent ovarian cancer treatment',
        
        # Category 4: Disease subtypes (3 searches)
        'triple negative breast cancer treatment',
        'BRAF mutant melanoma',
        'MSI-high colorectal cancer'
    ]

    
    all_trials = []
    seen_ids = set()
    
    for condition in conditions:
        print(f"\nSearching: {condition}")
        
        # Get trial IDs
        nct_ids = search_trials_v2(condition, max_results=15)
        
        if not nct_ids:
            print(f"  No trials found for {condition}")
            continue
        
        # Fetch details for each trial
        for nct_id in nct_ids:
            # Skip if already processed
            if nct_id in seen_ids:
                continue
            
            seen_ids.add(nct_id)
            
            print(f"  Fetching: {nct_id}", end='')
            trial_data = get_trial_details_v2(nct_id)
            
            if trial_data and trial_data['criteria_text']:
                all_trials.append(trial_data)
                print(f" ✓ {trial_data['title'][:40]}...")
            else:
                print(f" ✗ No eligibility criteria")
            
            # Be nice to the API
            time.sleep(0.5)
            
            # Stop if we have enough
            if len(all_trials) >= max_total:
                break
        
        if len(all_trials) >= max_total:
            break
    
    # Save results
    if all_trials:
        df = pd.DataFrame(all_trials)
        
        # Create directory if needed
        os.makedirs('data/raw', exist_ok=True)
        
        # Save to CSV
        df.to_csv('data/raw/api_trials.csv', index=False)
        
        print("\n" + "=" * 70)
        print(f"SUCCESS! Collected {len(all_trials)} trials")
        print("=" * 70)
        print(f"\nDataset Summary:")
        print(f"  Total trials: {len(df)}")
        print(f"  Unique conditions: {df['condition'].nunique()}")
        print(f"  Trials with eligibility criteria: {df['criteria_text'].notna().sum()}")
        print(f"  Saved to: data/raw/api_trials.csv")
        
        return df
    else:
        print("\n" + "=" * 70)
        print("ERROR: No trials collected!")
        print("=" * 70)
        print("\nPossible issues:")
        print("1. Check your internet connection")
        print("2. ClinicalTrials.gov API might be down")
        print("3. API format may have changed")
        print("\nTry visiting: https://clinicaltrials.gov/api/gui")
        return None

if __name__ == "__main__":
    # Run the collection
    df = collect_oncology_trials(max_total=40)
    
    if df is not None:
        print("\n✓ Data collection complete!")
    else:
        print("\n✗ Data collection failed")
        print("Please check the error messages above")