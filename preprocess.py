"""
Clinical Trial Eligibility Criteria Preprocessing

This script cleans and structures raw eligibility criteria text from clinical trials.
It separates inclusion and exclusion criteria, removes artifacts, and standardizes
formatting for downstream entity extraction.

Key Functions:
- Split combined criteria text into inclusion/exclusion sections
- Remove PDF artifacts, page numbers, formatting noise
- Parse bullet points and numbered lists into individual criteria items
- Handle missing/malformed data defensively

Input: Raw trial data from ClinicalTrials.gov API
Output: Cleaned, structured criteria ready for NLP extraction
"""

import pandas as pd
import re
import os

def clean_pdf_artifacts(text):
    """
    Remove common artifacts from text
    - Bullet symbols that render weird
    - Extra whitespace
    - Page numbers
    """
    if not text:
        return ""
    
    # Remove weird bullet symbols
    text = re.sub(r'[■●○▪▫•]', '', text)
    
    # Normalize whitespace (multiple spaces → single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common page markers
    text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
    
    return text.strip()

def clean_criteria(text):
    """
    Split eligibility criteria text into inclusion and exclusion sections.
    
    Clinical trials often provide criteria as combined text with section headers.
    This function uses regex to identify "Inclusion" and "Exclusion" sections
    and separates them for independent analysis.
    
    Args:
        text (str): Raw eligibility criteria text
        
    Returns:
        tuple: (inclusion_text, exclusion_text) as separate strings
        
    Pattern Matching:
        Uses case-insensitive regex to find "Inclusion Criteria:" and
        "Exclusion Criteria:" headers with flexible whitespace and punctuation.
        
    Edge Cases:
        - If only one section present, returns that section + empty string
        - If no clear sections, returns full text as inclusion criteria
    """
    if not text or len(text) < 10:
        return "", ""
    
    # Clean artifacts first
    text = clean_pdf_artifacts(text)
    
    inclusion = ""
    exclusion = ""
    
    # Case-insensitive regex to handle variations:
    # "Inclusion Criteria:", "INCLUSION CRITERIA", "Inclusion:", etc.
    # (?i) = case insensitive flag
    # \s+ = flexible whitespace
    exclusion_pattern = r'(?i)exclusion\s+criteria[:\s]*'
    
    # Check if we have an exclusion section
    if re.search(exclusion_pattern, text):
        # Split at exclusion criteria marker
        parts = re.split(exclusion_pattern, text, maxsplit=1)
        
        inclusion = parts[0] if len(parts) > 0 else ""
        exclusion = parts[1] if len(parts) > 1 else ""
        
        # Remove "Inclusion Criteria:" header from inclusion
        inclusion_pattern = r'(?i)inclusion\s+criteria[:\s]*'
        inclusion = re.sub(inclusion_pattern, '', inclusion)
    else:
        # No exclusion section - treat all as inclusion
        inclusion_pattern = r'(?i)inclusion\s+criteria[:\s]*'
        inclusion = re.sub(inclusion_pattern, '', text)
    
    return inclusion.strip(), exclusion.strip()

def extract_criteria_items(text):
    """
    Parse criteria text into individual criterion items.
    
    Eligibility criteria are often formatted as numbered lists or bullet points.
    This function splits on common delimiters and filters out artifacts.
    
    Args:
        text (str): Criteria section text (inclusion or exclusion)
        
    Returns:
        list: Individual criterion statements as separate strings
        
    Filtering:
        - Removes items shorter than 20 characters (likely artifacts)
        - Strips common PDF noise (##, bullets, page numbers)
        - Removes empty lines and whitespace-only items
        
    Delimiters:
        Splits on: newlines, numbered lists (1., 2.), bullets (•, -), asterisks
    """
    if not text:
        return []
    
    # Split by common delimiters
    # \n = newline
    # \d+\. = numbers like "1.", "2."
    # [-•] = dashes or bullets
    delimiters = r'\n|\d+\.(?=\s)|(?<=\s)-(?=\s)|•'
    items = re.split(delimiters, text)
    
    # Clean each item
    cleaned_items = []
    for item in items:
        item = item.strip()
        
        # Filter out short items (artifacts)
        if len(item) > 20:
            cleaned_items.append(item)
    
    return cleaned_items

def preprocess_trials(input_csv='data/raw/api_trials.csv', 
                      output_csv='data/processed/processed_trials.csv'):
    """
    Main preprocessing function
    
    Reads raw trial data, cleans criteria, saves processed data
    """
    print("=" * 70)
    print("Step 3: Preprocessing Eligibility Criteria")
    print("=" * 70)
    
    # Load data
    if not os.path.exists(input_csv):
        print(f"ERROR: Input file not found: {input_csv}")
        print("Please run collect_api.py first!")
        return None
    
    df = pd.read_csv(input_csv)
    print(f"\nLoaded {len(df)} trials from {input_csv}")
    
    # Process each trial
    processed_data = []
    
    for idx, row in df.iterrows():
        # Split and clean criteria
        inclusion, exclusion = clean_criteria(row['criteria_text'])
        
        # Parse into individual items
        inclusion_items = extract_criteria_items(inclusion)
        exclusion_items = extract_criteria_items(exclusion)
        
        # Store results
        processed_data.append({
            **row.to_dict(),  # Keep all original columns
            'inclusion_criteria': inclusion,
            'exclusion_criteria': exclusion,
            'inclusion_items': '; '.join(inclusion_items),  # Store as string for CSV
            'exclusion_items': '; '.join(exclusion_items),
            'num_inclusion': len(inclusion_items),
            'num_exclusion': len(exclusion_items)
        })
        
        # Progress indicator
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df)} trials")
    
    # Create processed dataframe
    processed_df = pd.DataFrame(processed_data)
    
    # Save to CSV
    os.makedirs('data/processed', exist_ok=True)
    processed_df.to_csv(output_csv, index=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Preprocessing Complete!")
    print("=" * 70)
    print(f"\nSummary Statistics:")
    print(f"  Total trials processed: {len(processed_df)}")
    print(f"  Average inclusion criteria: {processed_df['num_inclusion'].mean():.1f}")
    print(f"  Average exclusion criteria: {processed_df['num_exclusion'].mean():.1f}")
    print(f"  Trials with both inc/exc: {(processed_df['num_exclusion'] > 0).sum()}")
    print(f"\nSaved to: {output_csv}")
    
    return processed_df

if __name__ == "__main__":
    # Run preprocessing
    df = preprocess_trials()
    
    if df is not None:
        print("\n✓ Preprocessing complete!")
        print("\nSample processed criteria:")
        
        # Show first trial as example
        sample = df.iloc[0]
        print(f"\nTrial: {sample['nct_id']}")
        print(f"Title: {sample['title'][:60]}...")
        print(f"\nInclusion ({sample['num_inclusion']} items):")
        print(sample['inclusion_criteria'][:200] + "...")
        print(f"\nExclusion ({sample['num_exclusion']} items):")
        print(sample['exclusion_criteria'][:200] + "...")
        
        print("\nNext step: Run extract_entities.py for NLP extraction")