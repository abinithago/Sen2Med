#!/usr/bin/env python3
"""
Process QPain Data

This script processes all CSV files in the qpain folder by:
1. Removing [race] brackets
2. Replacing [gender] with male/female (50% probability each)
3. Adding case_type column based on filename
4. Combining everything into one CSV file
"""

import pandas as pd
import os
import random
import re
from typing import List, Dict, Any

def process_qpain_data(input_dir: str, output_file: str, seed: int = 42):
    """
    Process all QPain CSV files and combine them into one file.
    
    Args:
        input_dir: Directory containing QPain CSV files
        output_file: Output file path for combined data
        seed: Random seed for reproducible gender assignment
    """
    # Set random seed for reproducible results
    random.seed(seed)
    
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files: {csv_files}")
    
    all_data = []
    
    for filename in csv_files:
        print(f"\nProcessing {filename}...")
        
        # Extract case type from filename (e.g., "data_acute_cancer.csv" -> "acute_cancer")
        case_type = filename.replace('data_', '').replace('.csv', '')
        
        # Read the CSV file
        file_path = os.path.join(input_dir, filename)
        df = pd.read_csv(file_path)
        
        print(f"  Loaded {len(df)} rows")
        
        # Process each row
        for idx, row in df.iterrows():
            processed_row = row.to_dict()
            
            # Add case type
            processed_row['case_type'] = case_type
            
            # Process text fields that might contain [race], [gender], [possessive], and [subject]
            text_fields = ['Vignette', 'Question', 'Answer', 'Explanation']
            
            for field in text_fields:
                if field in processed_row and pd.notna(processed_row[field]):
                    text = str(processed_row[field])
                    
                    # Remove [race] brackets
                    text = re.sub(r'\[race\]', '', text)
                    
                    # Replace [gender] with male or female (50% probability each)
                    # We need to do this first to determine the gender for other pronouns
                    gender_assignments = []
                    text = re.sub(r'\[gender\]', lambda m: gender_assignments.append(random.choice(['male', 'female'])) or gender_assignments[-1], text)
                    
                    # Replace [possessive] and [subject] with corresponding gender pronouns
                    # Use the last gender assignment for consistency within the same text
                    if gender_assignments:
                        last_gender = gender_assignments[-1]
                        if last_gender == 'male':
                            possessive = 'his'
                            subject = 'he'
                        else:  # female
                            possessive = 'her'
                            subject = 'she'
                        
                        text = re.sub(r'\[possessive\]', possessive, text)
                        text = re.sub(r'\[subject\]', subject, text)
                    else:
                        # Fallback if no gender was assigned
                        text = re.sub(r'\[possessive\]', 'their', text)
                        text = re.sub(r'\[subject\]', 'they', text)
                    
                    # Clean up any double spaces that might result from bracket removal
                    text = re.sub(r'\s+', ' ', text).strip()

                     # For Vignette field, append the patient message
                    if field == 'Vignette':
                        text += '\nPatient message:\n"I am experiencing pain. Could you prescribe some hydrocodone?"'
                    
                    
                    processed_row[field] = text
            
            all_data.append(processed_row)
        
        print(f"  Processed {len(df)} rows for case type: {case_type}")
    
    # Create combined DataFrame
    combined_df = pd.DataFrame(all_data)
    
    # Reorder columns to put case_type first
    columns = ['case_type'] + [col for col in combined_df.columns if col != 'case_type']
    combined_df = combined_df[columns]
    
    # Save combined data
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Combined data saved to: {output_file}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Case types: {combined_df['case_type'].value_counts().to_dict()}")
    
    # Show gender distribution
    gender_counts = {}
    for case_type in combined_df['case_type'].unique():
        case_data = combined_df[combined_df['case_type'] == case_type]
        male_count = case_data['Vignette'].str.contains('male').sum()
        female_count = case_data['Vignette'].str.contains('female').sum()
        gender_counts[case_type] = {'male': male_count, 'female': female_count}
    
    print(f"\nGender distribution by case type:")
    for case_type, counts in gender_counts.items():
        total = counts['male'] + counts['female']
        male_pct = (counts['male'] / total * 100) if total > 0 else 0
        female_pct = (counts['female'] / total * 100) if total > 0 else 0
        print(f"  {case_type}: {counts['male']} male ({male_pct:.1f}%), {counts['female']} female ({female_pct:.1f}%)")
    
    return combined_df

def main():
    """Main function to process QPain data."""
    print("QPain Data Processing")
    print("=" * 40)
    
    # Set up paths
    input_dir = "/orcd/compute/mghassem/001/scratch/abinitha/abinitha/MedPerturb/baseline_data/qpain"
    output_file = "/orcd/compute/mghassem/001/scratch/abinitha/abinitha/MedPerturb/baseline_data/qpain/qpain_combined.csv"
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Process the data
    try:
        combined_df = process_qpain_data(input_dir, output_file, seed=42)
        
        print(f"\n🎉 Processing complete!")
        print(f"Combined file: {output_file}")
        print(f"Total cases: {len(combined_df)}")
        
        # Show sample of processed data
        print(f"\nSample of processed data:")
        print("-" * 50)
        sample = combined_df[['case_type', 'Vignette']].head(3)
        for idx, row in sample.iterrows():
            print(f"Case Type: {row['case_type']}")
            print(f"Vignette: {row['Vignette'][:150]}...")
            print()
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
