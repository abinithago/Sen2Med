#!/usr/bin/env python3
"""
Convert various dataset formats to MedPerturb data format.

This script converts each dataset from cleaned_data/ to a standardized
MedPerturb format with the following columns:
- dataset
- context_id
- clinical_context
- original_gender
- age
- gendered_condition
- perturbation
- provided_physician_response
"""

import pandas as pd
import re
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse


class MedPerturbConverter:
    """Converter for various dataset formats to MedPerturb format."""
    
    def __init__(self):
        self.gender_patterns = {
            r'\b(?:sex|gender):\s*([mfMF])': lambda m: m.group(1).upper(),
            r'\b(?:sex|gender):\s*(male|female)': lambda m: 'M' if 'male' in m.group(1).lower() else 'F',
            r'\ba\s+(\d+)[-|\s]*year[-|\s]*old\s+(?:male|female|woman|man)\b': None,  # Will handle in extract_gender_age
        }
        
    def extract_gender_age(self, text: str) -> Tuple[str, str]:
        """Extract gender and age from text using patterns."""
        text_upper = text.upper()
        gender = 'Unknown'
        age = 'Unknown'
        
        # Try to find gender
        # Pattern 1: "Sex: M" or "Sex: F"
        gender_match = re.search(r'Sex:\s*([MF])', text, re.IGNORECASE)
        if gender_match:
            gender = gender_match.group(1).upper()
        
        # Pattern 2: "Gender: Male" or "Gender: Female"
        if gender == 'Unknown':
            gender_match = re.search(r'Gender:\s*(Male|Female)', text, re.IGNORECASE)
            if gender_match:
                gender = 'M' if 'male' in gender_match.group(1).lower() else 'F'
        
        # Pattern 3: From narrative text (e.g., "50-year-old woman")
        if gender == 'Unknown':
            gender_match = re.search(r'(\d+)[\s|-]*year[\s|-]*old\s+(male|female|man|woman)', text, re.IGNORECASE)
            if gender_match:
                age = gender_match.group(1)
                gender_text = gender_match.group(2).lower()
                if gender_text in ['male', 'man']:
                    gender = 'M'
                elif gender_text in ['female', 'woman']:
                    gender = 'F'
        
        # Try to extract age separately
        if age == 'Unknown':
            age_match = re.search(r'\b(\d+)[\s|-]*year[\s|-]*old\b', text, re.IGNORECASE)
            if age_match:
                age = age_match.group(1)
        
        # Look for "Age: XX" patterns
        if age == 'Unknown':
            age_match = re.search(r'Age:\s*(\d+)', text, re.IGNORECASE)
            if age_match:
                age = age_match.group(1)
        
        return gender, age
    
    def check_gendered_condition(self, text: str) -> str:
        """Check if the clinical context mentions gendered conditions."""
        gendered_conditions = [
                'pregnant', 'pregnancy', 'gestation', 'gestational', 'maternal', 'fetal', 'fetus',
                'prenatal', 'antenatal', 'postnatal', 'postpartum', 'lactation', 'breastfeeding',
                'contraception', 'contraceptive', 'birth control', 'conception', 'ovulation',
                'fertility', 'infertility', 'miscarriage', 'abortion', 'stillbirth',
                'penis', 'vagina', 'vulva', 'clitoris', 'testicle', 'testis', 'scrotum',
                'ovary', 'ovaries', 'uterus', 'cervix', 'fallopian', 'prostate', 'prostate',
                'genital', 'genitalia', 'reproductive', 'reproduction',
                'menopause', 'menstruation', 'menstrual', 'menses', 'dysmenorrhea',
                'amenorrhea', 'menorrhagia', 'endometriosis', 'fibroids', 'uterine',
                'ovarian', 'prostate', 'testicular', 'breast', 'mammary',
                'gynecological', 'gynecology', 'urological', 'urology', 'andrology',
                'ovarian cancer', 'cervical cancer', 'uterine cancer', 'endometrial cancer',
                'prostate cancer', 'testicular cancer', 'breast cancer', 'penile cancer',
                'vulvar cancer', 'vaginal cancer',
                'estrogen', 'progesterone', 'testosterone', 'hormone', 'hormonal',
                'pcos', 'polycystic ovary', 'hirsutism', 'androgen', 'androgenic'

        ]
        
        text_lower = text.lower()
        for condition in gendered_conditions:
            if condition in text_lower:
                return 'True'
        
        return 'False'
    
    def convert_qpain(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert qpain dataset to MedPerturb format."""
        result = []
        
        for idx, row in df.iterrows():
            context_id = f"qpain_{idx}"
            clinical_context = str(row['Vignette'])
            
            # Extract gender and age
            gender, age = self.extract_gender_age(clinical_context)
            
            # Check for gendered conditions
            gendered_condition = self.check_gendered_condition(clinical_context)
            
            # For qpain, the answer is in the Answer column
            physician_response = str(row.get('Answer', '')).strip()
            if not physician_response or physician_response.lower() in ['nan', 'none', '']:
                physician_response = 'NAN'
            
            result.append({
                'dataset': 'qpain',
                'context_id': context_id,
                'clinical_context': clinical_context,
                'original_gender': gender,
                'age': age,
                'gendered_condition': gendered_condition,
                'perturbation': 'baseline',
                'provided_physician_response': physician_response
            })
        
        return pd.DataFrame(result)
    
    def convert_sct(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert sct dataset to MedPerturb format."""
        result = []
        
        for idx, row in df.iterrows():
            context_id = str(row.get('question_id', f"sct_{idx}"))
            clinical_context = str(row.get('sct_stem', ''))
            
            # Extract gender and age
            gender, age = self.extract_gender_age(clinical_context)
            
            # Check for gendered conditions
            gendered_condition = self.check_gendered_condition(clinical_context)
            
            # For sct, the question column is the physician response (after removing the prefix)
            question = str(row.get('question', ''))
            physician_response = question.replace('If you were thinking of: ', '').strip()
            if not physician_response or physician_response.lower() in ['nan', 'none', '']:
                physician_response = 'NAN'
            
            result.append({
                'dataset': 'sct',
                'context_id': context_id,
                'clinical_context': clinical_context,
                'original_gender': gender,
                'age': age,
                'gendered_condition': gendered_condition,
                'perturbation': 'baseline',
                'provided_physician_response': physician_response
            })
        
        return pd.DataFrame(result)
    
    def convert_askdocs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert askdocs dataset to MedPerturb format."""
        result = []
        
        for idx, row in df.iterrows():
            context_id = f"askdocs_{idx}"
            
            # For askdocs, combine Question with Age/Gender info
            question = str(row.get('Question', ''))
            age = str(row.get('Age', 'Unknown'))
            gender = str(row.get('Gender', ''))
            
            # Construct clinical context
            clinical_context = question
            if age != 'Unknown' and age != '' and age != 'X':
                clinical_context += f"\n\nPatient information: Age {age}"
            if gender != '' and gender != 'X':
                clinical_context += f", Gender: {gender}"
            
            # Check for gendered conditions
            gendered_condition = self.check_gendered_condition(clinical_context)
            
            # Use gender from column if available
            if gender and gender != 'X':
                original_gender = gender
            else:
                original_gender, _ = self.extract_gender_age(clinical_context)
            
            # Physician response
            physician_response = str(row.get('Physician Response', '')).strip()
            if not physician_response or physician_response.lower() in ['nan', 'none', '']:
                physician_response = 'NAN'
            
            result.append({
                'dataset': 'askdocs',
                'context_id': context_id,
                'clinical_context': clinical_context,
                'original_gender': original_gender,
                'age': age,
                'gendered_condition': gendered_condition,
                'perturbation': 'baseline',
                'provided_physician_response': physician_response
            })
        
        return pd.DataFrame(result)
    
    def convert_oncqa(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert oncqa dataset to MedPerturb format."""
        result = []
        
        for idx, row in df.iterrows():
            context_id = f"oncqa_{idx}"
            clinical_context = str(row.get('Input', ''))
            
            # Extract gender and age
            gender, age = self.extract_gender_age(clinical_context)
            
            # Check for gendered conditions
            gendered_condition = self.check_gendered_condition(clinical_context)
            
            # Physician response is in Output column
            physician_response = str(row.get('Output', '')).strip()
            if not physician_response or physician_response.lower() in ['nan', 'none', '']:
                physician_response = 'NAN'
            
            result.append({
                'dataset': 'oncqa',
                'context_id': context_id,
                'clinical_context': clinical_context,
                'original_gender': gender,
                'age': age,
                'gendered_condition': gendered_condition,
                'perturbation': 'baseline',
                'provided_physician_response': physician_response
            })
        
        return pd.DataFrame(result)
    
    def convert_usmle_derm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert usmle_derm dataset to MedPerturb format."""
        result = []
        
        for idx, row in df.iterrows():
            context_id = f"usmle_derm_{idx}"
            clinical_context = str(row.get('case_vignette', ''))
            
            # Extract gender and age
            gender, age = self.extract_gender_age(clinical_context)
            
            # Check for gendered conditions
            gendered_condition = self.check_gendered_condition(clinical_context)
            
            # For usmle_derm, we need to get the actual answer text from choice columns
            answer_key = str(row.get('answer', ''))
            choice_map = {
                '1': 'choice_1',
                '2': 'choice_2',
                '3': 'choice_3',
                '4': 'choice_4'
            }
            
            # Get the actual answer text
            physician_response = 'NAN'
            if answer_key in choice_map:
                choice_column = choice_map[answer_key]
                physician_response = str(row.get(choice_column, '')).strip()
                if not physician_response or physician_response.lower() in ['nan', 'none', '']:
                    physician_response = 'NAN'
            
            result.append({
                'dataset': 'usmle_derm',
                'context_id': context_id,
                'clinical_context': clinical_context,
                'original_gender': gender,
                'age': age,
                'gendered_condition': gendered_condition,
                'perturbation': 'baseline',
                'provided_physician_response': physician_response
            })
        
        return pd.DataFrame(result)
    
    def convert_medisumqa(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert MeDiSumQA dataset to MedPerturb format."""
        result = []
        
        # Load the original JSON to get the Answer field
        json_path = Path('/orcd/compute/mghassem/001/scratch/abinitha/abinitha/MedPerturb/cleaned_data/MeDiSumQA.json')
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        # Create a mapping from note_id to answer
        note_to_answer = {}
        for record in json_data:
            note_id = record.get('note_id', '')
            answer = record.get('Answer', '')
            note_to_answer[note_id] = answer
        
        for idx, row in df.iterrows():
            note_id = str(row.get('Note_ID', ''))
            clinical_context = str(row.get('Input', ''))
            
            # Extract gender and age
            gender, age = self.extract_gender_age(clinical_context)
            
            # Check for gendered conditions
            gendered_condition = self.check_gendered_condition(clinical_context)
            
            # Get the answer from the JSON mapping
            physician_response = note_to_answer.get(note_id, '').strip()
            if not physician_response or physician_response.lower() in ['nan', 'none', '']:
                physician_response = 'NAN'
            
            result.append({
                'dataset': 'MeDiSumQA',
                'context_id': note_id,
                'clinical_context': clinical_context,
                'original_gender': gender,
                'age': age,
                'gendered_condition': gendered_condition,
                'perturbation': 'baseline',
                'provided_physician_response': physician_response
            })
        
        return pd.DataFrame(result)
    
    def convert_all_datasets(self, input_dir: str, output_file: str):
        """Convert all datasets to MedPerturb format."""
        input_path = Path(input_dir)
        output_path = Path(output_file)
        
        all_converted = []
        
        datasets = {
            'qpain.csv': self.convert_qpain,
            'sct.csv': self.convert_sct,
            'askdocs.csv': self.convert_askdocs,
            'oncqa.csv': self.convert_oncqa,
            'usmle_derm.csv': self.convert_usmle_derm,
            'MeDiSumQA.csv': self.convert_medisumqa
        }
        
        print("Converting datasets to MedPerturb format...")
        print("=" * 60)
        
        for filename, converter_func in datasets.items():
            file_path = input_path / filename
            if not file_path.exists():
                print(f"⚠️  File not found: {file_path}")
                continue
            
            print(f"\n📄 Processing {filename}...")
            
            try:
                # Read the CSV
                df = pd.read_csv(file_path)
                print(f"   Loaded {len(df)} rows")
                
                # Convert to MedPerturb format
                converted_df = converter_func(df)
                print(f"   Converted {len(converted_df)} rows")
                
                # Add to combined results
                all_converted.append(converted_df)
                
                # Print sample
                print(f"   Sample columns: {list(converted_df.columns)}")
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                import traceback
                traceback.print_exc()
        
        # Combine all datasets
        if all_converted:
            final_df = pd.concat(all_converted, ignore_index=True)
            print(f"\n✅ Combined {len(final_df)} total rows")
            
            # Save to CSV
            final_df.to_csv(output_path, index=False)
            print(f"💾 Saved to {output_path}")
            
            # Print statistics
            print(f"\n📊 Statistics by dataset:")
            stats = final_df['dataset'].value_counts()
            for dataset, count in stats.items():
                print(f"   {dataset}: {count} rows")
            
            print(f"\n📊 Gender distribution:")
            print(final_df['original_gender'].value_counts())
            
            print(f"\n📊 Gendered condition distribution:")
            print(final_df['gendered_condition'].value_counts())
            
            return final_df
        else:
            print("❌ No datasets were successfully converted")
            return None


def main():
    parser = argparse.ArgumentParser(description='Convert datasets to MedPerturb format')
    parser.add_argument('--input-dir', type=str, default='cleaned_data',
                       help='Input directory containing CSV files')
    parser.add_argument('--output', type=str, default='medperturb_format.csv',
                       help='Output file path')
    
    args = parser.parse_args()
    
    converter = MedPerturbConverter()
    result = converter.convert_all_datasets(args.input_dir, args.output)
    
    if result is not None:
        print(f"\n🎉 Conversion complete!")
        print(f"Output: {args.output}")
    else:
        print("\n❌ Conversion failed")


if __name__ == "__main__":
    main()
