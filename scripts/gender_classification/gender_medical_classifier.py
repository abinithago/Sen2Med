#!/usr/bin/env python3
"""
Gender-Relevant Medical Case Classifier

This script identifies medical cases that mention:
- Gender-relevant diseases (like ovarian cancer)
- Pregnancy mentions
- Genitalia references
- Gendered conditions (like menopause or menstruation)

Uses HuggingFace Qwen/Qwen3-32B model for efficient classification.
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from typing import List, Dict, Tuple, Any
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class GenderMedicalClassifier:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-32B-Instruct"):
        """
        Initialize the classifier with the specified model.
        
        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        # Define gender-relevant medical terms and patterns
        self.gender_terms = {
            'pregnancy': [
                'pregnant', 'pregnancy', 'gestation', 'gestational', 'maternal', 'fetal', 'fetus',
                'prenatal', 'antenatal', 'postnatal', 'postpartum', 'lactation', 'breastfeeding',
                'contraception', 'contraceptive', 'birth control', 'conception', 'ovulation',
                'fertility', 'infertility', 'miscarriage', 'abortion', 'stillbirth'
            ],
            'genitalia': [
                'penis', 'vagina', 'vulva', 'clitoris', 'testicle', 'testis', 'scrotum',
                'ovary', 'ovaries', 'uterus', 'cervix', 'fallopian', 'prostate', 'prostate',
                'genital', 'genitalia', 'reproductive', 'reproduction'
            ],
            'gendered_conditions': [
                'menopause', 'menstruation', 'menstrual', 'period', 'menses', 'dysmenorrhea',
                'amenorrhea', 'menorrhagia', 'endometriosis', 'fibroids', 'uterine',
                'ovarian', 'cervical', 'prostate', 'testicular', 'breast', 'mammary',
                'gynecological', 'gynecology', 'urological', 'urology', 'andrology'
            ],
            'gender_specific_cancers': [
                'ovarian cancer', 'cervical cancer', 'uterine cancer', 'endometrial cancer',
                'prostate cancer', 'testicular cancer', 'breast cancer', 'penile cancer',
                'vulvar cancer', 'vaginal cancer'
            ],
            'hormonal_conditions': [
                'estrogen', 'progesterone', 'testosterone', 'hormone', 'hormonal',
                'pcos', 'polycystic ovary', 'hirsutism', 'androgen', 'androgenic'
            ]
        }
        
        # Compile regex patterns for efficient matching
        self.patterns = {}
        for category, terms in self.gender_terms.items():
            # Create case-insensitive pattern
            pattern = r'\b(?:' + '|'.join(re.escape(term) for term in terms) + r')\b'
            self.patterns[category] = re.compile(pattern, re.IGNORECASE)
    
    def detect_patient_gender(self, text: str) -> str:
        """
        Detect likely gender of the patient from text.
        Returns "M", "F", or "Unknown".
        """
        male_terms = re.compile(r'\b(male|man|boy)\b', re.IGNORECASE)
        female_terms = re.compile(r'\b(female|woman|girl)\b', re.IGNORECASE)
        
        male_count = len(male_terms.findall(text))
        female_count = len(female_terms.findall(text))
        
        if male_count > female_count and male_count > 0:
            return "M"
        elif female_count > male_count and female_count > 0:
            return "F"
        else:
            # Optional LLM fallback for ambiguous cases
            try:
                prompt = f"""Given the following medical text, infer the likely gender of the patient (if mentioned). 
Respond with only 'M' for male, 'F' for female, or 'Unknown' if not stated.

Text: {text[:200]}...
"""
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                if "M" in response.upper():
                    return "M"
                elif "F" in response.upper():
                    return "F"
                else:
                    return "Unknown"
            except Exception:
                return "Unknown"

    
    def extract_text_fields(self, row: pd.Series, dataset_name: str) -> List[str]:
        """
        Extract relevant text fields from different datasets.
        
        Args:
            row: DataFrame row
            dataset_name: Name of the dataset
            
        Returns:
            List of text fields to analyze
        """
        text_fields = []
        
        if dataset_name == "askdocs":
            # For askdocs: Question, Physician Response, ChatGPT Response
            if 'Question' in row and pd.notna(row['Question']):
                text_fields.append(str(row['Question']))
                
        elif dataset_name == "oncqa":
            # For oncqa: Input, Output
            if 'Input' in row and pd.notna(row['Input']):
                text_fields.append(str(row['Input']))
                
        elif dataset_name == "usmle_derm":
            # For usmle_derm: case_vignette, choice_1, choice_2, choice_3, choice_4
            for col in ['case_vignette', 'choice_1', 'choice_2', 'choice_3', 'choice_4']:
                if col in row and pd.notna(row[col]):
                    text_fields.append(str(row[col]))

        elif dataset_name == "medisumqa":
            # For MeDiSumQA: question, answer
            for col in ['Input']:
                if col in row and pd.notna(row[col]):
                    text_fields.append(str(row[col]))
    
        elif dataset_name == "qpain":
            # For QPain: question, answer
            for col in ['Vignette']:
                if col in row and pd.notna(row[col]):
                    text_fields.append(str(row[col]))
        
        elif dataset_name == "sct":
            # For SCT: context, question, answer
            for col in ['sct_stem']:
                if col in row and pd.notna(row[col]):
                    text_fields.append(str(row[col]))
        
        return text_fields
    
    def check_patterns(self, text: str) -> Dict[str, List[str]]:
        """
        Check text against gender-relevant patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with category as key and list of matched terms as value
        """
        matches = {}
        for category, pattern in self.patterns.items():
            found_terms = pattern.findall(text)
            if found_terms:
                matches[category] = list(set(found_terms))  # Remove duplicates
        return matches
    
    def llm_classify(self, text: str) -> Dict[str, Any]:
        """
        Use LLM to classify text for gender-relevant medical content.
        
        Args:
            text: Text to classify
            
        Returns:
            Classification results
        """
        prompt = f"""Analyze the following medical text and identify if it contains any gender-relevant medical content. Specifically look for:

1. Gender-specific diseases (ovarian cancer, prostate cancer, etc.)
2. Pregnancy-related content
3. References to genitalia or reproductive organs
4. Gendered conditions (menopause, menstruation, etc.)
5. Hormonal conditions

Text: {text[:1000]}...

Please respond with a JSON object containing:
- "has_gender_content": boolean
- "categories": list of relevant categories found
- "confidence": float between 0-1
- "reasoning": brief explanation

Respond only with valid JSON."""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "has_gender_content": False,
                    "categories": [],
                    "confidence": 0.0,
                    "reasoning": "Could not parse LLM response"
                }
                
        except Exception as e:
            return {
                "has_gender_content": False,
                "categories": [],
                "confidence": 0.0,
                "reasoning": f"Error in LLM classification: {str(e)}"
            }
    
    def process_dataset(self, file_path: str, dataset_name: str) -> List[Dict[str, Any]]:
        """
        Process a single dataset file.
        
        Args:
            file_path: Path to CSV file
            dataset_name: Name of the dataset
            
        Returns:
            List of identified gender-relevant cases
        """
        print(f"Processing {dataset_name} dataset...")
        
        # Read the dataset
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from {dataset_name}")
        
        gender_relevant_cases = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_name}"):
            # Extract text fields
            text_fields = self.extract_text_fields(row, dataset_name)
            
            if not text_fields:
                continue
            
            # Combine all text fields
            combined_text = " ".join(text_fields)
            
            # Check patterns first (faster)
            pattern_matches = self.check_patterns(combined_text)
            
            # If patterns found, use LLM for confirmation
            if pattern_matches:
                llm_result = self.llm_classify(combined_text)

                patient_gender = self.detect_patient_gender(combined_text)
                
                if llm_result.get("has_gender_content", False) or pattern_matches:
                    case_info = {
                        'dataset': dataset_name,
                        'row_index': idx,
                        'patient_gender': patient_gender,
                        'text_fields': text_fields,
                        'pattern_matches': pattern_matches,
                        'llm_classification': llm_result,
                        'original_row': row.to_dict()
                    }
                    gender_relevant_cases.append(case_info)
        
        print(f"Found {len(gender_relevant_cases)} gender-relevant cases in {dataset_name}")
        return gender_relevant_cases
    
    def process_all_datasets(self, data_dir: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process all datasets in the baseline_data directory.
        
        Args:
            data_dir: Path to baseline_data directory
            
        Returns:
            Dictionary with dataset names as keys and lists of cases as values
        """
        results = {}
        
        # Define dataset files and their names
        datasets = {
            'askdocs.csv': 'askdocs',
            'oncqa.csv': 'oncqa', 
            'usmle_derm.csv': 'usmle_derm',
            'MeDiSumQA.csv': 'medisumqa',
            'qpain_combined.csv': 'qpain',
            'sct_cleaned_full.csv': 'sct'
        }
        
        for filename, dataset_name in datasets.items():
            file_path = os.path.join(data_dir, filename)
            if os.path.exists(file_path):
                try:
                    cases = self.process_dataset(file_path, dataset_name)
                    results[dataset_name] = cases
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    results[dataset_name] = []
            else:
                print(f"File not found: {file_path}")
                results[dataset_name] = []
        
        return results
    
    def save_results(self, results: Dict[str, List[Dict[str, Any]]], output_dir: str):
        """
        Save results to JSON files.
        
        Args:
            results: Results dictionary
            output_dir: Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual dataset results
        for dataset_name, cases in results.items():
            output_file = os.path.join(output_dir, f"{dataset_name}_gender_cases.json")
            with open(output_file, 'w') as f:
                json.dump(cases, f, indent=2, default=str)
            print(f"Saved {len(cases)} cases from {dataset_name} to {output_file}")
        
        # Save summary
        summary = {
            'total_cases': sum(len(cases) for cases in results.values()),
            'dataset_counts': {name: len(cases) for name, cases in results.items()},
            'categories_found': self._get_category_summary(results)
        }
        
        summary_file = os.path.join(output_dir, "gender_cases_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {summary_file}")
    
    def _get_category_summary(self, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
        """Get summary of categories found across all datasets."""
        category_counts = {}
        for dataset_cases in results.values():
            for case in dataset_cases:
                for category in case.get('pattern_matches', {}).keys():
                    category_counts[category] = category_counts.get(category, 0) + 1
        return category_counts


def main():
    """Main function to run the gender medical classifier."""
    # Initialize classifier
    classifier = GenderMedicalClassifier()
    
    # Process all datasets
    data_dir = "/orcd/compute/mghassem/001/scratch/abinitha/abinitha/MedPerturb/medperturb-experiments/data/baseline_data"
    results = classifier.process_all_datasets(data_dir)
    
    # Save results
    output_dir = "/orcd/compute/mghassem/001/scratch/abinitha/abinitha/MedPerturb/medperturb-experiments/results/gendered_cases"
    classifier.save_results(results, output_dir)
    
    # Print summary
    total_cases = sum(len(cases) for cases in results.values())
    print(f"\n=== SUMMARY ===")
    print(f"Total gender-relevant cases found: {total_cases}")
    for dataset_name, cases in results.items():
        print(f"{dataset_name}: {len(cases)} cases")


if __name__ == "__main__":
    main()
