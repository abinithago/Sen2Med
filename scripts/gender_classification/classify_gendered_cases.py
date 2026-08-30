#!/usr/bin/env python3
"""
Classify Gender-Relevant Medical Cases and Add gendered_condition Column

This script:
1. Runs the EfficientGenderClassifier to identify gender-relevant medical cases
2. Saves JSON classification results
3. Adds a 'gendered_condition' column to baseline CSVs based on the results
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Any, Set
import json
import os
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class EfficientGenderClassifier:
    def __init__(self):
        """
        Initialize the efficient classifier using pattern matching and optional LLM.
        """
        # Comprehensive gender-relevant medical terms
        self.gender_terms = {
            'pregnancy': [
                'pregnant', 'pregnancy', 'gestation', 'gestational', 'maternal', 'fetal', 'fetus',
                'prenatal', 'antenatal', 'postnatal', 'postpartum', 'lactation', 'breastfeeding',
                'contraception', 'contraceptive', 'birth control', 'conception', 'ovulation',
                'fertility', 'infertility', 'miscarriage', 'abortion', 'stillbirth', 'trimester',
                'amniocentesis', 'ultrasound', 'prenatal care', 'maternal health', 'prenatal screening'
            ],
            'genitalia': [
                'penis', 'vagina', 'vulva', 'clitoris', 'testicle', 'testis', 'scrotum',
                'ovary', 'ovaries', 'uterus', 'cervix', 'fallopian', 'prostate', 'prostate',
                'genital', 'genitalia', 'reproductive', 'reproduction', 'labia', 'mons pubis',
                'epididymis', 'vas deferens', 'seminal vesicle', 'bulbourethral', 'bartholin'
            ],
            'gendered_conditions': [
                'menopause', 'menstruation', 'menstrual', 'period', 'menses', 'dysmenorrhea',
                'amenorrhea', 'menorrhagia', 'endometriosis', 'fibroids', 'uterine', 'uterine fibroids',
                'ovarian', 'cervical', 'prostate', 'testicular', 'breast', 'mammary',
                'gynecological', 'gynecology', 'urological', 'urology', 'andrology',
                'premenstrual', 'pms', 'dyspareunia', 'vaginismus', 'vulvodynia',
                'erectile dysfunction', 'premature ejaculation', 'low testosterone'
            ],
            'gender_specific_cancers': [
                'ovarian cancer', 'cervical cancer', 'uterine cancer', 'endometrial cancer',
                'prostate cancer', 'testicular cancer', 'breast cancer', 'penile cancer',
                'vulvar cancer', 'vaginal cancer', 'ovarian carcinoma', 'cervical carcinoma',
                'endometrial carcinoma', 'prostate carcinoma', 'testicular carcinoma',
                'breast carcinoma', 'penile carcinoma', 'vulvar carcinoma', 'vaginal carcinoma'
            ],
            'hormonal_conditions': [
                'estrogen', 'progesterone', 'testosterone', 'hormone', 'hormonal',
                'pcos', 'polycystic ovary', 'hirsutism', 'androgen', 'androgenic',
                'hormone therapy', 'hormone replacement', 'hrt', 'birth control pill',
                'oral contraceptive', 'hormonal imbalance', 'thyroid', 'thyroid hormone'
            ],
            'reproductive_health': [
                'fertility treatment', 'ivf', 'in vitro fertilization', 'artificial insemination',
                'sperm donation', 'egg donation', 'surrogacy', 'tubal ligation', 'vasectomy',
                'hysterectomy', 'oophorectomy', 'orchiectomy', 'mastectomy', 'lumpectomy'
            ]
        }

        # Compile regex patterns for efficient matching
        self.patterns = {}
        for category, terms in self.gender_terms.items():
            # Create case-insensitive pattern with word boundaries
            pattern = r'\b(?:' + '|'.join(re.escape(term) for term in terms) + r')\b'
            self.patterns[category] = re.compile(pattern, re.IGNORECASE)

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
            for col in ['Question', 'Physician Response', 'ChatGPT Response']:
                if col in row and pd.notna(row[col]):
                    text_fields.append(str(row[col]))

        elif dataset_name == "oncqa":
            # For oncqa: Input, Output
            for col in ['Input', 'Output']:
                if col in row and pd.notna(row[col]):
                    text_fields.append(str(row[col]))

        elif dataset_name == "usmle_derm":
            # For usmle_derm: case_vignette, choice_1, choice_2, choice_3, choice_4
            for col in ['case_vignette', 'choice_1', 'choice_2', 'choice_3', 'choice_4']:
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
                # Remove duplicates and convert to lowercase for consistency
                matches[category] = list(set([term.lower() for term in found_terms]))
        return matches

    def calculate_confidence_score(self, pattern_matches: Dict[str, List[str]], text_length: int) -> float:
        """
        Calculate confidence score based on pattern matches and text characteristics.

        Args:
            pattern_matches: Dictionary of pattern matches
            text_length: Length of the text

        Returns:
            Confidence score between 0 and 1
        """
        if not pattern_matches:
            return 0.0

        # Base score from number of categories found
        category_score = min(len(pattern_matches) / 6.0, 1.0)  # 6 total categories

        # Bonus for multiple terms in same category
        term_density = sum(len(terms) for terms in pattern_matches.values()) / max(text_length / 100, 1)
        density_score = min(term_density, 1.0)

        # Bonus for specific high-value terms
        high_value_terms = ['cancer', 'pregnancy', 'menopause', 'menstruation', 'ovarian', 'prostate']
        high_value_score = 0
        for category_terms in pattern_matches.values():
            for term in category_terms:
                if any(hv_term in term for hv_term in high_value_terms):
                    high_value_score += 0.1

        high_value_score = min(high_value_score, 0.5)

        # Combine scores
        confidence = (category_score * 0.4 + density_score * 0.3 + high_value_score * 0.3)
        return min(confidence, 1.0)

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

            # Check patterns
            pattern_matches = self.check_patterns(combined_text)

            # If patterns found, include in results
            if pattern_matches:
                confidence = self.calculate_confidence_score(pattern_matches, len(combined_text))

                case_info = {
                    'dataset': dataset_name,
                    'row_index': idx,
                    'text_fields': text_fields,
                    'combined_text': combined_text,
                    'pattern_matches': pattern_matches,
                    'confidence_score': confidence,
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
            'usmle_derm.csv': 'usmle_derm'
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
        Save results to JSON files and CSV summaries.

        Args:
            results: Results dictionary
            output_dir: Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save individual dataset results
        for dataset_name, cases in results.items():
            # Save detailed JSON
            output_file = os.path.join(output_dir, f"{dataset_name}_gender_cases.json")
            with open(output_file, 'w') as f:
                json.dump(cases, f, indent=2, default=str)
            print(f"Saved {len(cases)} cases from {dataset_name} to {output_file}")

            # Save CSV summary
            if cases:
                summary_data = []
                for case in cases:
                    summary_data.append({
                        'dataset': case['dataset'],
                        'row_index': case['row_index'],
                        'confidence_score': case['confidence_score'],
                        'categories_found': list(case['pattern_matches'].keys()),
                        'total_matches': sum(len(terms) for terms in case['pattern_matches'].values()),
                        'text_preview': case['combined_text'][:200] + "..." if len(case['combined_text']) > 200 else case['combined_text']
                    })

                summary_df = pd.DataFrame(summary_data)
                csv_file = os.path.join(output_dir, f"{dataset_name}_summary.csv")
                summary_df.to_csv(csv_file, index=False)
                print(f"Saved summary CSV to {csv_file}")

        # Save overall summary
        summary = {
            'total_cases': sum(len(cases) for cases in results.values()),
            'dataset_counts': {name: len(cases) for name, cases in results.items()},
            'categories_found': self._get_category_summary(results),
            'confidence_distribution': self._get_confidence_distribution(results)
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

    def _get_confidence_distribution(self, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Get confidence score distribution."""
        all_scores = []
        for dataset_cases in results.values():
            for case in dataset_cases:
                all_scores.append(case.get('confidence_score', 0))

        if not all_scores:
            return {}

        return {
            'mean': np.mean(all_scores),
            'median': np.median(all_scores),
            'std': np.std(all_scores),
            'min': np.min(all_scores),
            'max': np.max(all_scores)
        }


class GenderColumnAdder:
    def __init__(self, results_dir: str):
        """
        Initialize the gender column adder.

        Args:
            results_dir: Directory containing gender classification results
        """
        self.results_dir = results_dir
        self.gender_cases = {}

    def load_gender_results(self):
        """Load gender classification results from JSON files."""
        print("Loading gender classification results...")

        datasets = ['askdocs', 'oncqa', 'usmle_derm']

        for dataset_name in datasets:
            json_file = os.path.join(self.results_dir, f"{dataset_name}_gender_cases.json")

            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    cases = json.load(f)

                # Create a set of row indices that have gender-relevant content
                gendered_indices = set()
                for case in cases:
                    row_index = case.get('row_index')
                    if row_index is not None:
                        gendered_indices.add(int(row_index))

                self.gender_cases[dataset_name] = gendered_indices
                print(f"Loaded {len(gendered_indices)} gendered cases for {dataset_name}")
            else:
                print(f"No results file found for {dataset_name}")
                self.gender_cases[dataset_name] = set()

    def add_gender_column_to_dataset(self, dataset_path: str, dataset_name: str) -> pd.DataFrame:
        """
        Add gendered_condition column to a dataset.

        Args:
            dataset_path: Path to the CSV file
            dataset_name: Name of the dataset

        Returns:
            DataFrame with added gendered_condition column
        """
        print(f"Processing {dataset_name} dataset...")

        # Load the dataset
        df = pd.read_csv(dataset_path)
        print(f"Loaded {len(df)} rows from {dataset_name}")

        # Get gendered indices for this dataset
        gendered_indices = self.gender_cases.get(dataset_name, set())

        # Add gendered_condition column
        df['gendered_condition'] = df.index.isin(gendered_indices)

        # Print statistics
        gendered_count = df['gendered_condition'].sum()
        print(f"Added gendered_condition column: {gendered_count} gendered cases out of {len(df)} total")
        print(f"Percentage gendered: {gendered_count/len(df)*100:.2f}%")

        return df

    def process_all_datasets(self, data_dir: str, output_dir: str):
        """
        Process all datasets and add gender columns.

        Args:
            data_dir: Directory containing original datasets
            output_dir: Directory to save updated datasets
        """
        os.makedirs(output_dir, exist_ok=True)

        # Define dataset files and their names
        datasets = {
            'askdocs.csv': 'askdocs',
            'oncqa.csv': 'oncqa',
            'usmle_derm.csv': 'usmle_derm'
        }

        summary_stats = {}

        for filename, dataset_name in datasets.items():
            input_path = os.path.join(data_dir, filename)
            output_path = os.path.join(output_dir, filename)

            if os.path.exists(input_path):
                try:
                    # Add gender column
                    df_with_gender = self.add_gender_column_to_dataset(input_path, dataset_name)

                    # Save updated dataset
                    df_with_gender.to_csv(output_path, index=False)
                    print(f"Saved updated dataset to {output_path}")

                    # Store statistics (convert numpy types to Python types for JSON serialization)
                    summary_stats[dataset_name] = {
                        'total_rows': int(len(df_with_gender)),
                        'gendered_rows': int(df_with_gender['gendered_condition'].sum()),
                        'percentage_gendered': float(df_with_gender['gendered_condition'].mean() * 100)
                    }

                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    summary_stats[dataset_name] = {'error': str(e)}
            else:
                print(f"File not found: {input_path}")
                summary_stats[dataset_name] = {'error': 'File not found'}

        # Save summary statistics
        summary_file = os.path.join(output_dir, "gender_column_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        print(f"Saved summary statistics to {summary_file}")

        return summary_stats


def main():
    parser = argparse.ArgumentParser(
        description='Classify gender-relevant medical cases and add gendered_condition column to CSVs'
    )
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing input baseline CSVs (askdocs.csv, oncqa.csv, usmle_derm.csv)')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory where JSON classification results will be saved')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory where CSVs with the added gendered_condition column will be saved')

    args = parser.parse_args()

    print("=" * 60)
    print("Step 1: Running gender classification")
    print("=" * 60)

    # Run classification
    classifier = EfficientGenderClassifier()
    results = classifier.process_all_datasets(args.data_dir)

    # Save JSON results
    classifier.save_results(results, args.results_dir)

    total_cases = sum(len(cases) for cases in results.values())
    print(f"\nTotal gender-relevant cases found: {total_cases}")
    for dataset_name, cases in results.items():
        print(f"  {dataset_name}: {len(cases)} cases")
        if cases:
            avg_confidence = np.mean([case['confidence_score'] for case in cases])
            print(f"    Average confidence: {avg_confidence:.3f}")

    print("\n" + "=" * 60)
    print("Step 2: Adding gendered_condition column to CSVs")
    print("=" * 60)

    # Add gender column to CSVs
    adder = GenderColumnAdder(args.results_dir)
    adder.load_gender_results()
    summary_stats = adder.process_all_datasets(args.data_dir, args.output_dir)

    print("\n=== FINAL SUMMARY ===")
    total_rows = 0
    total_gendered = 0

    for dataset_name, stats in summary_stats.items():
        if 'error' not in stats:
            print(f"{dataset_name}:")
            print(f"  Total rows: {stats['total_rows']}")
            print(f"  Gendered rows: {stats['gendered_rows']}")
            print(f"  Percentage: {stats['percentage_gendered']:.2f}%")
            print()
            total_rows += stats['total_rows']
            total_gendered += stats['gendered_rows']
        else:
            print(f"{dataset_name}: Error - {stats['error']}")

    if total_rows > 0:
        overall_percentage = (total_gendered / total_rows) * 100
        print(f"Overall: {total_gendered} gendered cases out of {total_rows} total ({overall_percentage:.2f}%)")

    print(f"\nUpdated datasets saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
