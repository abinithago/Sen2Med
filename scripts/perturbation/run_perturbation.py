#!/usr/bin/env python3
"""
Unified Perturbation Runner for MedPerturb Dataset

Runs one of four perturbation types on medperturb_format.csv:
  - gender_swap: swaps patient gender in clinical contexts
  - gender_removal: removes gender markers, converts to gender-neutral language
  - colorful: adds dramatic/colorful language with intensifiers (OncQA + AskDocs only)
  - uncertain: adds uncertain/anxious hedging language (OncQA + AskDocs only)

Uses Azure OpenAI (GPT-4o) by default.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
import re
from tqdm import tqdm

from prompts import get_prompter

# Import Azure OpenAI client from utils.py
# sys.path.insert(0, INSERTHERE)
from utils import get_gpt4_client


SYSTEM_MESSAGES = {
    'gender_swap': "You are a medical text editor expert at gender swapping in clinical contexts.",
    'gender_removal': "You are a medical text editor expert at removing gender markers and converting clinical contexts to gender-neutral language.",
    'colorful': "You are a medical text editor expert at adding dramatic/colorful language to patient messages.",
    'uncertain': "You are a medical text editor expert at adding uncertain/anxious language to patient messages.",
}

PERTURBATION_LABELS = {
    'gender_swap': 'gender_swap',
    'gender_removal': 'gender_removal',
    'colorful': 'colorful',
    'uncertain': 'uncertain',
}

ERROR_LABELS = {
    'gender_swap': 'gender_swap_error',
    'gender_removal': 'gender_removal_error',
    'colorful': 'colorful_tone_error',
    'uncertain': 'uncertain_tone_error',
}

# Datasets allowed for tone perturbations
TONE_ALLOWED_DATASETS = ['oncqa', 'askdocs']


class PerturbationProcessor:
    """Unified perturbation processor using Azure OpenAI."""

    def __init__(self, perturbation_type: str, model_name: str = "gpt-4o"):
        """Initialize the processor."""
        self.perturbation_type = perturbation_type
        self.model_name = model_name
        self.prompter = get_prompter(perturbation_type)
        self.client = None

    def load_model(self):
        """Load the Azure OpenAI client."""
        print(f"Initializing Azure OpenAI client for model: {self.model_name}")
        self.client = get_gpt4_client()
        print("Azure OpenAI client initialized successfully!")

    def extract_patient_message(self, clinical_context: str, dataset_name: str) -> tuple:
        """
        Extract the patient message from clinical context for tone perturbations.

        Returns:
            (clinical_context_prefix, patient_message)
        """
        if dataset_name in ['qpain', 'oncqa']:
            # Split at "Patient message:"
            if 'Patient message:' in clinical_context:
                parts = clinical_context.split('Patient message:')
                clinical_prefix = parts[0].strip()
                patient_message = parts[1].strip() if len(parts) > 1 else ""
                return clinical_prefix, patient_message
            else:
                # No patient message found
                return clinical_context, ""
        elif dataset_name == 'askdocs':
            # For AskDocs, the entire input needs to be modified
            return "", clinical_context
        else:
            return clinical_context, ""

    def split_text_by_words(self, text: str, max_words: int = 150) -> tuple:
        """Split text into first N words and remaining words."""
        words = text.split()
        if len(words) <= max_words:
            return text, ""
        first_part = " ".join(words[:max_words])
        remaining_part = " ".join(words[max_words:])
        return first_part, remaining_part

    def generate_text(self, dataset_name: str, text: str, max_new_tokens: int = 512) -> str:
        """Generate perturbed text from prompt using Azure OpenAI chat completions."""
        if self.client is None:
            self.load_model()

        # Get the prompt from the prompter
        prompt = self.prompter.get_prompt(dataset_name, text)

        # Format as messages for chat model
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGES[self.perturbation_type]},
            {"role": "user", "content": prompt}
        ]

        # Tone perturbations use a slightly higher temperature
        temperature = 0.3 if self.perturbation_type in ('colorful', 'uncertain') else 0.1
        max_tokens = 500 if self.perturbation_type in ('colorful', 'uncertain') else max_new_tokens

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        generated_text = response.choices[0].message.content.strip()
        return generated_text

    def process_row_gender(self, row: pd.Series) -> dict:
        """Process a single row for gender perturbations (swap or removal)."""
        dataset_name = row['dataset']
        clinical_context = str(row['clinical_context'])
        context_id = row['context_id']

        try:
            perturbed_text = self.generate_text(dataset_name, clinical_context)

            return {
                'dataset': dataset_name,
                'context_id': context_id,
                'clinical_context': perturbed_text,
                'original_gender': row['original_gender'],
                'age': row['age'],
                'gendered_condition': row['gendered_condition'],
                'perturbation': PERTURBATION_LABELS[self.perturbation_type],
                'provided_physician_response': row['provided_physician_response']
            }

        except Exception as e:
            print(f"\nError processing row {context_id} (dataset: {dataset_name}): {e}")
            return {
                'dataset': dataset_name,
                'context_id': context_id,
                'clinical_context': f"[ERROR: {e}] {clinical_context}",
                'original_gender': row['original_gender'],
                'age': row['age'],
                'gendered_condition': row['gendered_condition'],
                'perturbation': ERROR_LABELS[self.perturbation_type],
                'provided_physician_response': row['provided_physician_response']
            }

    def process_row_tone(self, row: pd.Series) -> dict:
        """Process a single row for tone perturbations (colorful or uncertain)."""
        dataset_name = row['dataset']
        clinical_context = str(row['clinical_context'])
        context_id = row['context_id']

        try:
            if dataset_name == 'oncqa':
                clinical_prefix, patient_message = self.extract_patient_message(clinical_context, dataset_name)

                if not patient_message:
                    # No patient message found, return original
                    return {
                        'dataset': dataset_name,
                        'context_id': context_id,
                        'clinical_context': clinical_context,
                        'original_gender': row['original_gender'],
                        'age': row['age'],
                        'gendered_condition': row['gendered_condition'],
                        'perturbation': PERTURBATION_LABELS[self.perturbation_type],
                        'provided_physician_response': row['provided_physician_response']
                    }

                # Split patient message into first 150 words and remaining
                first_150_words, remaining_words = self.split_text_by_words(patient_message, max_words=150)

                # Generate perturbed version of first 150 words
                perturbed_first_part = self.generate_text(dataset_name, first_150_words)

                # Combine perturbed first part with remaining words
                if remaining_words:
                    perturbed_message = f"{perturbed_first_part} {remaining_words}"
                else:
                    perturbed_message = perturbed_first_part

                # Reconstruct clinical context with perturbed patient message
                perturbed_clinical_context = f"{clinical_prefix}\nPatient message:\n\"{perturbed_message}\""

                return {
                    'dataset': dataset_name,
                    'context_id': context_id,
                    'clinical_context': perturbed_clinical_context,
                    'original_gender': row['original_gender'],
                    'age': row['age'],
                    'gendered_condition': row['gendered_condition'],
                    'perturbation': PERTURBATION_LABELS[self.perturbation_type],
                    'provided_physician_response': row['provided_physician_response']
                }

            elif dataset_name == 'askdocs':
                # For AskDocs, modify the entire input
                # Split into first 150 words and remaining
                first_150_words, remaining_words = self.split_text_by_words(clinical_context, max_words=150)

                # Generate perturbed version of first 150 words
                perturbed_first_part = self.generate_text(dataset_name, first_150_words)

                # Combine perturbed first part with remaining words
                if remaining_words:
                    perturbed_input = f"{perturbed_first_part} {remaining_words}"
                else:
                    perturbed_input = perturbed_first_part

                return {
                    'dataset': dataset_name,
                    'context_id': context_id,
                    'clinical_context': perturbed_input,
                    'original_gender': row['original_gender'],
                    'age': row['age'],
                    'gendered_condition': row['gendered_condition'],
                    'perturbation': PERTURBATION_LABELS[self.perturbation_type],
                    'provided_physician_response': row['provided_physician_response']
                }
            else:
                # Skip other datasets
                return None

        except Exception as e:
            print(f"\nError processing row {context_id} (dataset: {dataset_name}): {e}")
            return {
                'dataset': dataset_name,
                'context_id': context_id,
                'clinical_context': f"[ERROR: {e}] {clinical_context}",
                'original_gender': row['original_gender'],
                'age': row['age'],
                'gendered_condition': row['gendered_condition'],
                'perturbation': ERROR_LABELS[self.perturbation_type],
                'provided_physician_response': row['provided_physician_response']
            }

    def process_dataset(self, input_file: str, output_file: str, max_rows: int = None):
        """Process the dataset and generate perturbed versions."""
        print(f"Loading dataset from {input_file}")
        df = pd.read_csv(input_file)

        if max_rows is not None:
            df = df.head(max_rows)
            print(f"Processing first {max_rows} rows only")

        print(f"Total rows to process: {len(df)}")

        # For tone perturbations, filter to allowed datasets
        is_tone = self.perturbation_type in ('colorful', 'uncertain')
        if is_tone:
            df = df[df['dataset'].isin(TONE_ALLOWED_DATASETS)].copy()
            print(f"Filtered to allowed datasets ({', '.join(TONE_ALLOWED_DATASETS)}): {len(df)} rows")

            if len(df) == 0:
                print(f"No rows found for {self.perturbation_type} perturbation. Exiting.")
                return pd.DataFrame()

        # Load client if not already loaded
        if self.client is None:
            self.load_model()

        # Process each row
        results = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            if is_tone:
                result = self.process_row_tone(row)
            else:
                result = self.process_row_gender(row)

            if result is not None:
                results.append(result)

        if len(results) == 0:
            print("No results generated. Exiting.")
            return pd.DataFrame()

        # Create output dataframe
        output_df = pd.DataFrame(results)

        # Save to CSV
        output_df.to_csv(output_file, index=False)
        print(f"\nSaved perturbed data to {output_file}")

        # Print statistics
        print(f"\nStatistics:")
        print(f"Total rows processed: {len(output_df)}")
        print(f"By dataset:")
        print(output_df['dataset'].value_counts())
        print(f"\nBy perturbation type:")
        print(output_df['perturbation'].value_counts())

        return output_df


def main():
    parser = argparse.ArgumentParser(description='Generate perturbed dataset using Azure OpenAI')
    parser.add_argument('--type', type=str, required=True,
                        choices=['gender_swap', 'gender_removal', 'colorful', 'uncertain'],
                        help='Type of perturbation to apply')
    parser.add_argument('--input', type=str, default='cleaned_data/medperturb_format.csv',
                        help='Input CSV file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='Azure OpenAI model/deployment name to use')
    parser.add_argument('--max-rows', type=int, default=None,
                        help='Maximum number of rows to process (for testing)')

    args = parser.parse_args()

    print(f"Perturbation Type: {args.type}")
    print("=" * 60)

    processor = PerturbationProcessor(
        perturbation_type=args.type,
        model_name=args.model,
    )

    try:
        result_df = processor.process_dataset(
            input_file=args.input,
            output_file=args.output,
            max_rows=args.max_rows
        )

        print(f"\nSuccessfully created {args.type} dataset")
        print(f"Output file: {args.output}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
