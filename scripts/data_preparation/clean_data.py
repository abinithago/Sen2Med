#!/usr/bin/env python3
"""
Unified Data Cleaning Script for MedPerturb Dataset

Modes:
  remove_errors       - removes rows where perturbation == 'gender_swap_error'
  remove_duplicates   - removes duplicate content rows (keeps first occurrence)
  clean_perturbation  - LLM-based cleaning of gender perturbation artifacts
  clean_tone          - regex-based cleaning of tone perturbation artifacts
"""

import pandas as pd
import numpy as np
import re
import argparse
import sys
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Mode: remove_errors
# ---------------------------------------------------------------------------

def remove_error_rows(input_file: str, output_file: str = None, in_place: bool = False):
    """
    Remove rows with gender_swap_error from the CSV file.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (if None and not in_place, adds '_cleaned' suffix)
        in_place: If True, overwrites the input file
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Reading CSV file: {input_file}")
    df = pd.read_csv(input_file)

    print(f"Original number of rows: {len(df)}")

    # Count errors before removal
    error_count = len(df[df['perturbation'] == 'gender_swap_error'])
    print(f"Number of rows with 'gender_swap_error': {error_count}")

    # Remove error rows
    df_cleaned = df[df['perturbation'] != 'gender_swap_error'].copy()

    print(f"Rows after removing errors: {len(df_cleaned)}")
    print(f"Removed {len(df) - len(df_cleaned)} rows")

    # Determine output file path
    if in_place:
        output_path = input_path
    elif output_file:
        output_path = Path(output_file)
    else:
        # Add '_cleaned' suffix before .csv extension
        output_path = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"

    # Save cleaned data
    print(f"\nSaving cleaned data to: {output_path}")
    df_cleaned.to_csv(output_path, index=False)

    # Print statistics
    print("\nStatistics after cleaning:")
    print("=" * 60)
    print(f"Total rows: {len(df_cleaned)}")
    print("\nBy dataset:")
    print(df_cleaned['dataset'].value_counts())
    print("\nBy perturbation type:")
    print(df_cleaned['perturbation'].value_counts())

    print(f"\nSuccessfully cleaned dataset saved to: {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# Mode: remove_duplicates
# ---------------------------------------------------------------------------

def remove_duplicate_content(input_file: str, output_file: str = None):
    """
    Remove duplicate content rows from a CSV file.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (if None, overwrites input file)
    """
    if output_file is None:
        output_file = input_file

    # Read the CSV file
    df = pd.read_csv(input_file)

    # Get initial row count
    initial_count = len(df)

    # Remove duplicate content rows, keeping the first occurrence
    # This will keep the first row for each unique content value
    df_deduped = df.drop_duplicates(subset=['content'], keep='first')

    # Get final row count
    final_count = len(df_deduped)
    removed_count = initial_count - final_count

    # Save the deduplicated dataframe
    df_deduped.to_csv(output_file, index=False)

    return initial_count, final_count, removed_count


def run_remove_duplicates(input_file: str, output_file: str = None):
    """
    Run duplicate removal on a single file or all CSVs in a directory.

    If input_file is a directory, processes all CSVs in it.
    If input_file is a file, processes that file.
    """
    input_path = Path(input_file)

    if input_path.is_dir():
        data_dir = input_path
        csv_files = list(data_dir.glob('*.csv'))

        if not csv_files:
            print(f"No CSV files found in {data_dir}")
            return

        print(f"Found {len(csv_files)} CSV file(s) to process\n")

        total_initial = 0
        total_final = 0
        total_removed = 0

        for csv_file in sorted(csv_files):
            print(f"Processing: {csv_file.name}")

            try:
                initial, final, removed = remove_duplicate_content(str(csv_file))
                total_initial += initial
                total_final += final
                total_removed += removed

                if removed > 0:
                    print(f"  - Initial rows: {initial:,}")
                    print(f"  - Final rows: {final:,}")
                    print(f"  - Removed duplicates: {removed:,}")
                else:
                    print(f"  - No duplicates found ({initial:,} rows)")
                print()

            except Exception as e:
                print(f"  - Error processing {csv_file.name}: {e}\n")

        print("=" * 60)
        print(f"Summary:")
        print(f"  Total initial rows: {total_initial:,}")
        print(f"  Total final rows: {total_final:,}")
        print(f"  Total duplicates removed: {total_removed:,}")
        print("=" * 60)

    else:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        print(f"Processing: {input_path.name}")
        initial, final, removed = remove_duplicate_content(str(input_path), output_file)

        if removed > 0:
            print(f"Initial rows: {initial:,}")
            print(f"Final rows: {final:,}")
            print(f"Removed duplicates: {removed:,}")
        else:
            print(f"No duplicates found ({initial:,} rows)")

        if output_file:
            print(f"Saved to: {output_file}")


# ---------------------------------------------------------------------------
# Mode: clean_perturbation (LLM-based)
# ---------------------------------------------------------------------------

class PerturbationCleaner:
    """Clean LLM artifacts from perturbed text using HuggingFace pipeline."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.3-70B-Instruct", device: int = 0):
        """Initialize the cleaner with a model."""
        self.model_name = model_name
        self.device = device
        self.pipe = None

    def load_model(self):
        """Load the LLM using pipeline (handles loading automatically)."""
        import torch
        print(f"Loading model: {self.model_name}")
        print("Using HuggingFace pipeline for reliable loading...")

        try:
            from transformers import pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model_name,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nTrying without GPU...")
            from transformers import pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model_name,
                model_kwargs={"torch_dtype": torch.float32},
                device_map="cpu",
            )
            print("Model loaded on CPU successfully!")

    def get_cleanup_prompt(self, text: str) -> str:
        """Generate a prompt to clean up the text."""
        prompt = f"""You are a medical text editor. The following text contains artifacts from automated processing that need to be removed.

Your task is to remove any prefatory text, explanations, or meta-commentary that was added during processing, and return ONLY the cleaned clinical text.

Remove phrases like:
- "Here is the output with the gender swapped:"
- "Here is the gender-swapped version:"
- "Since the input does not explicitly mention..."
- "Perturbed version:"
- Any explanations about what was or wasn't done
- Any meta-commentary about the text

Return ONLY the clean clinical text, starting directly with the patient information or clinical content. Do not include any explanation or preface.

Text to clean:
{text}

Cleaned text:"""
        return prompt

    def clean_text_with_llm(self, text: str, max_new_tokens: int = 1024) -> str:
        """Clean text using LLM to remove artifacts."""
        if self.pipe is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        prompt = self.get_cleanup_prompt(text)

        # Format as messages for chat model
        messages = [
            {"role": "system", "content": "You are a medical text editor that removes processing artifacts from clinical text. You only return clean text without any explanations."},
            {"role": "user", "content": prompt}
        ]

        try:
            # Generate using pipeline with chat format
            outputs = self.pipe(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
            )

            # Extract generated text (last message in the conversation)
            cleaned_text = outputs[0]["generated_text"][-1]["content"].strip()

            # Additional post-processing to remove common artifacts
            cleaned_text = self.post_process(cleaned_text)

            return cleaned_text

        except Exception as e:
            print(f"Error cleaning text with LLM: {e}")
            # Fall back to simple regex-based cleaning
            return self.simple_clean(text)

    def post_process(self, text: str) -> str:
        """Post-process to remove any remaining artifacts."""
        # Remove common prefixes if they somehow made it through
        patterns_to_remove = [
            r'^Here is the output.*?:',
            r'^Here is the.*?version.*?:',
            r'^Since the input.*?:',
            r'^Perturbed version.*?:',
            r'^Cleaned text.*?:',
            r'^Output.*?:',
            r'^Text.*?:',
        ]

        cleaned = text
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)

        # Remove leading/trailing whitespace and normalize
        cleaned = cleaned.strip()

        # If text starts with quotes, remove them
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1].strip()

        return cleaned

    def simple_clean(self, text: str) -> str:
        """Fallback simple regex-based cleaning if LLM fails."""
        # Remove common artifact patterns
        patterns = [
            r'^Since the input does not explicitly mention.*?(?=Patient|The patient|A )',
            r'^Since.*?output will remain the same.*?(?=Patient|The patient|A )',
            r'^Here is the output with the gender swapped:\s*',
            r'^Here is the gender-swapped version:\s*',
            r'^Perturbed version:\s*',
            r'^Here is the output:\s*',
        ]

        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)

        # Find the first sentence that looks like it starts the actual content
        # Usually starts with "Patient" or similar
        match = re.search(r'(Patient [A-Z]|The patient|A \d+)', cleaned)
        if match:
            cleaned = cleaned[match.start():]

        return cleaned.strip()

    def needs_cleaning(self, text: str) -> bool:
        """Check if text needs cleaning based on common artifact patterns."""
        artifact_patterns = [
            r'Here is the.*?output',
            r'Here is the.*?version',
            r'Since the input',
            r'Perturbed version',
            r'output will remain',
            r'no gender.*?to swap',
        ]

        for pattern in artifact_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def process_dataset(self, input_file: str, output_file: str, max_rows: int = None, use_llm: bool = True):
        """Process the dataset and clean perturbation artifacts."""
        print(f"Loading dataset from {input_file}")
        df = pd.read_csv(input_file)

        if max_rows is not None:
            df = df.head(max_rows)
            print(f"Processing first {max_rows} rows only")

        print(f"Total rows to process: {len(df)}")

        # Check which rows need cleaning
        needs_cleaning_mask = df['clinical_context'].apply(self.needs_cleaning)
        rows_needing_cleaning = needs_cleaning_mask.sum()
        print(f"Rows that need cleaning: {rows_needing_cleaning} / {len(df)}")

        if use_llm:
            # Load model if not already loaded
            if self.pipe is None:
                self.load_model()
        else:
            print("Using simple regex-based cleaning (no LLM)")

        # Process each row
        results = []
        cleaned_count = 0
        skipped_count = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            clinical_context = str(row['clinical_context'])

            # Check if this row needs cleaning
            if self.needs_cleaning(clinical_context):
                try:
                    if use_llm:
                        cleaned_text = self.clean_text_with_llm(clinical_context)
                    else:
                        cleaned_text = self.simple_clean(clinical_context)

                    # Only update if text actually changed
                    if cleaned_text != clinical_context and len(cleaned_text) > 50:
                        results.append({
                            'dataset': row['dataset'],
                            'context_id': row['context_id'],
                            'clinical_context': cleaned_text,
                            'original_gender': row['original_gender'],
                            'age': row['age'],
                            'gendered_condition': row['gendered_condition'],
                            'perturbation': row['perturbation'],
                            'provided_physician_response': row['provided_physician_response']
                        })
                        cleaned_count += 1
                    else:
                        # Keep original if cleaning didn't work well
                        results.append(row.to_dict())
                        skipped_count += 1

                except Exception as e:
                    print(f"\nError processing row {idx} (dataset: {row['dataset']}): {e}")
                    # Keep original row on error
                    results.append(row.to_dict())
                    skipped_count += 1
            else:
                # Keep original if it doesn't need cleaning
                results.append(row.to_dict())
                skipped_count += 1

        # Create output dataframe
        output_df = pd.DataFrame(results)

        # Save to CSV
        output_df.to_csv(output_file, index=False)
        print(f"\nSaved cleaned data to {output_file}")

        # Print statistics
        print(f"\nStatistics:")
        print(f"Total rows processed: {len(output_df)}")
        print(f"Rows cleaned: {cleaned_count}")
        print(f"Rows unchanged: {skipped_count}")
        print(f"\nBy dataset:")
        print(output_df['dataset'].value_counts())
        print(f"\nBy perturbation type:")
        print(output_df['perturbation'].value_counts())

        return output_df


# ---------------------------------------------------------------------------
# Mode: clean_tone (regex-based)
# ---------------------------------------------------------------------------

class ToneArtifactCleaner:
    """Clean LLM artifacts from tone-perturbed text."""

    def __init__(self):
        """Initialize the cleaner."""
        # Patterns to remove from the beginning of text
        # Make patterns more flexible to catch variations
        self.prefix_patterns = [
            r'^Here\'s the revised post with.*?:\s*\n+',
            r'^Here\'s the revised post.*?:\s*\n+',
            r'^Here\'s the revised version.*?:\s*\n+',
            r'^Here is the revised post.*?:\s*\n+',
            r'^Here is the revised version.*?:\s*\n+',
            r'^Revised post.*?:\s*\n+',
            r'^Revised version.*?:\s*\n+',
            r'^Here\'s.*?revised.*?:\s*\n+',
            r'^Here is.*?revised.*?:\s*\n+',
            r'^output with all gender markers removed:'
        ]

        # Patterns to remove "Note:" sections at the end
        self.note_patterns = [
            r'\n\s*Note that I\'ve.*$',
            r'\n\s*Note\s*:.*$',
            r'\n\s*NOTE:.*$',
            r'\n\s*Note:.*$',
            r'^\s*Note:.*$',
        ]

        # Patterns for explanatory text that might appear anywhere
        self.explanatory_patterns = [
            r'I\'ve tried to maintain.*?effect\.',
            r'I\'ve made minimal changes.*?symptoms\.',
            r'I made minor changes.*?message\.',
            r'I made.*?changes.*?preserve',
            r'Let me know if you\'d like me to revise anything!',
            r'to achieve this effect\.',
            r'while adding subtle.*?to emphasize.*?\.',
            r'while introducing subtle.*?uncertainty\.',
            r'while preserving.*?flow.*?\.',
            r'to convey a sense of uncertainty',
            r'to introduce doubt',
            r'I removed the gender marker',
            r'removed the gender information',
            r'removed.*?gender.*?marker',
            r'gender.*?removed',
            r'I also removed.*?Gender',
            r'I also removed.*?from',
            r'I also replaced.*?with',
            r'changed the pronouns',
            r'changed.*?pronouns.*?from',
            r'pronouns.*?from.*?to',
            r'to maintain a neutral tone',
            r'maintain.*?neutral.*?tone',
            r'replaced.*?with.*?family member',
            r'replaced.*?mum.*?with',
            r'The rest of the content remains unchanged',
            r'The rest of the content remains the same',
            r'remains unchanged',
            r'remains the same',
            r'remains.*?same.*?per.*?instructions',
        ]

    def split_into_sentences(self, text: str) -> list:
        """Split text into sentences while preserving structure."""
        # Normalize multiple spaces/newlines to single space, but preserve sentence boundaries
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Split on sentence endings (. ! ?) followed by space
        # Use a simple regex to find sentence boundaries
        sentences = []

        # Pattern: sentence ending (. ! ?) followed by space or end of string
        # Split but keep the punctuation
        parts = re.split(r'([.!?]+(?:\s+|$))', text)

        current = []
        for part in parts:
            if part.strip():
                current.append(part)
                # If this part ends with sentence punctuation, we have a complete sentence
                if re.search(r'[.!?]+\s*$', part):
                    sentence = ''.join(current).strip()
                    if sentence:
                        sentences.append(sentence)
                    current = []

        # Add any remaining text as a sentence
        if current:
            sentence = ''.join(current).strip()
            if sentence:
                sentences.append(sentence)

        return sentences

    def sentence_contains_artifact(self, sentence: str) -> bool:
        """Check if a sentence contains any artifact pattern."""
        # Check all artifact patterns
        all_patterns = (
            self.prefix_patterns +
            self.note_patterns +
            self.explanatory_patterns +
            [
                r'Here\'s the revised post',
                r'Here\'s the revised version',
                r'Here is the revised post',
                r'Here is the revised version',
                r'Revised post',
                r'Revised version',
                r'^Note:\s*I\'ve',
                r'^Note:\s*I made',
                r'^Note:\s*I\'ve made',
                r'^Note:\s*I removed',
                r'^Note that I\'ve',
                r'^Note that I made',
                r'Note:\s*I\'ve',
                r'Note:\s*I made',
                r'Note:\s*I removed',
                r'Note that I\'ve',
                r'Note that I made',
                r'Note:\s*I\'ve made',
                r'to achieve this effect',
                r'Let me know if you\'d like',
                r'I\'ve tried to maintain',
                r'I\'ve made minimal changes',
                r'I made minor changes',
                r'I made.*?changes',
                r'I removed.*?gender',
                r'removed.*?gender.*?marker',
                r'gender.*?removed',
                r'I also removed.*?Gender',
                r'I also removed.*?from',
                r'I also replaced.*?with',
                r'changed the pronouns',
                r'changed.*?pronouns.*?from',
                r'pronouns.*?from.*?to',
                r'to maintain a neutral tone',
                r'maintain.*?neutral.*?tone',
                r'replaced.*?with.*?family member',
                r'replaced.*?mum.*?with',
                r'remains unchanged',
                r'remains the same',
                r'remains.*?same.*?per.*?instructions',
                r'while adding subtle',
                r'while introducing subtle',
                r'while preserving',
                r'to convey a sense of uncertainty',
                r'to introduce doubt',
            ]
        )

        for pattern in all_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        return False

    def clean_text(self, text: str) -> str:
        """Clean artifacts from a single text by removing sentences containing artifacts."""
        if pd.isna(text) or not isinstance(text, str):
            return text

        # First, handle prefix patterns that appear at the very start (before any sentences)
        cleaned = text
        for pattern in self.prefix_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

        # Additional fallback patterns to catch any remaining prefixes
        # These are more aggressive and catch variations
        prefix_fallbacks = [
            r'^Here\'s the revised version.*?:\s*\n+',
            r'^Here\'s the revised post.*?:\s*\n+',
            r'^Here is the revised version.*?:\s*\n+',
            r'^Here is the revised post.*?:\s*\n+',
            r'^Revised\s+post.*?:\s*\n+',
            r'^Revised\s+version.*?:\s*\n+',
        ]
        for pattern in prefix_fallbacks:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

        # Remove any standalone "Revised post:" or "Revised version:" at the start
        cleaned = re.sub(r'^Revised\s+post\s*:?\s*\n?\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^Revised\s+version\s*:?\s*\n?\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^Here\'s\s+the\s+', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^Here\s+is\s+the\s+', '', cleaned, flags=re.IGNORECASE)

        # Final check: if text still starts with "Here's the revised" or "Here is the revised", remove it
        if re.match(r'^Here\'?s?\s+(the\s+)?revised', cleaned, re.IGNORECASE):
            cleaned = re.sub(r'^Here\'?s?\s+(the\s+)?revised.*?:\s*\n+', '', cleaned, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

        # Split into sentences
        sentences = self.split_into_sentences(cleaned)

        # Filter out sentences that contain artifact patterns
        cleaned_sentences = []
        for sentence in sentences:
            if not self.sentence_contains_artifact(sentence):
                cleaned_sentences.append(sentence)

        # Rejoin sentences with proper spacing
        cleaned = ' '.join(cleaned_sentences)

        # Clean up multiple newlines (more than 2 consecutive)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

        # Clean up multiple spaces
        cleaned = re.sub(r' +', ' ', cleaned)

        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()

        # Remove quotes if the entire text is wrapped in them
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1].strip()

        # Remove any trailing explanatory text that might have been missed
        # Look for patterns like "resources would be greatly appreciated. Thanks., Gender: F"
        cleaned = re.sub(r'\s+resources would be greatly appreciated\.\s+Thanks\.,\s+Gender:\s*[MF]\.?$', '', cleaned, flags=re.IGNORECASE)

        # Remove appended "Note:" sections at the end, even if not properly separated
        # These can be appended directly without newlines, sentence boundaries, or periods
        # Pattern: "Note:" followed by explanatory text until end
        # Try patterns both with and without leading whitespace/periods
        note_end_patterns = [
            # Patterns with leading space (most common)
            r'\s+Note\s*:\s*I removed.*?(?:remains unchanged|remains the same|unchanged|the same).*?\.?$',
            r'\s+Note\s*:\s*.*?removed.*?gender.*?(?:remains unchanged|remains the same|unchanged|the same).*?\.?$',
            r'\s+Note\s*:\s*.*?gender.*?removed.*?(?:remains unchanged|remains the same|unchanged|the same).*?\.?$',
            r'\s+Note\s*:\s*.*?(?:remains unchanged|remains the same).*?\.?$',
            r'\s+Note\s*:\s*.*?removed.*?gender.*?\.?$',
            r'\s+Note\s*:\s*.*?gender.*?removed.*?\.?$',
            r'\s+Note\s*:\s*I.*?removed.*?\.?$',
            r'\s+Note\s*:\s*.*?removed.*?\.?$',
            # Patterns without leading space (appended directly, no period before)
            r'Note\s*:\s*I removed.*?(?:remains unchanged|remains the same|unchanged|the same).*?\.?$',
            r'Note\s*:\s*.*?removed.*?gender.*?(?:remains unchanged|remains the same|unchanged|the same).*?\.?$',
            r'Note\s*:\s*.*?removed.*?marker.*?(?:remains unchanged|remains the same|unchanged|the same).*?\.?$',
            r'Note\s*:\s*.*?removed.*?gender.*?\.?$',
            r'Note\s*:\s*.*?removed.*?marker.*?\.?$',
            r'Note\s*:\s*I.*?removed.*?\.?$',
        ]
        for pattern in note_end_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)

        # Also check if the text ends with a Note: pattern (more general)
        # This catches Notes that are appended without space/newline separation or periods
        # Check last 500 characters for Notes (in case text is long)
        text_to_check = cleaned[-500:] if len(cleaned) > 500 else cleaned
        if re.search(r'Note\s*:\s*.*?(?:removed|gender|unchanged|the same|remains)', text_to_check, re.IGNORECASE):
            # Find the last occurrence of "Note:" - be very flexible, don't require anything before it
            # Search for "Note:" anywhere in the text, but prioritize ones near the end
            all_note_matches = list(re.finditer(r'Note\s*:', cleaned, re.IGNORECASE))

            if all_note_matches:
                # Get the last "Note:" occurrence
                last_note_match = all_note_matches[-1]
                last_note_pos = last_note_match.start()

                # Get the text from this Note to the end
                note_text = cleaned[last_note_pos:]

                # Check if this Note contains removal/unchanged/same language
                # Be very permissive - if it mentions removed, gender, unchanged, or remains, remove it
                if re.search(r'(?:removed.*?gender|gender.*?removed|removed.*?marker|remains (?:unchanged|the same)|unchanged|the same|I\s+(?:removed|made).*?gender|removed.*?information|removed.*?age)', note_text, re.IGNORECASE | re.DOTALL):
                    # Remove everything from this Note to the end
                    # But first, check if there's a space or period before the Note - if so, remove that too
                    before_note = cleaned[:last_note_pos]
                    # Remove trailing space/period before the Note
                    before_note = before_note.rstrip()
                    # If it ends with a period, keep it; otherwise we've already stripped spaces
                    cleaned = before_note

        # Also check for "I also..." sentences appended at the end (explanatory text without "Note:")
        # Check last 300 characters for "I also..." patterns
        text_to_check_also = cleaned[-300:] if len(cleaned) > 300 else cleaned
        if re.search(r'I also (?:removed|replaced|changed).*?(?:gender|pronouns|neutral|from|with)', text_to_check_also, re.IGNORECASE):
            # Find the last occurrence of "I also..."
            all_also_matches = list(re.finditer(r'I also (?:removed|replaced|changed)', cleaned, re.IGNORECASE))

            if all_also_matches:
                # Get the last "I also..." occurrence
                last_also_match = all_also_matches[-1]
                last_also_pos = last_also_match.start()

                # Get the text from this "I also..." to the end
                also_text = cleaned[last_also_pos:]

                # Check if this sentence contains removal/replacement/pronoun change language
                if re.search(r'(?:removed.*?Gender|removed.*?from|replaced.*?with|changed.*?pronouns|pronouns.*?from.*?to|neutral.*?tone|family member)', also_text, re.IGNORECASE | re.DOTALL):
                    # Remove everything from this "I also..." to the end
                    before_also = cleaned[:last_also_pos]
                    # Remove trailing space/period before the sentence
                    before_also = before_also.rstrip()
                    # If it ends with a period, keep it; otherwise we've already stripped spaces
                    cleaned = before_also

        # Final cleanup of leading/trailing whitespace
        cleaned = cleaned.strip()

        return cleaned

    def needs_cleaning(self, text: str) -> bool:
        """Check if text needs cleaning based on common artifact patterns."""
        if pd.isna(text) or not isinstance(text, str):
            return False

        artifact_patterns = [
            r'Here\'s the revised post',
            r'Here\'s the revised version',
            r'Here is the revised post',
            r'Here is the revised version',
            r'Revised post',
            r'Revised version',
            r'Note:\s*I\'ve',
            r'Note:\s*I made',
            r'Note:\s*I removed',
            r'Note that I\'ve',
            r'Note that I made',
            r'Note:\s*I\'ve made',
            r'removed.*?gender',
            r'gender.*?removed',
            r'I also removed.*?Gender',
            r'I also removed.*?from',
            r'I also replaced.*?with',
            r'changed the pronouns',
            r'changed.*?pronouns.*?from',
            r'pronouns.*?from.*?to',
            r'to maintain a neutral tone',
            r'maintain.*?neutral.*?tone',
            r'replaced.*?with.*?family member',
            r'replaced.*?mum.*?with',
            r'remains unchanged',
            r'remains the same',
            r'remains.*?same.*?per.*?instructions',
            r'to achieve this effect',
            r'Let me know if you\'d like',
            r'I\'ve tried to maintain',
            r'I\'ve made minimal changes',
            r'I made minor changes',
            r'to convey a sense of uncertainty',
            r'to introduce doubt',
        ]

        for pattern in artifact_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def process_dataset(self, input_file: str, output_file: str, max_rows: int = None):
        """Process the dataset and clean perturbation artifacts."""
        print(f"Loading dataset from {input_file}")
        df = pd.read_csv(input_file)

        if max_rows is not None:
            df = df.head(max_rows)
            print(f"Processing first {max_rows} rows only")

        print(f"Total rows to process: {len(df)}")

        # Check how many need cleaning
        needs_cleaning_count = 0
        for idx, row in df.iterrows():
            clinical_context = str(row.get('clinical_context', ''))
            if self.needs_cleaning(clinical_context):
                needs_cleaning_count += 1

        print(f"Rows that need cleaning: {needs_cleaning_count} / {len(df)}")

        # Process each row - clean all rows to be safe (cleaning is idempotent)
        cleaned_count = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning artifacts"):
            clinical_context = str(row.get('clinical_context', ''))

            # Always clean the text (cleaning is idempotent and safe)
            cleaned_context = self.clean_text(clinical_context)

            # Only count as "cleaned" if it actually changed
            if cleaned_context != clinical_context:
                df.at[idx, 'clinical_context'] = cleaned_context
                cleaned_count += 1

        print(f"\nCleaned {cleaned_count} rows")

        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"\nSaved cleaned data to {output_file}")

        # Print statistics
        print(f"\nStatistics:")
        print(f"Total rows: {len(df)}")
        print(f"Rows cleaned: {cleaned_count}")
        print(f"By perturbation type:")
        print(df['perturbation'].value_counts())

        return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Clean MedPerturb dataset files'
    )
    parser.add_argument('--mode', type=str, required=True,
                        choices=['remove_errors', 'remove_duplicates', 'clean_perturbation', 'clean_tone'],
                        help='Cleaning mode')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file (or directory for remove_duplicates)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file (optional; defaults depend on mode)')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.3-70B-Instruct',
                        help='Model name to use (for clean_perturbation mode)')
    parser.add_argument('--max-rows', type=int, default=None,
                        help='Maximum number of rows to process (for testing)')
    parser.add_argument('--in-place', action='store_true',
                        help='Overwrite the input file (for remove_errors mode; use with caution)')
    parser.add_argument('--no-llm', action='store_true',
                        help='Use simple regex-based cleaning instead of LLM (for clean_perturbation mode)')

    args = parser.parse_args()

    try:
        if args.mode == 'remove_errors':
            remove_error_rows(
                input_file=args.input,
                output_file=args.output,
                in_place=args.in_place
            )

        elif args.mode == 'remove_duplicates':
            run_remove_duplicates(
                input_file=args.input,
                output_file=args.output
            )

        elif args.mode == 'clean_perturbation':
            # Generate output filename if not provided
            if args.output is None:
                input_path = args.input
                if input_path.endswith('.csv'):
                    args.output = input_path[:-4] + '_cleaned.csv'
                else:
                    args.output = input_path + '_cleaned.csv'

            print("Perturbation Artifact Cleaning (LLM-based)")
            print("=" * 60)

            cleaner = PerturbationCleaner(model_name=args.model)
            cleaner.process_dataset(
                input_file=args.input,
                output_file=args.output,
                max_rows=args.max_rows,
                use_llm=not args.no_llm
            )

            print(f"\nSuccessfully cleaned perturbation artifacts")
            print(f"Output file: {args.output}")

        elif args.mode == 'clean_tone':
            # Generate output filename if not provided
            if args.output is None:
                input_path = args.input
                if input_path.endswith('.csv'):
                    args.output = input_path[:-4] + '_cleaned.csv'
                else:
                    args.output = input_path + '_cleaned.csv'

            print("Tone Perturbation Artifact Cleaning (regex-based)")
            print("=" * 60)

            cleaner = ToneArtifactCleaner()
            cleaner.process_dataset(
                input_file=args.input,
                output_file=args.output,
                max_rows=args.max_rows
            )

            print(f"\nSuccessfully cleaned tone perturbation artifacts")
            print(f"Output file: {args.output}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
