#!/usr/bin/env python3
"""
Unified Subsampling Script for MedPerturb Dataset

Modes:
  all       - run all subsampling steps in sequence
  baseline  - subsample baseline.csv (50 per dataset: oncqa, askdocs, MeDiSumQA, sct, usmle_derm)
  gender    - subsample gender_swap.csv and gender_removal.csv (50 per dataset: askdocs, sct, usmle_derm, oncqa)
  tone      - subsample colorful.csv and uncertain.csv (50 per dataset: askdocs, oncqa)
  combined  - create combined file matching baseline rows to all their perturbations
"""

import pandas as pd
import random
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Core sampling logic
# ---------------------------------------------------------------------------

def subsample_condition(input_file, output_file, datasets, seed=42, samples_per_dataset=50):
    """
    Filter for gendered_condition==False, deduplicate, sample N per dataset, save origin+content.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        datasets: Dict mapping display name → dataset column value
        seed: Random seed for reproducibility
        samples_per_dataset: Number of samples per dataset

    Returns:
        (total_count, dataset_counts dict) or None on error
    """
    random.seed(seed)

    # Read the CSV file
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Initial rows: {len(df):,}")

    # Filter for gendered_condition == False (handle both boolean and string "FALSE")
    print("\nFiltering for gendered_condition == False...")
    df['gendered_condition_str'] = df['gendered_condition'].astype(str).str.upper()
    df_filtered = df[df['gendered_condition_str'] == 'FALSE'].copy()
    df_filtered = df_filtered.drop(columns=['gendered_condition_str'])
    print(f"After filtering: {len(df_filtered):,} rows")

    # Drop duplicates based on clinical_context
    print("\nDropping duplicates based on clinical_context...")
    initial_count = len(df_filtered)
    df_deduped = df_filtered.drop_duplicates(subset=['clinical_context'], keep='first')
    duplicates_removed = initial_count - len(df_deduped)
    print(f"Removed {duplicates_removed:,} duplicate rows")
    print(f"After deduplication: {len(df_deduped):,} rows")

    # Sample from each dataset
    sampled_dfs = []
    dataset_counts = {}

    print("\nSampling from each dataset:")
    for dataset_name, dataset_value in datasets.items():
        dataset_df = df_deduped[df_deduped['dataset'] == dataset_value].copy()
        available_count = len(dataset_df)

        if available_count == 0:
            print(f"  {dataset_name} ({dataset_value}): No samples available")
            dataset_counts[dataset_name] = 0
            continue
        elif available_count < samples_per_dataset:
            print(f"  {dataset_name} ({dataset_value}): Only {available_count} samples available (requested {samples_per_dataset})")
            sampled_df = dataset_df.sample(n=available_count, random_state=seed)
            dataset_counts[dataset_name] = available_count
        else:
            print(f"  {dataset_name} ({dataset_value}): Sampling {samples_per_dataset} from {available_count} available")
            sampled_df = dataset_df.sample(n=samples_per_dataset, random_state=seed)
            dataset_counts[dataset_name] = samples_per_dataset

        sampled_dfs.append(sampled_df)

    # Combine all sampled dataframes
    if not sampled_dfs:
        print("\nError: No samples found for any dataset!")
        return None

    df_sampled = pd.concat(sampled_dfs, ignore_index=True)

    # Select only context_id and clinical_context columns
    df_output = df_sampled[['context_id', 'clinical_context']].copy()

    # Rename columns
    df_output.rename(columns={
        'context_id': 'origin',
        'clinical_context': 'content'
    }, inplace=True)

    # Shuffle the final dataframe
    df_output = df_output.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Save to output file
    print(f"\nSaving {len(df_output)} samples to {output_file}...")
    df_output.to_csv(output_file, index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total samples: {len(df_output)}")
    print("\n  Samples per dataset:")
    for dataset_name, count in dataset_counts.items():
        print(f"    {dataset_name}: {count}")
    print("=" * 60)

    return len(df_output), dataset_counts


# ---------------------------------------------------------------------------
# Specific subsample functions
# ---------------------------------------------------------------------------

def subsample_baseline(data_dir, seed=42):
    """
    Subsample baseline.csv with baseline datasets.

    Datasets: oncqa, askdocs, MeDiSumQA, sct, usmle_derm
    """
    data_dir = Path(data_dir)
    input_file = data_dir / 'baseline.csv'
    output_file = data_dir / 'baseline_subsampled.csv'

    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        return

    dataset_mapping = {
        'OncQA': 'oncqa',
        'r/AskDocs': 'askdocs',
        'MediSumQA': 'MeDiSumQA',
        'SCT': 'sct',
        'USMLE': 'usmle_derm'
    }

    print("=" * 60)
    print("PROCESSING BASELINE")
    print("=" * 60)
    subsample_condition(
        input_file=str(input_file),
        output_file=str(output_file),
        datasets=dataset_mapping,
        seed=seed,
        samples_per_dataset=50
    )


def subsample_gender(data_dir, seed=42):
    """
    Subsample gender_swap.csv and gender_removal.csv.

    Datasets: askdocs, sct, usmle_derm, oncqa
    """
    data_dir = Path(data_dir)
    gender_swap_file = data_dir / 'gender_swap.csv'
    gender_removal_file = data_dir / 'gender_removal.csv'
    gender_swap_output = data_dir / 'gender_swap_subsampled.csv'
    gender_removal_output = data_dir / 'gender_removal_subsampled.csv'

    dataset_mapping = {
        'r/AskDocs': 'askdocs',
        'SCT': 'sct',
        'USMLE': 'usmle_derm',
        'OncQA': 'oncqa'
    }

    if not gender_swap_file.exists():
        print(f"Error: Input file {gender_swap_file} does not exist")
        return
    if not gender_removal_file.exists():
        print(f"Error: Input file {gender_removal_file} does not exist")
        return

    print("=" * 60)
    print("PROCESSING GENDER_SWAP")
    print("=" * 60)
    swap_result = subsample_condition(
        input_file=str(gender_swap_file),
        output_file=str(gender_swap_output),
        datasets=dataset_mapping,
        seed=seed,
        samples_per_dataset=50
    )

    print("\n\n")
    print("=" * 60)
    print("PROCESSING GENDER_REMOVAL")
    print("=" * 60)
    removal_result = subsample_condition(
        input_file=str(gender_removal_file),
        output_file=str(gender_removal_output),
        datasets=dataset_mapping,
        seed=seed,
        samples_per_dataset=50
    )

    if swap_result and removal_result:
        swap_count, swap_counts = swap_result
        removal_count, removal_counts = removal_result
        print("\n\n")
        print("=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"\nGender Swap: {swap_count} total samples")
        print(f"Gender Removal: {removal_count} total samples")
        print("\nGender Swap samples per dataset:")
        for dataset_name, count in swap_counts.items():
            print(f"  {dataset_name}: {count}")
        print("\nGender Removal samples per dataset:")
        for dataset_name, count in removal_counts.items():
            print(f"  {dataset_name}: {count}")
        print("=" * 60)


def subsample_tone(data_dir, seed=42):
    """
    Subsample colorful.csv and uncertain.csv.

    Datasets: askdocs, oncqa
    """
    data_dir = Path(data_dir)
    colorful_file = data_dir / 'colorful.csv'
    uncertain_file = data_dir / 'uncertain.csv'
    colorful_output = data_dir / 'colorful_subsampled.csv'
    uncertain_output = data_dir / 'uncertain_subsampled.csv'

    dataset_mapping = {
        'r/AskDocs': 'askdocs',
        'OncQA': 'oncqa'
    }

    if not colorful_file.exists():
        print(f"Error: Input file {colorful_file} does not exist")
        return
    if not uncertain_file.exists():
        print(f"Error: Input file {uncertain_file} does not exist")
        return

    print("=" * 60)
    print("PROCESSING COLORFUL TONE")
    print("=" * 60)
    colorful_result = subsample_condition(
        input_file=str(colorful_file),
        output_file=str(colorful_output),
        datasets=dataset_mapping,
        seed=seed,
        samples_per_dataset=50
    )

    print("\n\n")
    print("=" * 60)
    print("PROCESSING UNCERTAIN TONE")
    print("=" * 60)
    uncertain_result = subsample_condition(
        input_file=str(uncertain_file),
        output_file=str(uncertain_output),
        datasets=dataset_mapping,
        seed=seed,
        samples_per_dataset=50
    )

    if colorful_result and uncertain_result:
        colorful_count, colorful_counts = colorful_result
        uncertain_count, uncertain_counts = uncertain_result
        print("\n\n")
        print("=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"\nColorful Tone: {colorful_count} total samples")
        print(f"Uncertain Tone: {uncertain_count} total samples")
        print("\nColorful Tone samples per dataset:")
        for dataset_name, count in colorful_counts.items():
            print(f"  {dataset_name}: {count}")
        print("\nUncertain Tone samples per dataset:")
        for dataset_name, count in uncertain_counts.items():
            print(f"  {dataset_name}: {count}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Combined subsample
# ---------------------------------------------------------------------------

def load_and_index_file(file_path, index_cols=['dataset', 'context_id'], filter_ungendered=True):
    """Load a CSV file, filter for ungendered contexts, and create a multi-index for fast lookups."""
    df = pd.read_csv(file_path)

    # Filter for gendered_condition == False if requested
    if filter_ungendered and 'gendered_condition' in df.columns:
        df['gendered_condition_str'] = df['gendered_condition'].astype(str).str.upper()
        df = df[df['gendered_condition_str'] == 'FALSE'].copy()
        df = df.drop(columns=['gendered_condition_str'])

    df_indexed = df.set_index(index_cols)
    return df_indexed


def create_combined(data_dir, seed=42, samples_per_dataset=50):
    """
    Create combined subsample file matching baseline rows to their perturbations.

    Conditions included: baseline, gender_swap, gender_removal, colorful, uncertain
    (summaries are excluded from scope)

    For each sampled baseline row, adds corresponding perturbations:
      - r/AskDocs: gender_swap, gender_removal, uncertain, colorful
      - OncQA: gender_swap, gender_removal, uncertain, colorful
      - SCT: gender_swap, gender_removal
      - USMLE: gender_swap, gender_removal
      - MeDiSumQA: (no perturbations, baseline only)
    """
    random.seed(seed)
    data_dir = Path(data_dir)

    # Input files
    baseline_file = data_dir / 'baseline.csv'
    gender_swap_file = data_dir / 'gender_swap.csv'
    gender_removal_file = data_dir / 'gender_removal.csv'
    uncertain_file = data_dir / 'uncertain.csv'
    colorful_file = data_dir / 'colorful.csv'

    # Output file
    output_file = data_dir / 'combined_subsampled.csv'

    # Check required files
    required_files = [baseline_file, gender_swap_file, gender_removal_file, uncertain_file, colorful_file]
    for file_path in required_files:
        if not file_path.exists():
            print(f"Error: Required file {file_path} does not exist")
            return

    print("=" * 60)
    print("LOADING FILES")
    print("=" * 60)

    # Load baseline file and filter for ungendered contexts
    print(f"Loading baseline: {baseline_file}...")
    df_baseline = pd.read_csv(baseline_file)
    print(f"  Loaded {len(df_baseline):,} rows")

    # Filter baseline for gendered_condition == False
    if 'gendered_condition' in df_baseline.columns:
        df_baseline['gendered_condition_str'] = df_baseline['gendered_condition'].astype(str).str.upper()
        df_baseline = df_baseline[df_baseline['gendered_condition_str'] == 'FALSE'].copy()
        df_baseline = df_baseline.drop(columns=['gendered_condition_str'])
        print(f"  After filtering for ungendered contexts: {len(df_baseline):,} rows")

    # Load perturbation files and create indexed dataframes (all filtered for ungendered contexts)
    print(f"\nLoading perturbation files (filtering for ungendered contexts)...")
    df_gender_swap = load_and_index_file(gender_swap_file, filter_ungendered=True)
    print(f"  gender_swap: {len(df_gender_swap):,} rows")

    df_gender_removal = load_and_index_file(gender_removal_file, filter_ungendered=True)
    print(f"  gender_removal: {len(df_gender_removal):,} rows")

    df_uncertain = load_and_index_file(uncertain_file, filter_ungendered=True)
    print(f"  uncertain: {len(df_uncertain):,} rows")

    df_colorful = load_and_index_file(colorful_file, filter_ungendered=True)
    print(f"  colorful: {len(df_colorful):,} rows")

    # Dataset mapping
    dataset_mapping = {
        'r/AskDocs': 'askdocs',
        'OncQA': 'oncqa',
        'MediSumQA': 'MeDiSumQA',
        'SCT': 'sct',
        'USMLE': 'usmle_derm'
    }

    # Deduplicate baseline
    print("\n" + "=" * 60)
    print("DEDUPLICATING BASELINE")
    print("=" * 60)
    initial_count = len(df_baseline)
    df_baseline_deduped = df_baseline.drop_duplicates(subset=['clinical_context'], keep='first')
    duplicates_removed = initial_count - len(df_baseline_deduped)
    print(f"After deduplication: {len(df_baseline_deduped):,} rows (removed {duplicates_removed:,} duplicates)")

    # Sample from each dataset
    print("\n" + "=" * 60)
    print("SAMPLING FROM BASELINE")
    print("=" * 60)

    all_sampled_rows = []

    for dataset_name, dataset_value in dataset_mapping.items():
        dataset_df = df_baseline_deduped[df_baseline_deduped['dataset'] == dataset_value].copy()
        available_count = len(dataset_df)

        if available_count == 0:
            print(f"{dataset_name} ({dataset_value}): No samples available")
            continue
        elif available_count < samples_per_dataset:
            print(f"{dataset_name} ({dataset_value}): Only {available_count} samples available (requested {samples_per_dataset})")
            sampled_df = dataset_df.sample(n=available_count, random_state=seed)
        else:
            print(f"{dataset_name} ({dataset_value}): Sampling {samples_per_dataset} from {available_count} available")
            sampled_df = dataset_df.sample(n=samples_per_dataset, random_state=seed)

        all_sampled_rows.append(sampled_df)

    if not all_sampled_rows:
        print("\nError: No samples found for any dataset!")
        return

    df_baseline_sampled = pd.concat(all_sampled_rows, ignore_index=True)
    print(f"\nTotal baseline samples: {len(df_baseline_sampled)}")

    # Collect all corresponding rows
    print("\n" + "=" * 60)
    print("COLLECTING CORRESPONDING ROWS")
    print("=" * 60)

    # Suffix map for perturbation types
    pert_suffix_map = {
        'gender_swap': '_gs',
        'gender_removal': '_gr',
        'uncertain': '_u',
        'colorful': '_c',
    }

    combined_rows = []

    for idx, row in df_baseline_sampled.iterrows():
        dataset = row['dataset']
        context_id = row['context_id']

        # Add baseline row (no suffix)
        combined_rows.append({
            'origin': row['context_id'],
            'content': row['clinical_context']
        })

        # Add perturbations based on dataset
        if dataset == 'askdocs':
            perturbations = [
                ('gender_swap', df_gender_swap),
                ('gender_removal', df_gender_removal),
                ('uncertain', df_uncertain),
                ('colorful', df_colorful),
            ]
        elif dataset == 'oncqa':
            perturbations = [
                ('gender_swap', df_gender_swap),
                ('gender_removal', df_gender_removal),
                ('uncertain', df_uncertain),
                ('colorful', df_colorful),
            ]
        elif dataset in ('sct', 'usmle_derm'):
            perturbations = [
                ('gender_swap', df_gender_swap),
                ('gender_removal', df_gender_removal),
            ]
        elif dataset == 'MeDiSumQA':
            perturbations = []
        else:
            perturbations = []

        # Try to find matching rows in perturbation files
        for pert_name, pert_df in perturbations:
            try:
                if (dataset, context_id) in pert_df.index:
                    pert_row = pert_df.loc[(dataset, context_id)]
                    suffix = pert_suffix_map.get(pert_name, '')
                    origin_with_suffix = f"{context_id}{suffix}"

                    # Handle case where multiple rows exist (take first)
                    if isinstance(pert_row, pd.Series):
                        combined_rows.append({
                            'origin': origin_with_suffix,
                            'content': pert_row['clinical_context']
                        })
                    else:
                        # Multiple rows, take first
                        combined_rows.append({
                            'origin': origin_with_suffix,
                            'content': pert_row.iloc[0]['clinical_context']
                        })
            except KeyError:
                # Not found, skip silently
                pass
            except Exception as e:
                print(f"  Error looking up {pert_name} for {dataset} {context_id}: {e}")

    # Create final dataframe
    df_combined = pd.DataFrame(combined_rows)

    # Shuffle
    df_combined = df_combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Save
    print("\n" + "=" * 60)
    print("SAVING OUTPUT")
    print("=" * 60)
    print(f"Saving {len(df_combined)} rows to {output_file}...")
    df_combined.to_csv(output_file, index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total rows in output: {len(df_combined):,}")
    print(f"Baseline samples: {len(df_baseline_sampled):,}")
    print("\nBaseline samples per dataset:")
    for dataset_name, dataset_value in dataset_mapping.items():
        count = len(df_baseline_sampled[df_baseline_sampled['dataset'] == dataset_value])
        print(f"  {dataset_name}: {count}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Subsample MedPerturb data files'
    )
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing input CSV files and where output files will be saved')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'baseline', 'gender', 'tone', 'combined'],
                        help='Subsampling mode (default: all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    if args.mode == 'baseline' or args.mode == 'all':
        subsample_baseline(args.data_dir, seed=args.seed)

    if args.mode == 'gender' or args.mode == 'all':
        subsample_gender(args.data_dir, seed=args.seed)

    if args.mode == 'tone' or args.mode == 'all':
        subsample_tone(args.data_dir, seed=args.seed)

    if args.mode == 'combined' or args.mode == 'all':
        create_combined(args.data_dir, seed=args.seed)


if __name__ == '__main__':
    main()
