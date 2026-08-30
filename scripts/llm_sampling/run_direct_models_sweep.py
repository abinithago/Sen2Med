#!/usr/bin/env python3
"""
Clinical Decision Pipeline Sweep Script for Direct Model Loading
Runs the direct model pipeline for DeepSeek and MedGemma models.
Organizes output into: output_dir/model_name/perturbation_name/filename_seed.csv
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
from itertools import product


# Mapping from input filename to perturbation name
PERTURBATION_MAP = {
    'baseline.csv': 'baseline',
    'colorful.csv': 'colorful',
    'uncertain.csv': 'uncertain',
    'gender_swap.csv': 'gender_swap',
    'gender_removal.csv': 'gender_removal',
}


def sanitize_model_name(model_name: str) -> str:
    """Convert model name to a filesystem-safe directory name."""
    return model_name.replace('/', '_').replace('\\', '_').replace(':', '_')


def run_sweep(
    models: list,
    seeds: list,
    data_dir: str,
    output_base_dir: str,
    device: int = 0,
    max_rows: int = None,
    dry_run: bool = False
):
    """
    Run clinical decision pipeline sweep for direct models.
    
    Args:
        models: List of model names to test
        seeds: List of seeds to test
        data_dir: Directory containing input CSV files
        output_base_dir: Base output directory
        device: GPU device ID
        max_rows: Maximum rows to process (for testing)
        dry_run: If True, only print commands without executing
    """
    data_path = Path(data_dir)
    output_base_path = Path(output_base_dir)
    
    # Find all CSV files in data directory
    csv_files = sorted(data_path.glob("*.csv"))
    
    if len(csv_files) == 0:
        print(f"Error: No CSV files found in {data_dir}")
        return
    
    print("=" * 80)
    print("CLINICAL DECISION PIPELINE SWEEP (Direct Models)")
    print("=" * 80)
    print(f"Models: {models}")
    print(f"Seeds: {seeds}")
    print(f"Input directory: {data_dir}")
    print(f"Output base directory: {output_base_dir}")
    print(f"Found {len(csv_files)} input files")
    print("=" * 80)
    
    total_jobs = len(models) * len(seeds) * len(csv_files)
    print(f"\nTotal jobs to run: {total_jobs}")
    
    if dry_run:
        print("\n[DRY RUN MODE - No commands will be executed]\n")
    
    job_count = 0
    
    for model, seed, csv_file in product(models, seeds, csv_files):
        job_count += 1
        
        # Get perturbation name
        perturbation_name = PERTURBATION_MAP.get(csv_file.name, csv_file.stem)
        
        # Create output directory structure: output_dir/model_name/perturbation_name/
        model_dir_name = sanitize_model_name(model)
        output_dir = output_base_path / model_dir_name / perturbation_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output filename: original_name_seed.csv
        output_filename = f"{csv_file.stem}_seed{seed}.csv"
        output_file = output_dir / output_filename
        
        print(f"\n[{job_count}/{total_jobs}] Model: {model}")
        print(f"         Seed: {seed}")
        print(f"         Input: {csv_file.name}")
        print(f"         Output: {output_file}")
        
        if output_file.exists():
            print(f"         ⚠️  Output file already exists, skipping...")
            continue
        
        # Build command - use Python from current environment
        python_exec = sys.executable
        # If we're in a conda environment, try to use that Python
        if 'CONDA_PREFIX' in os.environ:
            conda_python = os.path.join(os.environ['CONDA_PREFIX'], 'bin', 'python')
            if os.path.exists(conda_python):
                python_exec = conda_python
        
        cmd = [
            python_exec,
            'clinical_decision_direct_models.py',
            '--input-file', str(csv_file),
            '--output-file', str(output_file),
            '--model', model,
            '--seed', str(seed),
            '--device', str(device)
        ]
        
        if max_rows is not None:
            cmd.extend(['--max-rows', str(max_rows)])
        
        if dry_run:
            print(f"         Command: {' '.join(cmd)}")
        else:
            try:
                print(f"         Running...")
                result = subprocess.run(
                    cmd,
                    check=True
                )
                print(f"         ✅ Completed successfully")
            except subprocess.CalledProcessError as e:
                print(f"         ❌ Failed with exit code {e.returncode}")
                # Continue with next job even if this one fails
                continue
            except KeyboardInterrupt:
                print(f"\n\n⚠️  Interrupted by user. Processed {job_count}/{total_jobs} jobs.")
                sys.exit(1)
    
    print("\n" + "=" * 80)
    print(f"SWEEP COMPLETE: {job_count}/{total_jobs} jobs processed")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Run clinical decision pipeline sweep for direct models (DeepSeek/MedGemma)'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        help='List of model names (e.g., deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)'
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=[42, 123, 456],
        help='List of seeds (default: 42 123 456)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='cleaned_data/data_for_llm',
        help='Directory containing input CSV files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='cleaned_data/clinical_decisions',
        help='Base output directory'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='GPU device ID (default: 0)'
    )
    parser.add_argument(
        '--max-rows',
        type=int,
        default=None,
        help='Maximum rows to process per file (for testing)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing'
    )
    
    args = parser.parse_args()
    
    run_sweep(
        models=args.models,
        seeds=args.seeds,
        data_dir=args.data_dir,
        output_base_dir=args.output_dir,
        device=args.device,
        max_rows=args.max_rows,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
