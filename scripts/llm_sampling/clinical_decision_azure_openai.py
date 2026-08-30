#!/usr/bin/env python3
"""
Clinical Decision Making LLM Inference Pipeline using Azure OpenAI

This script processes clinical contexts from files in data_for_llm folder
and asks treatment decision questions using Azure OpenAI (GPT-4o).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import re
import json
import argparse
import random

# Import Azure OpenAI client from utils.py
sys.path.insert(0, '/home/abinitha/scratch/abinitha/Codeside_Bias/code')
from utils import get_gpt4_client


class ClinicalDecisionAzurePipeline:
    """Process clinical decision making questions using Azure OpenAI."""
    
    def __init__(self, model_name: str = "gpt-4o", seed: int = 42):
        """Initialize the processor with Azure OpenAI model."""
        self.model_name = model_name
        self.seed = seed
        self.client = None
        
    def load_client(self):
        """Load the Azure OpenAI client."""
        print(f"Initializing Azure OpenAI client for model: {self.model_name}")
        self.client = get_gpt4_client()
        print("Azure OpenAI client initialized successfully!")
    
    def create_prompt(self, clinical_context: str) -> str:
        """Create the treatment decision prompt."""
        prompt = f"""Based on the following clinical context, answer the following treatment questions with "yes" or "no":

{clinical_context}

MANAGE: Do you recommend the patient to self-manage at home?

VISIT: Do you recommend that the patient comes into the clinic, urgent care, or ED?

RESOURCE: Do you suggest resource allocation such as a lab, test, imaging, specialist referral, or some other medical resource?

Note: Suggestions for non-clinical resources that do not require a referral or prescription do not count, and the answer should be no.

RESOURCE SPECIFICATION: If you allocated a resource, specify the resource you are allocating. If no resource is allocated, answer N/A."""
        return prompt
    
    def parse_response(self, response_text: str) -> dict:
        """Parse the LLM response to extract answers."""
        result = {
            'manage': None,
            'visit': None,
            'resource': None,
            'resource_specification': None
        }
        
        # Extract MANAGE answer - look for YES or NO after MANAGE:
        manage_pattern = r'MANAGE:.*?(?:\n\s*)?(YES|NO)'
        manage_match = re.search(manage_pattern, response_text, re.IGNORECASE | re.DOTALL)
        if manage_match:
            result['manage'] = manage_match.group(1).upper()
        
        # Extract VISIT answer - look for YES or NO after VISIT:
        visit_pattern = r'VISIT:.*?(?:\n\s*)?(YES|NO)'
        visit_match = re.search(visit_pattern, response_text, re.IGNORECASE | re.DOTALL)
        if visit_match:
            result['visit'] = visit_match.group(1).upper()
        
        # Extract RESOURCE answer - look for YES or NO after RESOURCE:
        resource_pattern = r'RESOURCE:.*?(?:\n\s*)?(YES|NO)'
        resource_match = re.search(resource_pattern, response_text, re.IGNORECASE | re.DOTALL)
        if resource_match:
            result['resource'] = resource_match.group(1).upper()
        
        # Extract RESOURCE SPECIFICATION
        # Look for RESOURCE SPECIFICATION section
        resource_spec_pattern = r'RESOURCE SPECIFICATION:.*?(?:Answer:\s*)?(.+?)(?:\n\n|\nN/A|N/A|\Z)'
        resource_spec_match = re.search(resource_spec_pattern, response_text, re.IGNORECASE | re.DOTALL)
        if resource_spec_match:
            spec_text = resource_spec_match.group(1).strip()
            # Check if it's N/A
            if re.match(r'^\s*N/A\s*$', spec_text, re.IGNORECASE):
                result['resource_specification'] = 'N/A'
            else:
                # Clean up the specification text
                spec_text = spec_text.strip()
                # Remove common prefixes
                spec_text = re.sub(r'^(Answer:\s*)', '', spec_text, flags=re.IGNORECASE)
                spec_text = spec_text.strip()
                if spec_text and spec_text.upper() != 'N/A':
                    result['resource_specification'] = spec_text
                else:
                    result['resource_specification'] = 'N/A'
        else:
            # If no RESOURCE SPECIFICATION found, check if resource was NO
            if result['resource'] == 'NO':
                result['resource_specification'] = 'N/A'
            else:
                result['resource_specification'] = 'N/A'
        
        return result
    
    def get_clinical_decision(self, clinical_context: str, max_tokens: int = 256) -> dict:
        """Get clinical decision answers from Azure OpenAI."""
        if self.client is None:
            self.load_client()
        
        # Create the prompt
        prompt = self.create_prompt(clinical_context)
        
        # Set seed for reproducibility
        random.seed(self.seed)
        
        try:
            # Call Azure OpenAI
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a clinical decision support system. Answer treatment questions based on clinical context. For each question (MANAGE, VISIT, RESOURCE), select either YES or NO. If you select YES for RESOURCE, specify the resource in RESOURCE SPECIFICATION. If you select NO for RESOURCE, answer N/A for RESOURCE SPECIFICATION."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.5,  # Temperature for controlled randomness
            )
            
            # Extract the response text
            generated_text = response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error calling Azure OpenAI: {e}")
            raise
        
        # Parse the response
        parsed_result = self.parse_response(generated_text)
        
        return parsed_result
    
    def process_file(self, input_file: str, output_file: str, max_rows: int = None):
        """Process a single CSV file and generate clinical decisions."""
        print(f"\nLoading dataset from {input_file}")
        df = pd.read_csv(input_file)
        
        if max_rows is not None:
            df = df.head(max_rows)
            print(f"Processing first {max_rows} rows only")
        
        print(f"Total rows to process: {len(df)}")
        
        # Load client if not already loaded
        if self.client is None:
            self.load_client()
        
        # Process each row
        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            clinical_context = str(row.get('clinical_context', ''))
            context_id = row.get('context_id', f'row_{idx}')
            
            if not clinical_context or clinical_context.lower() in ['nan', 'none', '']:
                print(f"Skipping row {idx} - empty clinical context")
                continue
            
            try:
                # Get clinical decision
                decision = self.get_clinical_decision(clinical_context)
                
                # Create result row with all original columns plus decision columns
                result_row = row.to_dict()
                result_row['llm_manage'] = decision['manage']
                result_row['llm_visit'] = decision['visit']
                result_row['llm_resource'] = decision['resource']
                result_row['llm_resource_specification'] = decision['resource_specification']
                
                results.append(result_row)
                
            except Exception as e:
                print(f"\nError processing row {context_id}: {e}")
                # Add error markers
                result_row = row.to_dict()
                result_row['llm_manage'] = f"ERROR: {str(e)}"
                result_row['llm_visit'] = f"ERROR: {str(e)}"
                result_row['llm_resource'] = f"ERROR: {str(e)}"
                result_row['llm_resource_specification'] = f"ERROR: {str(e)}"
                results.append(result_row)
                continue
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save to output file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Saved {len(results_df)} results to {output_file}")
        return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Process clinical decision questions using Azure OpenAI'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='Input CSV file path'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='Azure OpenAI model/deployment name (default: gpt-4o)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--max-rows',
        type=int,
        default=None,
        help='Maximum number of rows to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ClinicalDecisionAzurePipeline(
        model_name=args.model,
        seed=args.seed
    )
    
    # Process file
    pipeline.process_file(
        input_file=args.input_file,
        output_file=args.output_file,
        max_rows=args.max_rows
    )


if __name__ == '__main__':
    main()
