#!/usr/bin/env python3
"""
Clinical Decision Making Pipeline - Minimal version based on working test script
"""

import csv
import sys
import re
import os
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Disable PyTorch compilation to avoid shared memory issues
os.environ['TORCH_COMPILE_DEBUG'] = '0'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
try:
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
except:
    pass


def parse_response(response_text: str) -> dict:
    """Parse the LLM response to extract answers."""
    result = {
        'manage': None,
        'visit': None,
        'resource': None,
        'resource_specification': None
    }
    
    # Extract MANAGE answer
    manage_pattern = r'MANAGE:.*?(?:\n\s*)?(YES|NO)'
    manage_match = re.search(manage_pattern, response_text, re.IGNORECASE | re.DOTALL)
    if manage_match:
        result['manage'] = manage_match.group(1).upper()
    
    # Extract VISIT answer
    visit_pattern = r'VISIT:.*?(?:\n\s*)?(YES|NO)'
    visit_match = re.search(visit_pattern, response_text, re.IGNORECASE | re.DOTALL)
    if visit_match:
        result['visit'] = visit_match.group(1).upper()
    
    # Extract RESOURCE answer
    resource_pattern = r'RESOURCE:.*?(?:\n\s*)?(YES|NO)'
    resource_match = re.search(resource_pattern, response_text, re.IGNORECASE | re.DOTALL)
    if resource_match:
        result['resource'] = resource_match.group(1).upper()
    
    # Extract RESOURCE SPECIFICATION
    resource_spec_pattern = r'RESOURCE SPECIFICATION:.*?(?:Answer:\s*)?(.+?)(?:\n\n|\nN/A|N/A|\Z)'
    resource_spec_match = re.search(resource_spec_pattern, response_text, re.IGNORECASE | re.DOTALL)
    if resource_spec_match:
        spec_text = resource_spec_match.group(1).strip()
        if re.match(r'^\s*N/A\s*$', spec_text, re.IGNORECASE):
            result['resource_specification'] = 'N/A'
        else:
            spec_text = re.sub(r'^(Answer:\s*)', '', spec_text, flags=re.IGNORECASE).strip()
            result['resource_specification'] = spec_text if spec_text and spec_text.upper() != 'N/A' else 'N/A'
    else:
        result['resource_specification'] = 'N/A'
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    model_name = args.model
    is_deepseek = "deepseek" in model_name.lower()
    is_medgemma = "medgemma" in model_name.lower()
    
    # Set seed
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    print("=" * 60)
    print(f"Loading model: {model_name}")
    print("=" * 60)
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    print("✓ Tokenizer loaded")
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("✓ Pad token set")
    
    # Load model
    print("\n2. Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("✓ Model loaded")
    
    # Read CSV
    print("\n3. Reading input CSV...")
    rows = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    if args.max_rows is not None:
        rows = rows[:args.max_rows]
        print(f"Processing first {args.max_rows} rows only")
    
    print(f"Total rows to process: {len(rows)}")
    
    # Process rows
    print("\n4. Processing rows...")
    results = []
    
    for idx, row in tqdm(enumerate(rows), total=len(rows), desc="Processing"):
        clinical_context = str(row.get('clinical_context', ''))
        
        if not clinical_context or clinical_context.lower() in ['nan', 'none', '']:
            continue
        
        # Create prompt
        if is_deepseek:
            prompt = f"""You are a clinical decision support system. Answer treatment questions based on clinical context. For each question (MANAGE, VISIT, RESOURCE), select either YES or NO. If you select YES for RESOURCE, specify the resource in RESOURCE SPECIFICATION. If you select NO for RESOURCE, answer N/A for RESOURCE SPECIFICATION.

Based on the following clinical context, answer the following treatment questions with "yes" or "no":

{clinical_context}

MANAGE: Do you recommend the patient to self-manage at home?

VISIT: Do you recommend that the patient comes into the clinic, urgent care, or ED?

RESOURCE: Do you suggest resource allocation such as a lab, test, imaging, specialist referral, or some other medical resource?

Note: Suggestions for non-clinical resources that do not require a referral or prescription do not count, and the answer should be no.

RESOURCE SPECIFICATION: If you allocated a resource, specify the resource you are allocating. If no resource is allocated, answer N/A."""
            messages = [{"role": "user", "content": prompt}]
        else:
            prompt = f"""Based on the following clinical context, answer the following treatment questions with "yes" or "no":

{clinical_context}

MANAGE: Do you recommend the patient to self-manage at home?

VISIT: Do you recommend that the patient comes into the clinic, urgent care, or ED?

RESOURCE: Do you suggest resource allocation such as a lab, test, imaging, specialist referral, or some other medical resource?

Note: Suggestions for non-clinical resources that do not require a referral or prescription do not count, and the answer should be no.

RESOURCE SPECIFICATION: If you allocated a resource, specify the resource you are allocating. If no resource is allocated, answer N/A."""
            messages = [
                {"role": "system", "content": "You are a clinical decision support system. Answer treatment questions based on clinical context. For each question (MANAGE, VISIT, RESOURCE), select either YES or NO. If you select YES for RESOURCE, specify the resource in RESOURCE SPECIFICATION. If you select NO for RESOURCE, answer N/A for RESOURCE SPECIFICATION."},
                {"role": "user", "content": prompt}
            ]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        model_inputs = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move to device
        device = next(model.parameters()).device
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        
        # Generate
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        if is_deepseek:
            generate_kwargs = {
                "max_new_tokens": 256,
                "do_sample": False,
                "repetition_penalty": 1.1,
                "pad_token_id": pad_token_id,
            }
        else:
            generate_kwargs = {
                "max_new_tokens": 256,
                "temperature": 0.1,
                "do_sample": True,
                "repetition_penalty": 1.1,
                "pad_token_id": pad_token_id,
            }
        
        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    model_inputs["input_ids"],
                    attention_mask=model_inputs.get("attention_mask", None),
                    **generate_kwargs
                )
            
            # Extract only generated part
            input_length = model_inputs["input_ids"].shape[1]
            generated_ids = generated_ids[:, input_length:]
            
            # Decode
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            # Parse response
            decision = parse_response(generated_text)
            
            # Add to results
            result_row = row.copy()
            result_row['llm_manage'] = decision['manage']
            result_row['llm_visit'] = decision['visit']
            result_row['llm_resource'] = decision['resource']
            result_row['llm_resource_specification'] = decision['resource_specification']
            results.append(result_row)
            
            # Clear cache periodically
            if (idx + 1) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\nError processing row {idx}: {e}")
            result_row = row.copy()
            result_row['llm_manage'] = f"ERROR: {e}"
            result_row['llm_visit'] = f"ERROR: {e}"
            result_row['llm_resource'] = f"ERROR: {e}"
            result_row['llm_resource_specification'] = f"ERROR: {e}"
            results.append(result_row)
    
    # Write output
    if results:
        fieldnames = list(results[0].keys())
        with open(args.output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\n✅ Saved {len(results)} results to {args.output_file}")


if __name__ == "__main__":
    main()
