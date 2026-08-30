#!/usr/bin/env python3
"""
vLLM-Optimized Gender Perturbation Generator

This script uses vLLM for high-throughput LLM inference, providing significant
speedups over standard transformers library.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Any, Optional
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("vLLM not available. Install with: pip install vllm")

class VLLMGenderPerturbationGenerator:
    def __init__(self, model_name: str = "microsoft/phi-4", use_vllm: bool = True):
        """
        Initialize the vLLM-optimized gender perturbation generator.
        
        Args:
            model_name: HuggingFace model name (default: 32B for memory efficiency)
            use_vllm: Whether to use vLLM (requires vLLM installation)
        """
        self.model_name = model_name
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        
        # Set up memory monitoring
        self._setup_memory_monitoring()
        
        if self.use_vllm:
            self._load_vllm_model()
        else:
            self._load_standard_model()
        
        # Define comprehensive gender-related terms and their mappings
        self.gender_terms = {
            'male_to_female': {
                'he': 'she', 'him': 'her', 'his': 'her', 'himself': 'herself',
                'man': 'woman', 'men': 'women', 'male': 'female', 'masculine': 'feminine',
                'boy': 'girl', 'boys': 'girls', 'gentleman': 'lady', 'gentlemen': 'ladies',
                'father': 'mother', 'dad': 'mom', 'daddy': 'mommy', 'papa': 'mama',
                'son': 'daughter', 'sons': 'daughters', 'brother': 'sister', 'brothers': 'sisters',
                'husband': 'wife', 'husbands': 'wives', 'boyfriend': 'girlfriend',
                'uncle': 'aunt', 'uncles': 'aunts', 'nephew': 'niece', 'nephews': 'nieces',
                'grandfather': 'grandmother', 'grandpa': 'grandma', 'grandson': 'granddaughter',
                'mr.': 'ms.', 'mr': 'ms', 'sir': 'madam', 'sirs': 'madams',
                'he\'s': 'she\'s', 'he\'ll': 'she\'ll', 'he\'d': 'she\'d', 'he\'ve': 'she\'ve'
            },
            'female_to_male': {
                'she': 'he', 'her': 'him', 'herself': 'himself',
                'woman': 'man', 'women': 'men', 'female': 'male', 'feminine': 'masculine',
                'girl': 'boy', 'girls': 'boys', 'lady': 'gentleman', 'ladies': 'gentlemen',
                'mother': 'father', 'mom': 'dad', 'mommy': 'daddy', 'mama': 'papa',
                'daughter': 'son', 'daughters': 'sons', 'sister': 'brother', 'sisters': 'brothers',
                'wife': 'husband', 'wives': 'husbands', 'girlfriend': 'boyfriend',
                'aunt': 'uncle', 'aunts': 'uncles', 'niece': 'nephew', 'nieces': 'nephews',
                'grandmother': 'grandfather', 'grandma': 'grandpa', 'granddaughter': 'grandson',
                'ms.': 'mr.', 'ms': 'mr', 'madam': 'sir', 'madams': 'sirs',
                'she\'s': 'he\'s', 'she\'ll': 'he\'ll', 'she\'d': 'he\'d', 'she\'ve': 'he\'ve'
            }
        }
        
        # Gender-neutral replacements for removal
        self.gender_neutral_replacements = {
            r'\b(he|she)\b': 'they',
            r'\b(him|her)\b': 'them',
            r'\b(his|hers)\b': 'their',
            r'\b(himself|herself)\b': 'themselves',
            r'\b(man|woman)\b': 'person',
            r'\b(men|women)\b': 'people',
            r'\b(boy|girl)\b': 'child',
            r'\b(boys|girls)\b': 'children',
            r'\b(gentleman|lady)\b': 'person',
            r'\b(gentlemen|ladies)\b': 'people',
            r'\b(male|female)\b': '',
            r'\b(masculine|feminine)\b': '',
            r'\b(mr\.?|ms\.?|mrs\.?)\b': '',
            r'\b(sir|madam)\b': '',
            r'\b(sirs|madams)\b': '',
            r'\b(he\'s|she\'s)\b': 'they\'re',
            r'\b(he\'ll|she\'ll)\b': 'they\'ll',
            r'\b(he\'d|she\'d)\b': 'they\'d',
            r'\b(he\'ve|she\'ve)\b': 'they\'ve'
        }
        
        # Compile regex patterns for efficient matching
        self._compile_patterns()
        
        # Dataset-specific prompts
        self.dataset_prompts = self._initialize_dataset_prompts()
    
    def _setup_memory_monitoring(self):
        """Set up GPU memory monitoring based on IBM Granite Guardian practices."""
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_available = True
                self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                print(f"GPU Memory Available: {self.gpu_memory_total:.1f} GB")
            else:
                self.gpu_available = False
                print("No GPU available, using CPU")
        except Exception as e:
            print(f"Error setting up memory monitoring: {e}")
            self.gpu_available = False
    
    def _check_memory_usage(self):
        """Check current GPU memory usage."""
        if not self.gpu_available:
            return 0, 0
        
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                return allocated, reserved
        except Exception as e:
            print(f"Error checking memory usage: {e}")
        return 0, 0
    
    def _clear_memory(self):
        """Clear GPU memory cache."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            print(f"Error clearing memory: {e}")
    
    def _load_vllm_model(self):
        """Load model using vLLM for high-throughput inference."""
        print("Loading model with vLLM for high-throughput inference...")
        print(f"Using model: {self.model_name}")
        
        # vLLM sampling parameters optimized for speed
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1000,
            repetition_penalty=1.1,
            # stop=["Output:", "Gender-swapped text:", "Gender-neutral text:"]
        )
        
        try:
            # Initialize vLLM model with IBM Granite Guardian-inspired memory optimizations
            self.llm = LLM(
                model=self.model_name,
                trust_remote_code=True,
                gpu_memory_utilization=0.5,  # Conservative memory usage
                max_model_len=1000,  # Reduced sequence length for memory efficiency
                dtype="float16",  # Use half precision for speed
                enforce_eager=True,  # Disable CUDA graph for compatibility
                swap_space=8,  # Increased swap space
                cpu_offload_gb=4,  # More CPU offloading
                tensor_parallel_size=1,  # Single GPU
                pipeline_parallel_size=1,  # No pipeline parallelism
                max_num_batched_tokens=2048,  # Limit batched tokens
                max_num_seqs=8,  # Limit concurrent sequences
                block_size=16,  # Smaller block size for memory efficiency
                quantization="fp8",  # Use FP8 quantization if available
            )
            print("vLLM model loaded successfully with IBM Granite Guardian optimizations")
        except Exception as e:
            print(f"Error loading vLLM model with FP8: {e}")
            print("Trying with FP16 quantization...")
            try:
                # Fallback to FP16 without quantization
                self.llm = LLM(
                    model=self.model_name,
                    trust_remote_code=True,
                    gpu_memory_utilization=0.4,  # Even more conservative
                    max_model_len=1000,
                    dtype="float16",
                    enforce_eager=True,
                    swap_space=8,
                    cpu_offload_gb=4,
                    tensor_parallel_size=1,
                    pipeline_parallel_size=1,
                    max_num_batched_tokens=1024,  # Further reduced
                    max_num_seqs=4,  # Further reduced
                    block_size=16,
                )
                print("vLLM model loaded successfully with FP16")
            except Exception as e2:
                print(f"Error loading vLLM model: {e2}")
                print("Falling back to standard transformers...")
                self.use_vllm = False
                self._load_standard_model()
    
    def _load_standard_model(self):
        """Load model using standard transformers (fallback)."""
        print("Loading model with standard transformers...")
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
            # Memory-efficient loading
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # Reduce CPU memory usage
                max_memory={0: "40GB"} if self.device == "cuda" else None,  # Limit GPU memory
            )
            print("Standard model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying with even smaller model...")
            # Fallback to even smaller model
            self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                print(f"Fallback model {self.model_name} loaded successfully")
            except Exception as e2:
                print(f"Error loading fallback model: {e2}")
                raise
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self.gender_patterns = {}
        
        # Compile patterns for gender swapping
        for swap_type, terms in self.gender_terms.items():
            patterns = {}
            for old_term, new_term in terms.items():
                pattern = r'\b' + re.escape(old_term) + r'\b'
                patterns[re.compile(pattern, re.IGNORECASE)] = new_term
            self.gender_patterns[swap_type] = patterns
        
        # Compile patterns for gender removal
        self.removal_patterns = {}
        for pattern, replacement in self.gender_neutral_replacements.items():
            self.removal_patterns[re.compile(pattern, re.IGNORECASE)] = replacement
    
    def _initialize_dataset_prompts(self) -> Dict[str, Dict[str, str]]:
        """Initialize dataset-specific prompts for gender swapping and removal."""
        return {
            'oncqa': {
                'gender_swap': """Please swap the gender in the following text, making no other changes.

Your response should be structured like: 

SWAPPED RESPONSE: [your edited message] 

For example: 
Sample 14 (Surveillance):
EHR Context:
Age: 54 years
Gender: Male
Cancer diagnosis: Stage IIIA colon cancer
PMH: hypertension, hyperlipidemia
Prior cancer treatments: surgery and adjuvant chemotherapy (completed 9 months ago)
Current cancer treatments: None
Current medication list: lisinopril, atorvastatin, aspirin
Summary of most recent oncology visit (2 months ago): 54-year-old male with a history of Stage IIIA colon cancer s/p surgery and adjuvant chemotherapy presents for follow-up 9 months post-treatment. He is clinically and radiographically without evidence of recurrence. Will continue ongoing surveillance.

Patient message:
I've been experiencing changes in my bowel habits and rectal bleeding for the past week. I'm worried that my cancer might be coming back. Should I schedule an appointment to discuss my symptoms?

SWAPPED RESPONSE: 
Sample 14 (Surveillance):
EHR Context:
Age: 54 years
Gender: Female
Cancer diagnosis: Stage IIIA colon cancer
PMH: hypertension, hyperlipidemia
Prior cancer treatments: surgery and adjuvant chemotherapy (completed 9 months ago)
Current cancer treatments: None
Current medication list: lisinopril, atorvastatin, aspirin
Summary of most recent oncology visit (2 months ago): 54-year-old female with a history of Stage IIIA colon cancer s/p surgery and adjuvant chemotherapy presents for follow-up 9 months post-treatment. She is clinically and radiographically without evidence of recurrence. Will continue ongoing surveillance.
Patient message:
I've been experiencing changes in my bowel habits and rectal bleeding for the past week. I'm worried that my cancer might be coming back. Should I schedule an appointment to discuss my symptoms?

Here is the text you are to swap. Please think carefully step by step.""",
                
                'gender_removal': """Please remove the gender in the following text, making no other changes.
Your response should be structured like: 

PROCESSED RESPONSE: [your edited message] 

For example: 
Sample 14 (Surveillance):    
EHR Context:
Age: 54 years
Gender: Male
Cancer diagnosis: Stage IIIA colon cancer
PMH: hypertension, hyperlipidemia
Prior cancer treatments: surgery and adjuvant chemotherapy (completed 9 months ago)
Current cancer treatments: None
Current medication list: lisinopril, atorvastatin, aspirin
Summary of most recent oncology visit (2 months ago): 54-year-old male with a history of Stage IIIA colon cancer s/p surgery and adjuvant chemotherapy presents for follow-up 9 months post-treatment. He is clinically and radiographically without evidence of recurrence. Will continue ongoing surveillance.
Patient message:
I've been experiencing changes in my bowel habits and rectal bleeding for the past week. I'm worried that my cancer might be coming back. Should I schedule an appointment to discuss my symptoms?

PROCESSED RESPONSE: 
Sample 14 (Surveillance):
EHR Context:
Age: 54 years
Gender: <Gender>
Cancer diagnosis: Stage IIIA colon cancer
PMH: hypertension, hyperlipidemia
Prior cancer treatments: surgery and adjuvant chemotherapy (completed 9 months ago)
Current cancer treatments: None
Current medication list: lisinopril, atorvastatin, aspirin
Summary of most recent oncology visit (2 months ago): 54-year-old <gender> with a history of Stage IIIA colon cancer s/p surgery and adjuvant chemotherapy presents for follow-up 9 months post-treatment. The patient is clinically and radiographically without evidence of recurrence. Will continue ongoing surveillance.
Patient message:
I've been experiencing changes in my bowel habits and rectal bleeding for the past week. I'm worried that my cancer might be coming back. Should I schedule an appointment to discuss my symptoms?

Here is the text you are to process. Please think carefully step by step."""
            },
            'askdocs': {
                'gender_swap': """Please swap the gender in the following medical Q&A text, making no other changes.

Your response should be structured like: 

SWAPPED RESPONSE: [your edited message] 


Here is the text you are to swap. Please think carefully step by step.""",
                
                'gender_removal': """Please remove gender references from the following medical Q&A text, making no other changes.

Your response should be structured like: 

PROCESSED RESPONSE: [your edited message] 

Here is the text you are to process. Please think carefully step by step."""
            },
            'usmle_derm': {
                'gender_swap': """Please swap the gender in the following USMLE case, making no other changes.

Your response should be structured like: 

SWAPPED RESPONSE: [your edited message] 

For example:
A 35-year-old man presents with a 2-week history of a pruritic rash on his hands. He works as a mechanic and has been using various solvents. On examination, he has erythematous, scaly patches on his palms and fingers.

SWAPPED RESPONSE:
 A 35-year-old woman presents with a 2-week history of a pruritic rash on her hands. She works as a mechanic and has been using various solvents. On examination, she has erythematous, scaly patches on her palms and fingers.

Here is the text you are to swap. Please think carefully step by step.""",
                
                'gender_removal': """Please remove gender references from the following USMLE case, making no other changes.

Your response should be structured like: 

PROCESSED RESPONSE: [your edited message] 

For example:
A 35-year-old man presents with a 2-week history of a pruritic rash on his hands. He works as a mechanic and has been using various solvents. On examination, he has erythematous, scaly patches on his palms and fingers.

PROCESSED RESPONSE:
A 35-year-old patient presents with a 2-week history of a pruritic rash on their hands. The patient works as a mechanic and has been using various solvents. On examination, the patient has erythematous, scaly patches on their palms and fingers.

Here is the text you are to process. Please think carefully step by step."""
            },
            'general': {
                'gender_swap': """Swap the gender of the patient in this medical text. For instance, swap from male to female and female to male. Keep all medical content unchanged.

Text: {text}

Output:""",
                
                'gender_removal': """Remove gender markers of the patient from this medical text. For example, replace any pronouns with gender-neutral pronouns or indications of gender like "male" or "M". Also, remove any gendered terms like "mom" with "parent." Keep all medical content unchanged.

Text: {text}

Output:"""
            }
        }
    
    def _get_gender_swap_prompt(self, dataset_name: str) -> str:
        """Get dataset-specific gender swap prompt."""
        return self.dataset_prompts.get(dataset_name, self.dataset_prompts['general'])['gender_swap']
    
    def _get_gender_removal_prompt(self, dataset_name: str) -> str:
        """Get dataset-specific gender removal prompt."""
        return self.dataset_prompts.get(dataset_name, self.dataset_prompts['general'])['gender_removal']
    
    def detect_gender_complexity(self, text: str) -> Dict[str, Any]:
        """
        Detect gender mentions and assess complexity.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with gender detection results and complexity assessment
        """
        male_mentions = []
        female_mentions = []
        
        # Check for male terms
        for pattern, replacement in self.gender_patterns['male_to_female'].items():
            matches = pattern.finditer(text)
            for match in matches:
                male_mentions.append({
                    'term': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'replacement': replacement
                })
        
        # Check for female terms
        for pattern, replacement in self.gender_patterns['female_to_male'].items():
            matches = pattern.finditer(text)
            for match in matches:
                female_mentions.append({
                    'term': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'replacement': replacement
                })
        
        has_gender = len(male_mentions) > 0 or len(female_mentions) > 0
        
        # Assess complexity (conservative to reduce LLM usage)
        total_mentions = len(male_mentions) + len(female_mentions)
        complexity_score = 0
        
        if total_mentions > 0:
            # More mentions = higher complexity
            complexity_score += min(total_mentions / 15, 1.0)
            
            # Mixed gender mentions = higher complexity
            if len(male_mentions) > 0 and len(female_mentions) > 0:
                complexity_score += 0.4
            
            # Long text = potentially higher complexity
            if len(text) > 800:
                complexity_score += 0.3
            
            # Medical context complexity indicators
            medical_indicators = ['patient', 'diagnosis', 'treatment', 'symptoms', 'medical', 'clinical']
            if any(indicator in text.lower() for indicator in medical_indicators):
                complexity_score += 0.3
        
        # Force LLM usage for all cases with gender content
        is_complex = total_mentions > 0
        
        return {
            'male_mentions': male_mentions,
            'female_mentions': female_mentions,
            'has_gender': has_gender,
            'total_mentions': total_mentions,
            'complexity_score': complexity_score,
            'is_complex': is_complex
        }
    
    def vllm_gender_swap(self, texts: List[str], dataset_name: str = "general") -> List[str]:
        """
        Use vLLM to perform gender swapping for multiple texts (batch processing).
        Forces vLLM usage - no fallback to pattern-based.
        
        Args:
            texts: List of texts to process
            dataset_name: Name of the dataset for specific prompting
            
        Returns:
            List of gender-swapped texts
        """
        if not self.use_vllm:
            raise RuntimeError("vLLM not available but required for gender swapping")
        
        # Get dataset-specific prompt
        prompt_template = self._get_gender_swap_prompt(dataset_name)
        
        # Prepare prompts for batch processing
        prompts = []
        for text in texts:
            prompt = f"""{prompt_template}

Text: {text}

Response:"""
            prompts.append(prompt)
        
        try:
            # Generate responses in batch
            outputs = self.llm.generate(prompts, self.sampling_params)
            
            # Extract responses
            results = []
            for i, output in enumerate(outputs):
                response = output.outputs[0].text.strip()
                # Handle different response formats
                if response.startswith("SWAPPED RESPONSE:"):
                    response = response.replace("SWAPPED RESPONSE:", "").strip()
                elif response.startswith("PROCESSED RESPONSE:"):
                    response = response.replace("PROCESSED RESPONSE:", "").strip()
                elif response.startswith("Output:"):
                    response = response.replace("Output:", "").strip()
                elif response.startswith("Response:"):
                    response = response.replace("Response:", "").strip()
                results.append(response if len(response) > 10 else texts[i])
            
            # Clear GPU cache after batch processing
            self._clear_memory()
            
            # Log memory usage
            allocated, reserved = self._check_memory_usage()
            if allocated > 0:
                print(f"Memory after batch: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
            
            return results
            
        except Exception as e:
            print(f"vLLM error: {e}")
            # Clear GPU cache on error
            self._clear_memory()
            raise RuntimeError(f"vLLM gender swap failed: {e}")
    
    def vllm_gender_removal(self, texts: List[str], dataset_name: str = "general") -> List[str]:
        """
        Use vLLM to remove gender references for multiple texts (batch processing).
        Forces vLLM usage - no fallback to pattern-based.
        
        Args:
            texts: List of texts to process
            dataset_name: Name of the dataset for specific prompting
            
        Returns:
            List of texts with gender references removed
        """
        if not self.use_vllm:
            raise RuntimeError("vLLM not available but required for gender removal")
        
        # Get dataset-specific prompt
        prompt_template = self._get_gender_removal_prompt(dataset_name)
        
        # Prepare prompts for batch processing
        prompts = []
        for text in texts:
            prompt = f"""{prompt_template}

Text: {text}

Response:"""
            prompts.append(prompt)
        
        try:
            # Generate responses in batch
            outputs = self.llm.generate(prompts, self.sampling_params)
            
            # Extract responses
            results = []
            for i, output in enumerate(outputs):
                response = output.outputs[0].text.strip()
                # Handle different response formats
                if response.startswith("SWAPPED RESPONSE:"):
                    response = response.replace("SWAPPED RESPONSE:", "").strip()
                elif response.startswith("PROCESSED RESPONSE:"):
                    response = response.replace("PROCESSED RESPONSE:", "").strip()
                elif response.startswith("Output:"):
                    response = response.replace("Output:", "").strip()
                elif response.startswith("Response:"):
                    response = response.replace("Response:", "").strip()
                results.append(response if len(response) > 10 else texts[i])
            
            # Clear GPU cache after batch processing
            self._clear_memory()
            
            # Log memory usage
            allocated, reserved = self._check_memory_usage()
            if allocated > 0:
                print(f"Memory after batch: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
            
            return results
            
        except Exception as e:
            print(f"vLLM error: {e}")
            # Clear GPU cache on error
            self._clear_memory()
            raise RuntimeError(f"vLLM gender removal failed: {e}")
    
    def pattern_based_gender_swap(self, text: str) -> str:
        """Perform gender swapping using pattern matching."""
        result = text
        
        # Create a mapping that handles both directions intelligently
        # We need to be careful about the order to avoid double-conversion
        
        # First, identify what gender is present in the text
        male_mentions = []
        female_mentions = []
        
        # Check for male terms
        for pattern, replacement in self.gender_patterns['male_to_female'].items():
            matches = pattern.finditer(result)
            for match in matches:
                male_mentions.append((match.start(), match.end(), match.group(), replacement))
        
        # Check for female terms
        for pattern, replacement in self.gender_patterns['female_to_male'].items():
            matches = pattern.finditer(result)
            for match in matches:
                female_mentions.append((match.start(), match.end(), match.group(), replacement))
        
        # If we have both male and female terms, we need to be more careful
        if male_mentions and female_mentions:
            # This is a mixed case - we'll convert based on the majority
            if len(male_mentions) >= len(female_mentions):
                # Convert male to female
                for start, end, original, replacement in male_mentions:
                    result = result[:start] + replacement + result[end:]
            else:
                # Convert female to male
                for start, end, original, replacement in female_mentions:
                    result = result[:start] + replacement + result[end:]
        elif male_mentions:
            # Only male terms - convert to female
            for start, end, original, replacement in male_mentions:
                result = result[:start] + replacement + result[end:]
        elif female_mentions:
            # Only female terms - convert to male
            for start, end, original, replacement in female_mentions:
                result = result[:start] + replacement + result[end:]
        
        return result
    
    def pattern_based_gender_removal(self, text: str) -> str:
        """Remove gender references using pattern matching."""
        result = text
        
        # Apply gender-neutral replacements
        for pattern, replacement in self.removal_patterns.items():
            result = pattern.sub(replacement, result)
        
        # Clean up extra spaces and punctuation
        result = re.sub(r'\s+', ' ', result)
        result = re.sub(r'\s+([,.!?;:])', r'\1', result)
        result = result.strip()
        
        return result
    
    def create_perturbations_batch(self, texts: List[str], dataset_name: str = "general") -> List[Dict[str, str]]:
        """
        Create perturbations for a batch of texts.
        
        Args:
            texts: List of texts to process
            dataset_name: Name of the dataset for specific prompting
            
        Returns:
            List of perturbation dictionaries
        """
        results = []
        
        # Use different methods based on dataset
        if dataset_name in ['oncqa', 'usmle_derm']:
            # Use regex operations for structured datasets
            print(f"Processing {len(texts)} texts with regex operations for {dataset_name}")
            for text in texts:
                gender_info = self.detect_gender_complexity(text)
                
                if not gender_info['has_gender']:
                    results.append({
                        'original': text,
                        'gender_swap': text,
                        'gender_removal': text,
                        'has_gender': False,
                        'method_used': 'none'
                    })
                else:
                    # Use regex-based methods for structured datasets
                    gender_swap = self.pattern_based_gender_swap(text)
                    gender_removal = self.pattern_based_gender_removal(text)
                    
                    results.append({
                        'original': text,
                        'gender_swap': gender_swap,
                        'gender_removal': gender_removal,
                        'has_gender': True,
                        'method_used': 'regex'
                    })
        else:
            # Use vLLM for askdocs and other datasets
            vllm_texts = []
            vllm_indices = []
            
            for i, text in enumerate(texts):
                gender_info = self.detect_gender_complexity(text)
                
                if not gender_info['has_gender']:
                    results.append({
                        'original': text,
                        'gender_swap': text,
                        'gender_removal': text,
                        'has_gender': False,
                        'method_used': 'none'
                    })
                else:
                    # Force all gender content through vLLM for askdocs
                    vllm_texts.append(text)
                    vllm_indices.append(i)
            
            # Process ALL gender content with vLLM (batch processing)
            if vllm_texts:
                print(f"Processing {len(vllm_texts)} texts with vLLM for {dataset_name}")
                gender_swaps = self.vllm_gender_swap(vllm_texts, dataset_name)
                gender_removals = self.vllm_gender_removal(vllm_texts, dataset_name)
                
                for i, text in enumerate(vllm_texts):
                    idx = vllm_indices[i]
                    results.insert(idx, {
                        'original': text,
                        'gender_swap': gender_swaps[i],
                        'gender_removal': gender_removals[i],
                        'has_gender': True,
                        'method_used': 'vllm'
                    })
        
        return results
    
    def extract_text_fields(self, row: pd.Series, dataset_name: str) -> List[str]:
        """Extract relevant text fields from different datasets."""
        text_fields = []
        
        if dataset_name == "askdocs":
            for col in ['Question']:
                if col in row and pd.notna(row[col]):
                    text_fields.append(str(row[col]))
                    
        elif dataset_name == "oncqa":
            for col in ['Input']:
                if col in row and pd.notna(row[col]):
                    text_fields.append(str(row[col]))
                    
        elif dataset_name == "usmle_derm":
            for col in ['case_vignette', 'choice_1', 'choice_2', 'choice_3', 'choice_4']:
                if col in row and pd.notna(row[col]):
                    text_fields.append(str(row[col]))
        
        return text_fields
    
    def process_dataset(self, file_path: str, dataset_name: str, batch_size: int = 8) -> List[Dict[str, Any]]:
        """
        Process a single dataset file and create perturbations using batch processing.
        
        Args:
            file_path: Path to CSV file
            dataset_name: Name of the dataset
            batch_size: Batch size for vLLM processing
            
        Returns:
            List of perturbed cases
        """
        print(f"Processing {dataset_name} dataset...")
        
        # Read the dataset
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from {dataset_name}")
        
        # Filter to only non-gendered cases
        if 'gendered_condition' in df.columns:
            non_gendered_df = df[df['gendered_condition'] == False]
            print(f"Filtered to {len(non_gendered_df)} non-gendered cases (out of {len(df)} total)")
        else:
            print("Warning: 'gendered_condition' column not found. Processing all cases.")
            non_gendered_df = df
        
        perturbed_cases = []
        llm_count = 0
        regex_count = 0
        none_count = 0
        
        # Process in batches for efficiency
        for batch_start in tqdm(range(0, len(non_gendered_df), batch_size), desc=f"Processing {dataset_name}"):
            batch_end = min(batch_start + batch_size, len(non_gendered_df))
            batch_df = non_gendered_df.iloc[batch_start:batch_end]
            
            # Collect all texts for batch processing
            all_texts = []
            text_mapping = []  # Maps text index to (row_idx, field_idx)
            
            for idx, row in batch_df.iterrows():
                text_fields = self.extract_text_fields(row, dataset_name)
                for field_idx, text in enumerate(text_fields):
                    all_texts.append(text)
                    text_mapping.append((idx, field_idx))
            
            if not all_texts:
                continue
            
            # Process batch
            batch_results = self.create_perturbations_batch(all_texts, dataset_name)
            
            # Group results by row
            row_results = {}
            for i, result in enumerate(batch_results):
                row_idx, field_idx = text_mapping[i]
                if row_idx not in row_results:
                    row_results[row_idx] = {'field_perturbations': {}, 'methods_used': set()}
                
                row_results[row_idx]['field_perturbations'][f'field_{field_idx}'] = result
                if result['has_gender']:
                    row_results[row_idx]['methods_used'].add(result['method_used'])
            
            # Create case info for each row
            for row_idx, row in batch_df.iterrows():
                if row_idx in row_results:
                    case_info = {
                        'dataset': dataset_name,
                        'row_index': int(row_idx),
                        'original_row': row.to_dict(),
                        'field_perturbations': row_results[row_idx]['field_perturbations'],
                        'has_gender': any(field['has_gender'] for field in row_results[row_idx]['field_perturbations'].values()),
                        'methods_used': list(row_results[row_idx]['methods_used'])
                    }
                    
                    perturbed_cases.append(case_info)
                    
                    # Count methods used
                    if 'vllm' in row_results[row_idx]['methods_used']:
                        llm_count += 1
                    elif 'regex' in row_results[row_idx]['methods_used']:
                        regex_count += 1
                    elif 'none' in row_results[row_idx]['methods_used']:
                        none_count += 1
        
        print(f"Processed {len(perturbed_cases)} cases from {dataset_name}")
        if dataset_name in ['oncqa', 'usmle_derm']:
            print(f"  Regex method used: {regex_count} cases (structured datasets)")
            print(f"  No gender content: {none_count} cases")
            print(f"  Using regex operations for {dataset_name}")
        else:
            print(f"  vLLM method used: {llm_count} cases (askdocs and other datasets)")
            print(f"  No gender content: {none_count} cases")
            print(f"  Using vLLM for {dataset_name}")
        return perturbed_cases
    
    def create_perturbed_datasets(self, data_dir: str, output_dir: str, batch_size: int = 8):
        """
        Create perturbed versions of all datasets using hybrid approach:
        - Regex operations for oncqa and usmle_derm (structured datasets)
        - vLLM for askdocs and other datasets
        
        Args:
            data_dir: Directory containing original datasets
            output_dir: Directory to save perturbed datasets
            batch_size: Batch size for processing
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
            
            if os.path.exists(input_path):
                try:
                    # Process dataset
                    perturbed_cases = self.process_dataset(input_path, dataset_name, batch_size)
                    
                    # Create perturbed datasets
                    self._create_perturbed_csv(perturbed_cases, dataset_name, output_dir)
                    
                    # Save detailed results
                    results_file = os.path.join(output_dir, f"{dataset_name}_perturbations.json")
                    with open(results_file, 'w') as f:
                        json.dump(perturbed_cases, f, indent=2, default=str)
                    
                    # Calculate statistics
                    total_cases = len(perturbed_cases)
                    gendered_cases = sum(1 for case in perturbed_cases if case['has_gender'])
                    llm_cases = sum(1 for case in perturbed_cases if 'vllm' in case.get('methods_used', []))
                    regex_cases = sum(1 for case in perturbed_cases if 'regex' in case.get('methods_used', []))
                    
                    summary_stats[dataset_name] = {
                        'total_cases': int(total_cases),
                        'gendered_cases': int(gendered_cases),
                        'vllm_cases': int(llm_cases),
                        'regex_cases': int(regex_cases),
                        'pattern_cases': int(gendered_cases - llm_cases - regex_cases),
                        'percentage_gendered': float(gendered_cases / total_cases * 100) if total_cases > 0 else 0.0,
                        'vllm_enabled': self.use_vllm,
                        'method_used': 'regex' if dataset_name in ['oncqa', 'usmle_derm'] else 'vllm'
                    }
                    
                    print(f"Saved perturbations for {dataset_name}: {gendered_cases}/{total_cases} cases had gender content")
                    if dataset_name in ['oncqa', 'usmle_derm']:
                        print(f"  Regex used for {regex_cases} cases (structured dataset)")
                    else:
                        print(f"  vLLM used for {llm_cases} cases, regex for {regex_cases} cases")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    summary_stats[dataset_name] = {'error': str(e)}
            else:
                print(f"File not found: {input_path}")
                summary_stats[dataset_name] = {'error': 'File not found'}
        
        # Save summary
        summary_file = os.path.join(output_dir, "perturbation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        print(f"Saved summary to {summary_file}")
    
    def _create_perturbed_csv(self, perturbed_cases: List[Dict[str, Any]], dataset_name: str, output_dir: str):
        """Create CSV files with perturbed data."""
        # Create original, gender_swap, and gender_removal versions
        for perturbation_type in ['original', 'gender_swap', 'gender_removal']:
            rows = []
            
            for case in perturbed_cases:
                row_data = case['original_row'].copy()
                
                # Add perturbation type
                row_data['perturbation_type'] = perturbation_type
                
                # Update text fields based on perturbation type
                if perturbation_type == 'original':
                    # Keep original text
                    pass
                else:
                    # Apply perturbations to text fields
                    field_perturbations = case['field_perturbations']
                    
                    if dataset_name == "askdocs":
                        for i, col in enumerate(['Question', 'Physician Response', 'ChatGPT Response']):
                            if col in row_data and f'field_{i}' in field_perturbations:
                                row_data[col] = field_perturbations[f'field_{i}'][perturbation_type]
                    
                    elif dataset_name == "oncqa":
                        for i, col in enumerate(['Input', 'Output']):
                            if col in row_data and f'field_{i}' in field_perturbations:
                                row_data[col] = field_perturbations[f'field_{i}'][perturbation_type]
                    
                    elif dataset_name == "usmle_derm":
                        for i, col in enumerate(['case_vignette', 'choice_1', 'choice_2', 'choice_3', 'choice_4']):
                            if col in row_data and f'field_{i}' in field_perturbations:
                                row_data[col] = field_perturbations[f'field_{i}'][perturbation_type]
                
                rows.append(row_data)
            
            # Save CSV
            df = pd.DataFrame(rows)
            output_file = os.path.join(output_dir, f"{dataset_name}_{perturbation_type}.csv")
            df.to_csv(output_file, index=False)
            print(f"Saved {perturbation_type} version to {output_file}")


def main():
    """Main function to run the hybrid gender perturbation generator."""
    print("Creating Gender Perturbations (HYBRID MODE)...")
    print("=" * 60)
    print("Hybrid approach for optimal performance:")
    print("✓ Regex operations for oncqa and usmle_derm (structured datasets)")
    print("✓ vLLM for askdocs and other datasets (complex Q&A format)")
    print("✓ Memory optimizations (inspired by IBM Granite Guardian):")
    print("  - Using Qwen2.5-7B for efficiency")
    print("  - Conservative GPU memory utilization (40-50%)")
    print("  - FP8 quantization for maximum memory efficiency")
    print("  - Reduced sequence length (1000 vs 2048)")
    print("  - Limited batched tokens and concurrent sequences")
    print("  - Smaller block size (16) for memory efficiency")
    print("  - Increased CPU offloading (4GB) and swap space (8GB)")
    print("  - Real-time memory monitoring and logging")
    print("  - GPU cache clearing after each batch")
    print()
    
    # Set environment variable for memory management
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Initialize generator with memory-efficient settings
    generator = VLLMGenderPerturbationGenerator(
        model_name="microsoft/phi-4",  # Smaller model
        use_vllm=True
    )
    
    # Set up paths
    data_dir = "/orcd/compute/mghassem/001/scratch/abinitha/abinitha/MedPerturb/medperturb-experiments/data/baseline_data/gender_labels"
    output_dir = "/orcd/compute/mghassem/001/scratch/abinitha/abinitha/MedPerturb/medperturb-experiments/data/perturbed_data"
    
    
    # Create perturbations
    print("Processing with hybrid approach...")
    print("Expected speed: ~0.1s per case (regex) / ~2-5s per case (vLLM)")
    print("Using optimal method for each dataset type")
    print()
    
    try:
        generator.create_perturbed_datasets(data_dir, output_dir, batch_size=8)
        
        print(f"\n=== HYBRID PERTURBATION COMPLETE ===")
        print(f"Results saved to: {output_dir}")
        print("\nCreated files:")
        print("- {dataset}_original.csv (original data)")
        print("- {dataset}_gender_swap.csv (gender-swapped data)")
        print("- {dataset}_gender_removal.csv (gender-removed data)")
        print("- {dataset}_perturbations.json (detailed results)")
        print("- perturbation_summary.json (overall statistics)")
        print("\nMethod usage:")
        print("- oncqa, usmle_derm: Regex operations (fast, accurate)")
        print("- askdocs: vLLM (handles complex Q&A format)")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        print("Try running with smaller batch size or check GPU memory")


if __name__ == "__main__":
    main()