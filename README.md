This document describes the `scripts/` directory and the main dataset file `dataset.csv`.

## Dataset: `dataset.csv`

**6,233 rows × 71 columns.** Each row is one clinical vignette under one perturbation condition. A single vignette therefore appears up to five times (once per condition), linked by `context_id`.

### Source datasets

| `dataset` value | Description | Perturbations available |
|---|---|---|
| `askdocs` | Real patient posts from r/AskDocs | All five |
| `oncqa` | Oncology Q&A vignettes | All five |
| `sct` | Script Concordance Test cases | Baseline + gender only |
| `usmle_derm` | USMLE dermatology vignettes | Baseline + gender only |

### Perturbation conditions (`perturbation` column)

| Value | Description |
|---|---|
| `baseline` | Original, unmodified clinical context |
| `gender_swap` | Patient gender swapped (M→F or F→M) |
| `gender_removal` | Gender markers removed; gender-neutral language |
| `colorful` | Dramatic/emotionally charged language added (AskDocs + OncQA only) |
| `uncertain` | Anxious hedging language added (AskDocs + OncQA only) |

### Column groups

**Identifiers and metadata** (8 columns)

| Column | Type | Description |
|---|---|---|
| `context_id` | string | Unique vignette ID, format `{dataset}_{n}` (e.g. `askdocs_42`) |
| `perturbation` | string | Perturbation condition (see above) |
| `dataset` | string | Source dataset |
| `clinical_context` | string | Full text of the clinical vignette under this condition |
| `original_gender` | string | Patient gender in the original text: `M`, `F`, or `Unknown` |
| `age` | numeric | Patient age (where available) |
| `gendered_condition` | bool | Whether the case involves a gender-specific medical condition |
| `provided_physician_response` | string | Original physician/expert response (where available) |

**Gold-standard labels** (3 columns) — `YES` / `NO` strings

| Column | Task description |
|---|---|
| `gold_manage` | Should the patient self-manage vs. seek clinical management? (`YES` = clinical management) |
| `gold_visit` | Should the patient visit a clinician? (`YES` = visit recommended) |
| `gold_resource` | Should tests or referrals be ordered? (`YES` = resources recommended) |

**Clinician rater annotations** (15 columns) — `0.0` / `1.0` floats (NaN if not rated)

Five raters × three tasks: `clin_rater{1–5}_{manage,visit,resource}`

A value of `1.0` corresponds to YES (recommend clinical action); `0.0` to NO.

**LLM responses** (45 columns) — `YES` / `NO` strings (NaN if not run)

Five models × three seeds × three tasks: `{model}_seed{0–2}_{manage,visit,resource}`

| Model prefix | Model |
|---|---|
| `gpt4o` | GPT-4o |
| `deepseek` | DeepSeek |
| `medgemma` | MedGemma |
| `llama` | Llama-3 |
| `qwen` | Qwen |

Majority vote across three seeds (mean ≥ 0.5 → YES) gives the per-model prediction used in all analyses.

---

## Scripts: `scripts/`

The scripts implement the full  data pipeline in four stages. Run them in order: **data preparation → gender classification → perturbation → LLM sampling**.

```
scripts/
├── data_preparation/
│   ├── clean_data.py
│   ├── create_gold_standard_baseline.py
│   ├── _format.py
│   ├── process_qpain_data.py
│   └── subsample.py
├── gender_classification/
│   ├── classify_gendered_cases.py
│   ├── gender_medical_classifier.py
│   └── vllm_gender_perturbation.py
├── perturbation/
│   ├── prompts.py
│   └── run_perturbation.py
└── llm_sampling/
    ├── clinical_decision_azure_openai.py
    ├── clinical_decision_direct_models.py
    ├── run_azure_openai_sweep.py
    └── run_direct_models_sweep.py
```

---

### Stage 1: `data_preparation/`

Scripts that ingest raw source datasets, clean them, and produce the standardised  format.

#### `_format.py`
Converts each raw source dataset (AskDocs, OncQA, SCT, USMLE-Derm) from its original format into the shared  CSV schema. Output is `cleaned_data/_format.csv`.

#### `clean_data.py`
Multi-mode cleaning pipeline. Removes error rows, deduplicates content, and optionally runs an LLM-based cleanup pass (Llama-3.3-70B-Instruct via HuggingFace) to standardise formatting. Supports in-place editing or writing to a separate output file.

#### `create_gold_standard_baseline.py`
Builds `gold_standard_baseline.csv` from the baseline split by replacing raw LLM label columns (`llm_manage`, etc.) with the curated gold-standard columns (`gold_standard_manage`, etc.).

#### `subsample.py`
Produces balanced sub-samples for annotation or piloting. Two modes:
- `subsample_baseline`: draws stratified samples from the baseline split.
- `subsample_condition`: draws from a specific perturbation condition, outputting `context_id` + `clinical_context` for annotation.

Key args: `--input`, `--output`, `--seed`, `--samples-per-dataset`.

#### `process_qpain_data.py`
Specialised preprocessing for QPain data files (pain-case CSVs). Standardises fields to match the  schema before passing to `_format.py`.

---

### Stage 2: `gender_classification/`

Scripts that detect whether a case involves gender-specific clinical content and apply gender-based rewrites.

#### `gender_medical_classifier.py`
Core rule-based + pattern-matching classifier. Scans `clinical_context` and `provided_physician_response` for gender-relevant terms (anatomical, pronoun, condition-specific). Computes a confidence score and assigns a gender-relevance label. Used as a library by the other scripts in this stage.

#### `classify_gendered_cases.py`
Runs `gender_medical_classifier.py` across all source datasets and writes a `gendered_condition` column back to each dataset CSV. Produces a per-dataset summary of gender-relevant case proportions. Takes `--data-dir` and `--output-dir` as arguments.

#### `vllm_gender_perturbation.py`
LLM-based gender rewriting using the vLLM inference backend (default: `microsoft/phi-4`). Supports both `gender_swap` and `gender_removal` rewrites with dataset-specific prompts, pattern-based fallbacks when the LLM produces low-quality output, and GPU memory monitoring. Can fall back to standard HuggingFace pipeline if vLLM is unavailable.

---

### Stage 3: `perturbation/`

Scripts that apply all four perturbation types to produce the perturbed dataset splits.

#### `prompts.py`
Prompt library. Defines four prompter classes — `GenderSwapPrompter`, `GenderRemovalPrompter`, `ColorfulTonePrompter`, `UncertainTonePrompter` — each with dataset-specific prompt templates. Import via `get_prompter(perturbation_type)`. Used internally by `run_perturbation.py`.

#### `run_perturbation.py`
Unified perturbation runner. Loads `_format.csv`, applies one perturbation type to the `clinical_context` column using Llama-3.3-70B-Instruct (via HuggingFace pipeline), and writes the result to a new CSV.

```bash
python run_perturbation.py \
  --type   {gender_swap,gender_removal,colorful,uncertain} \
  --input  cleaned_data/_format.csv \
  --output perturbed_data/gender_swap.csv \
  --model  meta-llama/Llama-3.3-70B-Instruct \   # optional
  --max-rows 100                                  # optional, for testing
```

Note: `colorful` and `uncertain` are applied only to AskDocs and OncQA rows.

---

### Stage 4: `llm_sampling/`

Scripts that query each LLM on each perturbation split for the three clinical-decision tasks.

#### `clinical_decision_azure_openai.py`
Inference client for Azure-hosted models (GPT-4o). For each row, sends the clinical context with a structured prompt and parses YES/NO responses for `manage`, `visit`, and `resource`. Requires `AZURE_OPENAI_KEY` and `AZURE_OPENAI_ENDPOINT` environment variables.

```bash
python clinical_decision_azure_openai.py \
  --input-file  data_for_llm/gender_swap.csv \
  --output-file results/gpt4o_gender_swap_seed0.csv \
  --model       gpt-4o \
  --seed        0 \
  --max-rows    50   # optional
```

#### `clinical_decision_direct_models.py`
Inference client for locally-hosted HuggingFace models (DeepSeek, MedGemma, Llama, Qwen). Same interface as the Azure script; loads the model onto a specified GPU device.

```bash
python clinical_decision_direct_models.py \
  --input-file  data_for_llm/baseline.csv \
  --output-file results/llama_baseline_seed1.csv \
  --model       meta-llama/Llama-3.3-70B-Instruct \
  --seed        1 \
  --device      0
```

#### `run_azure_openai_sweep.py`
Orchestrates a full sweep of `clinical_decision_azure_openai.py` across all perturbation CSVs in a data directory and all requested seeds. Spawns one subprocess per (file, seed) combination. Use `--model` and `--seeds` to configure the sweep.

#### `run_direct_models_sweep.py`
Same as above for direct (HuggingFace) models. Supports multiple models via `--models` and GPU device assignment via `--device`.

## License

This dataset and associated scripts are released under the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

You are free to share and adapt the material for any purpose, provided you give appropriate credit and distribute any derivative works under the same license.
