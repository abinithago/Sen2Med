#!/usr/bin/env python3
"""
Create gold_standard_baseline.csv from baseline.csv:
- Replace llm_manage, llm_visit, llm_resource, llm_resource_specification with
  gold_standard_manage, gold_standard_visit, gold_standard_resource, gold_standard_resource_allocation.
- For context_ids that appear in treatment_data (baseline perturbation): use mode of
  MANAGE, VISIT, RESOURCE from treatment_data and one Resource Allocation from a row with those mode values.
- For context_ids not in treatment_data: use mode across the 3 gpt-4o/baseline seed files and one
  llm_resource_specification from a row with those mode values.
"""
import csv
from collections import Counter
from pathlib import Path

# Paths
BASE = Path("/home/abinitha/scratch/abinitha/MedPerturb")
BASELINE_CSV = BASE / "cleaned_data/clinical_decisions/baseline.csv"
TREATMENT_CSV = BASE / "cleaned_data/centaur_lab_results/treatment_data.csv"
SEED_DIR = BASE / "cleaned_data/clinical_decisions/gpt-4o/baseline"
OUT_CSV = BASE / "cleaned_data/clinical_decisions/gold_standard_baseline.csv"

# Normalize baseline dataset name to treatment_data Dataset (for matching)
DATASET_NORM = {"usmle_derm": "usmle", "MeDiSumQA": "MediSumQA"}


def norm_dataset(ds):
    return DATASET_NORM.get(ds, ds)


def mode_val(values):
    """Return most common value; tie-break by first occurrence."""
    if not values:
        return None
    c = Counter(values)
    max_count = max(c.values())
    for v in values:
        if c[v] == max_count:
            return v
    return c.most_common(1)[0][0]


def to_yes_no(val):
    """Convert 0/1 to NO/YES for consistency with baseline."""
    if val in (1, "1", "YES"):
        return "YES"
    if val in (0, "0", "NO"):
        return "NO"
    return val


def build_gold_from_treatment():
    """For each (Dataset, Context_ID) in treatment_data with Perturbation=baseline,
    compute mode of MANAGE, VISIT, RESOURCE and pick one Resource Allocation.
    Returns dict: (dataset_norm, context_id) -> (manage, visit, resource, resource_allocation).
    """
    gold = {}
    with open(TREATMENT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows_by_key = {}
        for row in reader:
            if row.get("Perturbation") != "baseline":
                continue
            ds = row["Dataset"]
            ctx = str(row["Context_ID"]).strip()
            key = (ds, ctx)
            try:
                m, v, r = int(row["MANAGE"]), int(row["VISIT"]), int(row["RESOURCE"])
            except (ValueError, KeyError):
                continue
            ra = row.get("Resource Allocation", "").strip()
            if key not in rows_by_key:
                rows_by_key[key] = []
            rows_by_key[key].append((m, v, r, ra))

    for key, rows in rows_by_key.items():
        ds, ctx = key
        manage_vals = [r[0] for r in rows]
        visit_vals = [r[1] for r in rows]
        resource_vals = [r[2] for r in rows]
        mode_m = mode_val(manage_vals)
        mode_v = mode_val(visit_vals)
        mode_r = mode_val(resource_vals)
        # Pick one row that has (mode_m, mode_v, mode_r) and use its Resource Allocation
        ra = ""
        for m, v, r, allocation in rows:
            if m == mode_m and v == mode_v and r == mode_r:
                ra = allocation
                break
        if not ra and rows:
            ra = rows[0][3]
        gold[key] = (
            to_yes_no(mode_m),
            to_yes_no(mode_v),
            to_yes_no(mode_r),
            ra,
        )
    return gold


def build_gold_from_seeds():
    """For each (dataset, context_id) in the seed files, compute mode of llm_manage, llm_visit, llm_resource
    and pick one llm_resource_specification. Returns dict: (dataset, context_id) -> (manage, visit, resource, resource_allocation).
    """
    all_rows = []
    for p in sorted(SEED_DIR.glob("baseline_seed*.csv")):
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_source_file"] = p.name
                all_rows.append(row)

    keys = set((row["dataset"], str(row["context_id"]).strip()) for row in all_rows)
    gold = {}
    for (ds, ctx) in keys:
        rows = [
            r
            for r in all_rows
            if r["dataset"] == ds and str(r["context_id"]).strip() == ctx
        ]
        if not rows:
            continue
        m_vals = [r["llm_manage"].strip().upper() for r in rows]
        v_vals = [r["llm_visit"].strip().upper() for r in rows]
        r_vals = [r["llm_resource"].strip().upper() for r in rows]
        mode_m = mode_val(m_vals) or "NO"
        mode_v = mode_val(v_vals) or "NO"
        mode_r = mode_val(r_vals) or "NO"
        if mode_m not in ("YES", "NO"):
            mode_m = "YES" if mode_m == "YES" else "NO"
        if mode_v not in ("YES", "NO"):
            mode_v = "YES" if mode_v == "YES" else "NO"
        if mode_r not in ("YES", "NO"):
            mode_r = "YES" if mode_r == "YES" else "NO"
        ra = ""
        for r in rows:
            if (
                r["llm_manage"].strip().upper() == mode_m
                and r["llm_visit"].strip().upper() == mode_v
                and r["llm_resource"].strip().upper() == mode_r
            ):
                ra = r.get("llm_resource_specification", "").strip()
                break
        if not ra and rows:
            ra = rows[0].get("llm_resource_specification", "").strip()
        gold[(ds, ctx)] = (mode_m, mode_v, mode_r, ra)
    return gold


def main():
    treatment_gold = build_gold_from_treatment()
    seed_gold = build_gold_from_seeds()

    # Baseline columns: drop llm_* and add gold_standard_*
    out_cols = [
        "dataset",
        "context_id",
        "clinical_context",
        "original_gender",
        "age",
        "gendered_condition",
        "perturbation",
        "provided_physician_response",
        "gold_standard_manage",
        "gold_standard_visit",
        "gold_standard_resource",
        "gold_standard_resource_allocation",
    ]

    with open(BASELINE_CSV, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        baseline_cols = reader.fieldnames
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=out_cols, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for row in reader:
                ds = row["dataset"]
                ctx = str(row["context_id"]).strip()
                key_norm = (norm_dataset(ds), ctx)
                if key_norm in treatment_gold:
                    gs_m, gs_v, gs_r, gs_ra = treatment_gold[key_norm]
                else:
                    seed_key = (ds, ctx)
                    if seed_key in seed_gold:
                        gs_m, gs_v, gs_r, gs_ra = seed_gold[seed_key]
                    else:
                        # Fallback: use original llm values if no seed (shouldn't happen for baseline)
                        gs_m = row.get("llm_manage", "NO")
                        gs_v = row.get("llm_visit", "NO")
                        gs_r = row.get("llm_resource", "NO")
                        gs_ra = row.get("llm_resource_specification", "")
                out_row = {
                    "dataset": row["dataset"],
                    "context_id": row["context_id"],
                    "clinical_context": row["clinical_context"],
                    "original_gender": row["original_gender"],
                    "age": row["age"],
                    "gendered_condition": row["gendered_condition"],
                    "perturbation": row["perturbation"],
                    "provided_physician_response": row["provided_physician_response"],
                    "gold_standard_manage": gs_m,
                    "gold_standard_visit": gs_v,
                    "gold_standard_resource": gs_r,
                    "gold_standard_resource_allocation": gs_ra,
                }
                writer.writerow(out_row)

    print("Wrote", OUT_CSV)
    print("Treatment-derived keys:", len(treatment_gold))
    print("Seed-derived keys:", len(seed_gold))


if __name__ == "__main__":
    main()
