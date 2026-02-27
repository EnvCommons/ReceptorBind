"""Generate receptor binding tasks from EveBio data release 9.

Reads compound, target, and summary result tables from evebio_data_release9.zip
and creates multiple-choice negative classification tasks:
"Which molecule is NOT a [receptor] [mode]?"

Outputs:
    data/train.json  (1000 tasks)
    data/test.json   (100 tasks)
"""

import csv
import io
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
except ImportError:
    Chem = None
    AllChem = None
    DataStructs = None
    print("WARNING: rdkit not installed. SMILES will not be canonicalized.")

ZIP_PATH = Path(__file__).parent.parent / "evebio_data_release9.zip"
OUT_DIR = Path(__file__).parent / "data"

TRAIN_COUNT = 1000
TEST_COUNT = 100
OPTIONS_PER_TASK = 4  # 3 active + 1 inactive
SEED = 42

# Only human cell receptors (exclude Kinases)
ALLOWED_CLASSES = {"NR", "7TM"}
# Only agonist/antagonist modes (exclude generic "Binding")
ALLOWED_MODES = {"Agonist", "Antagonist"}
# Minimum active compounds needed for a target+mode to be eligible
MIN_ACTIVE = 4
# For Tanimoto-based distractor selection: pick distractors from the top-K
# most similar active compounds (then sample 3 from those K candidates).
# This ensures structural similarity without requiring a hard threshold.
TANIMOTO_TOP_K = 10


def canonicalize_smiles(smiles: str) -> str | None:
    """Canonicalize a SMILES string using RDKit. Returns None if invalid."""
    if Chem is None:
        return smiles.strip()
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def read_csv_from_zip(zf: zipfile.ZipFile, filename: str) -> list[dict]:
    """Read a CSV file from the zip archive and return list of dicts."""
    with zf.open(filename) as f:
        text = io.TextIOWrapper(f, encoding="utf-8")
        reader = csv.DictReader(text)
        return list(reader)


def load_data(zip_path: Path):
    """Load and join compound, target, and summary result data."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        targets = read_csv_from_zip(zf, "target_table.csv")
        compounds = read_csv_from_zip(zf, "compound_table.csv")
        results = read_csv_from_zip(zf, "summary_result_table.csv")

    # Build target info lookup: Target_ID -> {Name, Gene, Class}
    target_info = {}
    for t in targets:
        target_info[t["Target_ID"]] = {
            "name": t["Name"],
            "gene": t["Gene"],
            "class": t["Class"],
        }

    # Build compound SMILES lookup and Morgan fingerprints
    compound_smiles = {}
    compound_fps = {}
    for c in compounds:
        smiles = c.get("SMILES", "").strip()
        if not smiles:
            continue
        canonical = canonicalize_smiles(smiles)
        if canonical:
            cid = c["Compound_ID"]
            compound_smiles[cid] = canonical
            if AllChem is not None:
                mol = Chem.MolFromSmiles(canonical)
                if mol is not None:
                    compound_fps[cid] = AllChem.GetMorganFingerprintAsBitVect(
                        mol, radius=2, nBits=2048
                    )

    # Classify compounds per (target, mode) as active or inactive
    # active_compounds[(target_id, mode)] = set of compound_ids
    # inactive_compounds[(target_id, mode)] = set of compound_ids
    active_compounds = defaultdict(set)
    inactive_compounds = defaultdict(set)

    for row in results:
        cls = row["Class"]
        mode = row["Mode"]
        target_id = row["Target_ID"]
        compound_id = row["Compound_ID"]
        result_str = row["Result"]

        # Filter to allowed classes and modes
        if cls not in ALLOWED_CLASSES or mode not in ALLOWED_MODES:
            continue

        # Skip compounds without valid SMILES
        if compound_id not in compound_smiles:
            continue

        # Skip mutant targets
        if row.get("Is_Mutant_Flag", "FALSE") == "TRUE":
            continue

        key = (target_id, mode)
        if "Active" in result_str:
            active_compounds[key].add(compound_id)
        elif "Inactive" in result_str or "Likely Inactive" in result_str:
            inactive_compounds[key].add(compound_id)

    return target_info, compound_smiles, compound_fps, active_compounds, inactive_compounds


def build_eligible_pairs(target_info, active_compounds, inactive_compounds):
    """Find (target, mode) pairs with enough active and inactive compounds."""
    eligible = []
    for key in active_compounds:
        target_id, mode = key
        if target_id not in target_info:
            continue
        if target_info[target_id]["class"] not in ALLOWED_CLASSES:
            continue
        n_active = len(active_compounds[key])
        # Need inactive compounds that are NOT in the active set for this target+mode
        pure_inactive = inactive_compounds[key] - active_compounds[key]
        n_inactive = len(pure_inactive)
        if n_active >= MIN_ACTIVE and n_inactive >= 1:
            eligible.append({
                "target_id": target_id,
                "mode": mode,
                "n_active": n_active,
                "n_inactive": n_inactive,
            })
    return eligible


def generate_task(
    task_id: str,
    split: str,
    target_id: str,
    mode: str,
    target_info: dict,
    compound_smiles: dict,
    compound_fps: dict,
    active_set: set,
    inactive_set: set,
    rng: random.Random,
) -> dict | None:
    """Generate a single multiple-choice task.

    Uses Tanimoto similarity to select distractors (active compounds) that are
    structurally similar to the inactive answer, making the task harder.
    """
    active_list = list(active_set)
    inactive_list = list(inactive_set)

    if len(active_list) < OPTIONS_PER_TASK - 1 or len(inactive_list) < 1:
        return None

    # Pick 1 inactive compound (the answer)
    chosen_inactive = rng.choice(inactive_list)

    # Select distractors by Tanimoto similarity to the inactive compound
    inact_fp = compound_fps.get(chosen_inactive)
    if inact_fp is not None and DataStructs is not None:
        # Rank active compounds by similarity to the inactive compound
        scored = []
        for cid in active_list:
            fp = compound_fps.get(cid)
            if fp is not None:
                sim = DataStructs.TanimotoSimilarity(inact_fp, fp)
                scored.append((sim, cid))
        scored.sort(reverse=True)

        # Take top-K candidates, then sample 3 from them
        candidates = [cid for _, cid in scored[:TANIMOTO_TOP_K]]
        if len(candidates) >= OPTIONS_PER_TASK - 1:
            chosen_active = rng.sample(candidates, OPTIONS_PER_TASK - 1)
        else:
            # Not enough candidates with fingerprints; use what we have
            chosen_active = rng.sample(active_list, OPTIONS_PER_TASK - 1)
    else:
        # Fallback: random selection if fingerprints unavailable
        chosen_active = rng.sample(active_list, OPTIONS_PER_TASK - 1)

    # Build options with SMILES
    options = []
    for cid in chosen_active:
        options.append(compound_smiles[cid])
    answer_smiles = compound_smiles[chosen_inactive]
    options.append(answer_smiles)

    # Shuffle options
    rng.shuffle(options)
    answer_index = options.index(answer_smiles)

    # Build question text
    tinfo = target_info[target_id]
    receptor_name = tinfo["name"]
    receptor_gene = tinfo["gene"]
    target_class = tinfo["class"]

    labels = ["A", "B", "C", "D"]
    options_text = "\n".join(
        f"{labels[i]}) {options[i]}" for i in range(len(options))
    )

    question = (
        f"Among the following molecules, which one is likely NOT considered "
        f"{receptor_name} {mode}?\n\n"
        f"{options_text}\n\n"
        f"Respond with the SMILES string of the molecule that is NOT "
        f"a {receptor_name} {mode}."
    )

    return {
        "task_id": task_id,
        "split": split,
        "receptor_name": receptor_name,
        "receptor_gene": receptor_gene,
        "target_class": target_class,
        "mode": mode,
        "options": options,
        "question": question,
        "answer_smiles": answer_smiles,
        "answer_index": answer_index,
    }


def split_compounds(compound_smiles: dict, rng: random.Random, test_frac: float = 0.15):
    """Split compound IDs into train and test sets."""
    all_ids = sorted(compound_smiles.keys())
    rng.shuffle(all_ids)
    n_test = max(1, int(len(all_ids) * test_frac))
    test_ids = set(all_ids[:n_test])
    train_ids = set(all_ids[n_test:])
    return train_ids, test_ids


def generate_all_tasks(
    target_info,
    compound_smiles,
    compound_fps,
    active_compounds,
    inactive_compounds,
    train_compound_ids,
    test_compound_ids,
    n_train,
    n_test,
    rng,
):
    """Generate train and test task sets."""
    eligible = build_eligible_pairs(target_info, active_compounds, inactive_compounds)
    print(f"Found {len(eligible)} eligible (target, mode) pairs")

    # For each eligible pair, compute split-specific active/inactive sets
    pair_data = []
    for ep in eligible:
        target_id = ep["target_id"]
        mode = ep["mode"]
        key = (target_id, mode)
        pure_inactive = inactive_compounds[key] - active_compounds[key]

        train_active = active_compounds[key] & train_compound_ids
        train_inactive = pure_inactive & train_compound_ids
        test_active = active_compounds[key] & test_compound_ids
        test_inactive = pure_inactive & test_compound_ids

        pair_data.append({
            "target_id": target_id,
            "mode": mode,
            "train_active": train_active,
            "train_inactive": train_inactive,
            "test_active": test_active,
            "test_inactive": test_inactive,
        })

    # Filter pairs that have enough compounds in each split
    train_eligible = [
        p for p in pair_data
        if len(p["train_active"]) >= OPTIONS_PER_TASK - 1 and len(p["train_inactive"]) >= 1
    ]
    test_eligible = [
        p for p in pair_data
        if len(p["test_active"]) >= OPTIONS_PER_TASK - 1 and len(p["test_inactive"]) >= 1
    ]

    print(f"Train-eligible pairs: {len(train_eligible)}")
    print(f"Test-eligible pairs: {len(test_eligible)}")

    def generate_for_split(eligible_pairs, split_name, count):
        tasks = []
        attempts = 0
        max_attempts = count * 20  # avoid infinite loops

        while len(tasks) < count and attempts < max_attempts:
            attempts += 1
            pair = rng.choice(eligible_pairs)
            active_key = f"{split_name}_active"
            inactive_key = f"{split_name}_inactive"

            task = generate_task(
                task_id=f"{split_name}_{len(tasks):04d}",
                split=split_name,
                target_id=pair["target_id"],
                mode=pair["mode"],
                target_info=target_info,
                compound_smiles=compound_smiles,
                compound_fps=compound_fps,
                active_set=pair[active_key],
                inactive_set=pair[inactive_key],
                rng=rng,
            )
            if task is not None:
                tasks.append(task)

        return tasks

    train_tasks = generate_for_split(train_eligible, "train", n_train)
    test_tasks = generate_for_split(test_eligible, "test", n_test)

    print(f"Generated {len(train_tasks)} train tasks, {len(test_tasks)} test tasks")
    return train_tasks, test_tasks


def main():
    rng = random.Random(SEED)

    print(f"Loading data from {ZIP_PATH}...")
    target_info, compound_smiles, compound_fps, active_compounds, inactive_compounds = load_data(ZIP_PATH)
    print(f"Loaded {len(target_info)} targets, {len(compound_smiles)} compounds with SMILES, {len(compound_fps)} fingerprints")

    # Split compounds into train/test
    train_ids, test_ids = split_compounds(compound_smiles, rng)
    print(f"Compound split: {len(train_ids)} train, {len(test_ids)} test")

    # Generate tasks
    train_tasks, test_tasks = generate_all_tasks(
        target_info,
        compound_smiles,
        compound_fps,
        active_compounds,
        inactive_compounds,
        train_ids,
        test_ids,
        TRAIN_COUNT,
        TEST_COUNT,
        rng,
    )

    # Write output
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUT_DIR / "train.json"
    test_path = OUT_DIR / "test.json"

    with open(train_path, "w") as f:
        json.dump(train_tasks, f, indent=2)
    print(f"Wrote {len(train_tasks)} tasks to {train_path}")

    with open(test_path, "w") as f:
        json.dump(test_tasks, f, indent=2)
    print(f"Wrote {len(test_tasks)} tasks to {test_path}")


if __name__ == "__main__":
    main()
