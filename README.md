# ReceptorBind

An OpenReward environment for evaluating an agent's ability to identify molecules that do **not** bind to a given human cell receptor in a specified mode (agonist/antagonist).

## Task

Each task presents 4 SMILES strings and asks:

> Among the following molecules, which one is likely NOT considered [receptor name] [mode]?

Three of the four molecules are confirmed active binders (pXC50-based from EveBio screening data). The fourth is a confirmed inactive compound at that receptor+mode. The agent must identify the non-binder.

Distractors (the 3 active binders) are selected using **Tanimoto similarity** (ECFP4 Morgan fingerprints, radius 2) to be structurally similar to the inactive answer. This prevents the agent from simply spotting the structural outlier and forces genuine pharmacological reasoning. Inspired by the [ether0](https://github.com/Future-House/ether0) approach.

- **1000 training tasks**, **100 test tasks**
- **247 unique (receptor, mode) pairs** in train, **84** in test
- Covers **Nuclear Receptors (NR)** and **7 Transmembrane Receptors (7TM)**
- Modes: **Agonist** and **Antagonist**
- Binary scoring: 1.0 for correct, 0.0 for incorrect
- Verification via canonical SMILES comparison (RDKit)

## Data Source

Tasks are generated from [EveBio Data Release 9](https://evebio.org), a large-scale receptor pharmacology dataset with screening results across ~1,400 compounds and ~200 targets. Compounds are classified as active or inactive based on the `Result` field in the summary results table. Train/test splits are done at the compound level to prevent data leakage.

### Distractor Selection

For each task, the generator:
1. Picks an inactive compound (the answer) for a given receptor+mode
2. Computes Tanimoto similarity (Morgan fingerprints, radius 2, 2048 bits) between the answer and all active compounds for that receptor+mode
3. Ranks active compounds by similarity and takes the top 10 most similar
4. Samples 3 from those 10 as distractors

This yields a **1.6x improvement** in mean answer-distractor Tanimoto similarity compared to random distractor selection (0.155 vs 0.096).

## File Structure

```
receptorbind/
├── receptorbind.py      # Environment class (extends Environment)
├── server.py            # Minimal server wrapper
├── test_agent.py        # Agent integration test
├── golden_tests.py      # Unit tests (8 tests)
├── generate_tasks.py    # Data pipeline: EveBio zip -> task JSON
├── data/
│   ├── train.json       # 1000 training tasks
│   └── test.json        # 100 test tasks
├── requirements.txt
├── Dockerfile
└── README.md
```

## Quickstart

### Install dependencies

```bash
pip install -r requirements.txt
```

### Regenerate tasks (optional)

Requires `evebio_data_release9.zip` in the parent directory:

```bash
python generate_tasks.py
```

### Run tests

```bash
pytest golden_tests.py -v
```

### Start local server

```bash
python server.py
```

### Run agent test

Requires `OPENAI_API_KEY` environment variable and a running server:

```bash
python test_agent.py
```

### Docker

```bash
docker build -t receptorbind:latest .
docker run -p 8080:8080 receptorbind:latest
```

## Environment API

| Method | Description |
|---|---|
| `list_splits()` | Returns `["train", "test"]` |
| `list_tasks(split)` | Returns task specs (answer fields filtered out) |
| `get_prompt()` | Returns the multiple-choice question as `List[TextBlock]` |
| `answer(params)` | Submit a SMILES string; returns reward 1.0 if correct, 0.0 otherwise |

## Task Spec Fields

| Field | Description |
|---|---|
| `task_id` | Unique identifier (e.g., `train_0042`) |
| `split` | `"train"` or `"test"` |
| `receptor_name` | Full receptor name (e.g., "5-hydroxytryptamine receptor 1D") |
| `receptor_gene` | Gene symbol (e.g., "HTR1D") |
| `target_class` | `"NR"` or `"7TM"` |
| `mode` | `"Agonist"` or `"Antagonist"` |
| `options` | List of 4 SMILES strings |
| `question` | Full question text with labeled options |
