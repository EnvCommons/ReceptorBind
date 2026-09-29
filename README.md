# ReceptorBind

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/ReceptorBind)

## Description

**ReceptorBind** is an environment for evaluating agents on receptor binding identification tasks. Each task presents four molecules (SMILES strings) and asks the agent to identify which molecule does NOT bind to a given human cell receptor in a specified mode (agonist or antagonist). The dataset is generated from EveBio Data Release 9, a large-scale receptor pharmacology dataset.

## Capabilities

- Identifying non-binding molecules from a set of structurally similar candidates
- Reasoning about receptor pharmacology (agonist/antagonist modes)
- Understanding structure-activity relationships for nuclear receptors and 7TM receptors
- Working with molecular SMILES notation for pharmacological reasoning

## Compute Requirements

ReceptorBind does not require a sandbox. It has minimal compute requirements.

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en).

## Tasks

There are two splits: train (1,000 tasks) and test (100 tasks). Each task presents a multiple-choice question with 4 SMILES string options. Three options are confirmed active binders (based on pXC50 from EveBio screening data), and one is a confirmed inactive compound. The agent must identify the non-binder.

Distractors (the 3 active binders) are selected by Tanimoto similarity (ECFP4 Morgan fingerprints, radius 2) to be structurally similar to the inactive answer, preventing the agent from simply spotting a structural outlier and forcing genuine pharmacological reasoning.

Tasks cover 247 unique (receptor, mode) pairs in training and 84 in test, spanning Nuclear Receptors (NR) and 7 Transmembrane Receptors (7TM) with both agonist and antagonist modes.

## Reward Structure

This is a sparse, verifiable reward environment. The agent calls the `answer` tool once with the SMILES string of its chosen non-binder.

- **Correct**: Reward **1.0** if the submitted canonical SMILES matches the expected non-binder.
- **Incorrect**: Reward **0.0** if the submitted SMILES is valid but does not match.
- **Invalid SMILES**: Reward **0.0**, not graded; the episode stays open so the agent can resubmit.

We do not use LLM graders for this task.

## Data

Tasks are generated from EveBio Data Release 9, a large-scale receptor pharmacology dataset with screening results across ~1,400 compounds and ~200 targets. Train/test splits are performed at the compound level to prevent data leakage. Data files are stored on the OpenReward platform.

## Tools

Agents are given a single tool:

- `answer`: Submit the SMILES string of the molecule that is NOT a binder for the given receptor and mode. The SMILES is validated with RDKit and compared against the expected answer via canonical SMILES matching. Only one answer is graded per task; an unparseable SMILES is not graded and can be resubmitted.

## Time Horizon

ReceptorBind is a single-turn environment. The agent receives a multiple-choice question and submits one answer. Each task requires one tool call, plus a resubmission if the first SMILES cannot be parsed.

## Environment Difficulty

[Statistics on environment difficulty here]

## Other Environment Requirements

There are no further environment requirements; ReceptorBind works out of the box with the OpenReward endpoint without any secrets.

## Safety

Agents in ReceptorBind are asked to identify non-binding molecules from pharmacological screening data. The environment does not present direct safety risks, as agents only provide SMILES strings evaluated computationally, with no access to external systems or real pharmacological processes.

## Citations

```bibtex
@dataset{evebio2024,
  author    = {EveBio},
  title     = {EveBio Data Release 9},
  year      = {2024},
  url       = {https://evebio.org}
}
```
