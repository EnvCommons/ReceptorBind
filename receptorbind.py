"""ReceptorBind environment: identify which molecule does NOT bind a given receptor."""

import json
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field
from rdkit import Chem
import os

from openreward.environments import (
    Environment,
    JSONObject,
    Split,
    TextBlock,
    ToolOutput,
    tool,
)

if os.path.exists("/orwd_data"):
    ENV_PATH = Path("/orwd_data")
else:
    ENV_PATH = Path(__file__).parent


# Reward for a submission made after the task has already been graded. Negative
# so repeat submissions are actively discouraged, not merely left unscored.
REPEAT_SUBMISSION_PENALTY = -0.1


class ReceptorBindTaskSpec(BaseModel):
    task_id: str
    split: str
    receptor_name: str
    receptor_gene: str
    target_class: str
    mode: str
    options: list[str]
    question: str


class AnswerInput(BaseModel):
    answer: str = Field(
        description="The SMILES string of the molecule that is NOT a binder for the given receptor and mode."
    )


def load_all_tasks() -> dict[str, list[dict]]:
    """Load tasks from JSON files at module import time."""
    data_dir = ENV_PATH / "data"
    all_tasks = {}
    for split in ["train", "test"]:
        json_file = data_dir / f"{split}.json"
        if json_file.exists():
            with open(json_file, "r", encoding="utf-8") as f:
                all_tasks[split] = json.load(f)
    return all_tasks


ALL_TASKS = load_all_tasks()
ANSWERS = {
    task["task_id"]: task["answer_smiles"]
    for split_tasks in ALL_TASKS.values()
    for task in split_tasks
}


class ReceptorBind(Environment):
    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.validated = ReceptorBindTaskSpec.model_validate(task_spec)
        if self.validated.task_id not in ANSWERS:
            raise ValueError(f"Task {self.validated.task_id} not found in loaded data")
        self.submitted = 0
        self.answer_smiles = ANSWERS[self.validated.task_id]

    @classmethod
    def list_splits(cls) -> list[Split]:
        return [
            Split(name="train", type="train"),
            Split(name="test", type="test"),
        ]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split not in ALL_TASKS:
            return []
        return [
            {k: v for k, v in task.items() if k not in ("answer_smiles", "answer_index")}
            for task in ALL_TASKS[split]
        ]

    async def get_prompt(self) -> List[TextBlock]:
        return [TextBlock(text=self.validated.question)]

    @tool
    async def answer(self, params: AnswerInput) -> ToolOutput:
        """Submit the SMILES string of the molecule that is NOT a binder."""
        if self.submitted > 0:
            return ToolOutput(
                blocks=[TextBlock(text="An answer has already been submitted for this task. "
                                       "This episode is over: it is not re-graded, and repeat "
                                       "submissions are penalised (reward -0.1).")],
                metadata={"already_submitted": True, "submission_count": self.submitted},
                reward=REPEAT_SUBMISSION_PENALTY,
                finished=True,
            )

        submitted = params.answer.strip()

        # Validate SMILES. An unparseable or empty answer (an empty string parses
        # to a molecule with no atoms) is not graded and does not end the
        # episode, so the agent can resubmit one of the options.
        mol = Chem.MolFromSmiles(submitted)
        if mol is None or mol.GetNumAtoms() == 0:
            return ToolOutput(
                blocks=[TextBlock(text=f"Invalid SMILES: {submitted}. This answer was not graded; "
                                       "submit one of the option SMILES.")],
                metadata={"submitted": submitted, "valid_smiles": False, "correct": False},
                reward=0.0,
                finished=False,
            )

        # Canonicalize for comparison
        canonical_submitted = Chem.MolToSmiles(mol, canonical=True)
        canonical_answer = self.answer_smiles  # Already canonical from generation

        is_correct = canonical_submitted == canonical_answer

        if is_correct:
            feedback = (
                f"Correct! {canonical_submitted} is NOT a "
                f"{self.validated.receptor_name} {self.validated.mode}."
            )
        else:
            feedback = (
                f"Incorrect. {canonical_submitted} is not the right answer."
            )

        # An unparseable SMILES returns above without reaching the comparison, so
        # it neither consumes the attempt nor ends the episode.
        self.submitted += 1

        return ToolOutput(
            blocks=[TextBlock(text=feedback)],
            metadata={
                "submitted": canonical_submitted,
                "valid_smiles": True,
                "correct": is_correct,
            },
            reward=1.0 if is_correct else 0.0,
            finished=True,
        )
