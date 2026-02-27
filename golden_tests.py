"""Golden tests for the ReceptorBind environment."""

import json
from pathlib import Path

import pytest

from receptorbind import AnswerInput, ReceptorBind

DATA_DIR = Path(__file__).parent / "data"


def _load_tasks(split: str) -> list[dict]:
    with open(DATA_DIR / f"{split}.json", "r") as f:
        return json.load(f)


# --- Tests ---


def test_list_splits():
    splits = ReceptorBind.list_splits()
    names = {s.name for s in splits}
    assert names == {"train", "test"}


def test_task_counts():
    train_tasks = ReceptorBind.list_tasks("train")
    test_tasks = ReceptorBind.list_tasks("test")
    assert len(train_tasks) == 1000
    assert len(test_tasks) == 100


def test_list_tasks_no_leakage():
    """list_tasks() must not expose answer_smiles or answer_index."""
    for split in ["train", "test"]:
        tasks = ReceptorBind.list_tasks(split)
        for task in tasks:
            assert "answer_smiles" not in task
            assert "answer_index" not in task


@pytest.mark.asyncio
async def test_correct_answer():
    """Submitting the correct answer_smiles should give reward 1.0."""
    tasks = _load_tasks("test")
    for task in tasks[:5]:
        env = ReceptorBind(task_spec=task, secrets={})
        result = await env.answer(AnswerInput(answer=task["answer_smiles"]))
        assert result.reward == 1.0
        assert result.finished is True


@pytest.mark.asyncio
async def test_wrong_answer():
    """Submitting a wrong option should give reward 0.0."""
    tasks = _load_tasks("test")
    task = tasks[0]
    env = ReceptorBind(task_spec=task, secrets={})

    # Pick an option that is NOT the answer
    wrong = [opt for opt in task["options"] if opt != task["answer_smiles"]][0]
    result = await env.answer(AnswerInput(answer=wrong))
    assert result.reward == 0.0
    assert result.finished is True


@pytest.mark.asyncio
async def test_invalid_smiles():
    """Submitting invalid SMILES should give reward 0.0."""
    tasks = _load_tasks("test")
    task = tasks[0]
    env = ReceptorBind(task_spec=task, secrets={})

    result = await env.answer(AnswerInput(answer="not_a_smiles_string!!!"))
    assert result.reward == 0.0
    assert result.finished is True
    assert result.metadata["valid_smiles"] is False


@pytest.mark.asyncio
async def test_smiles_canonicalization():
    """A non-canonical form of the correct answer should still match."""
    tasks = _load_tasks("test")
    task = tasks[0]
    env = ReceptorBind(task_spec=task, secrets={})

    # RDKit canonical SMILES is already stored; submit with extra whitespace
    result = await env.answer(AnswerInput(answer=f"  {task['answer_smiles']}  "))
    assert result.reward == 1.0


@pytest.mark.asyncio
async def test_prompt_returns_textblock():
    """get_prompt() should return a list of TextBlock."""
    tasks = _load_tasks("test")
    task = tasks[0]
    env = ReceptorBind(task_spec=task, secrets={})
    prompt = await env.get_prompt()
    assert len(prompt) == 1
    assert "NOT considered" in prompt[0].text
    assert task["receptor_name"] in prompt[0].text
