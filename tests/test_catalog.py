from itertools import chain
from pathlib import Path

from shopmini.catalog import load_tasks, score

DATA = Path(__file__).parents[1] / "data/catalog_tasks.jsonl"


def test_unique_splits_and_exactly_one_correct_product():
    splits = [load_tasks(DATA, split) for split in ("train", "dev", "final")]
    ids = [r["task_id"] for split in splits for r in split]
    assert len(ids) == len(set(ids))
    for task in chain.from_iterable(splits):
        results = [score(task, '{"product_id":"' + p["id"] + '"}') for p in task["products"]]
        assert sum(r["success"] for r in results) == 1
        assert min(r["reward"] for r in results) < 0


def test_failures_are_explicit_environment_rewards():
    task = load_tasks(DATA, "train")[0]
    for text in ("hello", "[]", '{"product_id":"Z"}', '{"product_id":"A","extra":1}'):
        result = score(task, text)
        assert result["reward_valid"] and result["reward"] == -1
    assert score(task, '{"product_id":"A"}', truncated=True)["reason"] == "generation_truncated"
