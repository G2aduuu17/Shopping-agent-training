"""Synthetic, single-decision shopping environment; NOT ShopSimulator Reward v3."""
from __future__ import annotations

import json
from pathlib import Path

REWARD_VERSION = "mini-catalog-v1"
SYSTEM = ('Select one product satisfying the customer requirements. '
          'Return only a JSON object: {"product_id":"ID"}. No explanation.')


def load_tasks(path: str | Path, split: str) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    ids = [r["task_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task_id (including across splits)")
    for row in rows:
        if row["split"] not in {"train", "dev", "final"}:
            raise ValueError("Unknown split")
        products = row["products"]
        if not products or len({p["id"] for p in products}) != len(products):
            raise ValueError("Catalog must have unique product IDs")
        req = row["requirements"]
        if not isinstance(req["max_price"], (int, float)) or req["max_price"] <= 0:
            raise ValueError("Invalid budget")
    selected = [r for r in rows if r["split"] == split]
    if not selected:
        raise ValueError(f"No tasks in split {split}")
    return selected


def messages(task: dict) -> list[dict]:
    # Only public requirements/catalog; no reward breakdown or private answer in the prompt.
    public = {"requirements": task["requirements"], "products": task["products"]}
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(public, ensure_ascii=False)}]


def score(task: dict, response: str, *, truncated: bool = False) -> dict:
    result = {"reward_version": REWARD_VERSION, "reward_valid": True,
              "reward": -1.0, "success": False, "reason": "invalid_json"}
    if truncated:
        return {**result, "reason": "generation_truncated"}
    try:
        choice = json.loads(response.strip())
    except (ValueError, TypeError):
        return result
    if not isinstance(choice, dict) or set(choice) != {"product_id"}:
        return {**result, "reason": "invalid_schema"}
    product = next((p for p in task["products"] if p["id"] == choice["product_id"]), None)
    if product is None:
        return {**result, "reason": "unknown_product"}
    req = task["requirements"]
    checks = {k: product.get(k) == v for k, v in req.items() if k != "max_price"}
    checks["budget"] = product["price"] <= req["max_price"]
    # Category and budget are hard gates; other public constraints provide partial credit.
    hard_ok = checks.get("category", True) and checks["budget"]
    success = all(checks.values())
    reward = (1.0 if success else 0.25 * sum(checks.values()) / len(checks)) if hard_ok else -0.5
    return {**result, "reward": reward, "success": success, "checks": checks,
            "reason": "success" if success else "constraint_failure"}
