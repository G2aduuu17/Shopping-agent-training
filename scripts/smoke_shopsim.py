"""CPU-only contract smoke with real HTTP environment rewards, never an agent accuracy result."""
import argparse
import json
from datetime import UTC, datetime
from uuid import uuid4

from shopmini.artifacts import append_jsonl, sha256, write_json
from shopmini.shopsim import ShopSession
from shopmini.upstream import MINI_ROOT, UPSTREAM_COMMIT, require_upstream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    args = parser.parse_args()
    root = require_upstream()
    train_file = root / "data/grpo/train.jsonl"
    train_ids = {json.loads(s)["task_id"] for s in train_file.read_text().splitlines() if s.strip()}
    final_ids = {json.loads(s)["task_id"] for s in (root / "data/evaluation/tasks.jsonl").read_text().splitlines()
                 if s.strip()}
    task_id = 53  # Frozen train task; query is derived from its public user instruction.
    if task_id not in train_ids or task_id in final_ids:
        raise ValueError("Contract task must remain in train and outside final")
    out = MINI_ROOT / "outputs" / ("shopsim-smoke-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
                                  + "-" + uuid4().hex[:6])
    out.mkdir(parents=True)
    print(f"Results: {out}", flush=True)
    events = []

    def record(event):
        events.append(event)
        append_jsonl(out / "events.jsonl", event)

    checks = {}
    try:
        with ShopSession(args.base_url, event_sink=record) as session:
            initial = session.reset(task_id)
            if "浇水器" not in initial["instruction"]:
                raise ValueError("Frozen task public instruction changed")
            checks["hidden_reset_fields_excluded"] = set(initial) == {"instruction", "observation"}
            bad = session.step("open_product", {"asin": "INVALID"})
            checks["invalid_product_blocked"] = not bad["executed"]
            bad = session.step("view_description", {"extra": "bad"})
            checks["extra_arguments_blocked"] = not bad["executed"]
            result = session.step("search_products", {"query": "自动浇水器"})
            products = session.state.get("products", [])
            checks["search_returns_products"] = result["executed"] and bool(products)
            if not products:
                raise RuntimeError("No products returned from frozen search query")
            session.step("open_product", {"asin": products[0]["asin"]})
            checks["product_detail"] = session.state["page_type"] == "product_detail"
            session.step("view_description", {})
            checks["description_page"] = session.state["page_type"] == "information_subpage"
            bad = session.step("search_products", {"query": "自动浇水器"})
            checks["stale_page_action_blocked"] = not bad["executed"]
            session.step("prev_page", {})
            selected = 0
            for _, values in list(session.state.get("available_options", {}).items()):
                if not values:
                    raise ValueError("Empty option axis")
                response = session.step("select_option", {"value": values[0]})
                selected += int(response["executed"])
            checks["option_selected"] = selected > 0
            # Purchase the first publicly visible candidate: contract test, not reward optimization.
            purchase = session.step("buy_now", {})
            checks["purchase_terminal_valid"] = purchase.get("outcome", {}).get("training_eligible", False)
            try:
                session.step("buy_now", {})
            except ValueError:
                checks["post_terminal_step_blocked"] = True
            else:
                checks["post_terminal_step_blocked"] = False
        with ShopSession(args.base_url, event_sink=record) as session:
            session.reset(task_id)
            finish = session.step("finish_without_purchase", {"reason": "no_suitable_product"})
            outcome = finish.get("outcome", {})
            checks["early_stop_valid_negative_reward"] = (
                outcome.get("training_eligible") and outcome.get("termination_reason") == "early_abstain"
                and outcome.get("reward", 0) < 0)
        try:
            with ShopSession(args.base_url, event_sink=record) as session:
                session.reset(task_id)
                raise RuntimeError("intentional-smoke-exception")
        except RuntimeError as exc:
            if str(exc) != "intentional-smoke-exception":
                raise
        checks["exception_released"] = events[-1]["type"] == "released"
        # With one server slot, this would fail if any prior lease leaked.
        with ShopSession(args.base_url, event_sink=record) as session:
            session.reset(task_id)
        checks["reset_after_release"] = events[-1]["type"] == "released"
        summary = {"passed": all(checks.values()), "checks": checks, "task_id": task_id,
                   "upstream_commit": UPSTREAM_COMMIT, "task_file_sha256": sha256(train_file),
                   "scenario": "scripted_environment_contract; no LLM; not success-rate evaluation",
                   "terminals": [e["outcome"] for e in events if "outcome" in e],
                   "release_count": sum(e["type"] == "released" for e in events)}
    except Exception as exc:
        write_json(out / "summary.json", {"passed": False, "checks": checks,
                                            "error_type": type(exc).__name__, "error": str(exc)})
        raise
    write_json(out / "summary.json", summary)
    console = {**summary, "terminals": [
        {k: v for k, v in outcome.items() if k != "reward_detail"}
        for outcome in summary["terminals"]]}
    print(json.dumps(console, indent=2), flush=True)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
