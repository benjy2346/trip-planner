"""grounded 规则打分器：读记录 + generations.jsonl → 逐条 metrics + 聚合报告。

与旧 rule_eval.py 的区别：吃 grounded 记录（compact_planner_context）、用
validate_grounded_trip_plan / recompute_grounded_budget，并解耦生成（见 generate.py）。
打分语义参照 helloagents eval_rule_metrics.py::evaluate_output，逻辑复用我们已移植的
app.planner.validation。

运行（在 backend/ 下）：
  python -m ml.planner.eval.rule_metrics \
    --records ml/planner/eval/records.jsonl \
    --generations runs_eval/base_standard/generations.jsonl \
    --output-dir runs_eval/base_standard
"""
import argparse
import json
from pathlib import Path

from app.models.schemas import TripPlan
from app.planner.output import is_lodging_breakfast_meal, meal_diversity_key, name_in_candidates
from app.planner.validation import recompute_grounded_budget, validate_grounded_trip_plan

_ATTRACTION_BUCKETS = ("classic_pois", "preference_pois", "scenic_pois", "experience_pois")


def _strip_fences(text: str) -> str:
    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def _candidate_names(snapshot: dict, buckets) -> list[str]:
    names = []
    for bucket in buckets:
        for item in snapshot.get(bucket) or []:
            if item.get("name"):
                names.append(item["name"])
    return names


def _rate(hit: int, total: int) -> float:
    return round(hit / total * 100, 1) if total else 100.0


def evaluate_output(record: dict, output_text: str) -> dict:
    ctx = record["compact_planner_context"]
    m = {"record_id": record["record_id"], "json_ok": False, "schema_ok": False,
         "violations": [], "hard_pass": False, "meal_repeat_count": 0,
         "attraction_grounding_rate": 0.0, "meal_grounding_rate": 0.0,
         "hotel_grounding_rate": 0.0, "meal_diversity_unique_rate": 0.0,
         "budget_ok": True, "recomputed_total": None, "soft_pass": False}
    try:
        data = json.loads(_strip_fences(output_text))
        m["json_ok"] = True
        plan = TripPlan(**data)
        m["schema_ok"] = True
    except Exception as e:
        m["violations"] = [f"parse: {e}"]
        return m

    m["violations"] = validate_grounded_trip_plan(plan, ctx)
    m["hard_pass"] = not m["violations"]

    snapshot = ctx["tool_snapshot"]
    attraction_candidates = _candidate_names(snapshot, _ATTRACTION_BUCKETS)
    hotel_candidates = _candidate_names(snapshot, ("hotel_pois",))
    food_candidates = _candidate_names(snapshot, ("food_pois",))

    # grounding 率
    a_hit = a_tot = h_hit = h_tot = f_hit = f_tot = 0
    meal_keys = []
    lunch_dinner_names = []
    for d in plan.days:
        for a in d.attractions:
            a_tot += 1
            if not attraction_candidates or name_in_candidates(a.name, attraction_candidates):
                a_hit += 1
        if d.hotel is not None:
            h_tot += 1
            if not hotel_candidates or name_in_candidates(d.hotel.name, hotel_candidates):
                h_hit += 1
        for meal in d.meals:
            mtype = (meal.type or "").lower()
            if is_lodging_breakfast_meal(meal.name, mtype):
                continue
            f_tot += 1
            if not food_candidates or name_in_candidates(meal.name, food_candidates):
                f_hit += 1
            key = meal_diversity_key(meal.name)
            if key:
                meal_keys.append(key)
            if mtype in ("lunch", "dinner"):
                lunch_dinner_names.append(meal.name)

    m["attraction_grounding_rate"] = _rate(a_hit, a_tot)
    m["meal_grounding_rate"] = _rate(f_hit, f_tot)
    m["hotel_grounding_rate"] = _rate(h_hit, h_tot)
    m["meal_diversity_unique_rate"] = _rate(len(set(meal_keys)), len(meal_keys))
    m["meal_repeat_count"] = len(lunch_dinner_names) - len(set(lunch_dinner_names))

    budget = recompute_grounded_budget(plan, ctx["party"]["total"])
    m["recomputed_total"] = budget.total
    bc = ctx.get("budget_constraint") or {}
    amount = bc.get("amount")
    if amount:
        if bc.get("strictness") == "hard":
            m["budget_ok"] = 0.4 * amount <= budget.total <= amount
        else:
            m["budget_ok"] = budget.total <= 1.2 * amount

    m["soft_pass"] = m["hard_pass"] and m["budget_ok"] and m["meal_repeat_count"] == 0
    return m


def aggregate(metrics: list[dict]) -> dict:
    n = len(metrics)
    rate = lambda k: round(sum(1 for x in metrics if x[k]) / n * 100, 1) if n else 0.0
    avg = lambda k: round(sum(x[k] for x in metrics) / n, 1) if n else 0.0
    return {"count": n, "json_ok": rate("json_ok"), "schema_ok": rate("schema_ok"),
            "hard_pass": rate("hard_pass"), "soft_pass": rate("soft_pass"),
            "budget_ok": rate("budget_ok"),
            "attraction_grounding_avg": avg("attraction_grounding_rate"),
            "meal_grounding_avg": avg("meal_grounding_rate"),
            "hotel_grounding_avg": avg("hotel_grounding_rate"),
            "meal_diversity_unique_avg": avg("meal_diversity_unique_rate"),
            "meal_repeat_avg": round(sum(x["meal_repeat_count"] for x in metrics) / n, 2) if n else 0}


def _render_md(model_tag: str, summary: dict) -> str:
    s = summary
    return (f"# Rule Eval: {model_tag}\n\n| 指标 | 值 |\n| --- | ---: |\n"
            f"| 样本数 | {s['count']} |\n| json_ok | {s['json_ok']}% |\n"
            f"| schema_ok | {s['schema_ok']}% |\n| **hard_pass** | **{s['hard_pass']}%** |\n"
            f"| soft_pass | {s['soft_pass']}% |\n| budget_ok | {s['budget_ok']}% |\n"
            f"| 景点 grounding 均值 | {s['attraction_grounding_avg']}% |\n"
            f"| 餐饮 grounding 均值 | {s['meal_grounding_avg']}% |\n"
            f"| 酒店 grounding 均值 | {s['hotel_grounding_avg']}% |\n"
            f"| 餐饮多样性均值 | {s['meal_diversity_unique_avg']}% |\n"
            f"| 午晚餐平均重复 | {s['meal_repeat_avg']} |\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--generations", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-tag", default="model")
    args = ap.parse_args()

    with open(args.records, encoding="utf-8") as f:
        records = {json.loads(l)["record_id"]: json.loads(l) for l in f if l.strip()}
    metrics = []
    with open(args.generations, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            gen = json.loads(line)
            rec = records.get(gen["record_id"])
            if rec is None:
                continue
            metrics.append(evaluate_output(rec, gen["output"]))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in metrics), encoding="utf-8")
    summary = aggregate(metrics)
    (out_dir / "report.json").write_text(
        json.dumps({"model": args.model_tag, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    md = _render_md(args.model_tag, summary)
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
