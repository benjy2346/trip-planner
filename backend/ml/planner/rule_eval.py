"""规则评测：对任意 OpenAI-compatible 端点跑冻结评测集，输出 hardpass/softpass。

运行（在 backend/ 下）：
  python -m ml.planner.rule_eval --records ml/planner/eval/records.jsonl \
    --base-url https://api.deepseek.com/v1 --model deepseek-chat --api-key-env DEEPSEEK_API_KEY \
    --output-dir ml/planner/runs_eval/deepseek_standard
"""
import argparse
import asyncio
import json
import os
from pathlib import Path

from langchain_openai import ChatOpenAI
from app.models.schemas import TripPlan
from app.planner.context import build_planner_messages
from app.planner.validation import validate_trip_plan, recompute_budget


def _strip_fences(text: str) -> str:
    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def evaluate_output(record: dict, output_text: str) -> dict:
    m = {"record_id": record["record_id"], "json_ok": False, "schema_ok": False,
         "violations": [], "hard_pass": False, "meal_repeat_count": 0,
         "budget_ok": True, "recomputed_total": None, "soft_pass": False}
    try:
        data = json.loads(_strip_fences(output_text))
        m["json_ok"] = True
        plan = TripPlan(**data)
        m["schema_ok"] = True
    except Exception as e:
        m["violations"] = [f"parse: {e}"]
        return m

    ctx = record["context"]
    m["violations"] = validate_trip_plan(plan, ctx)
    m["hard_pass"] = not m["violations"]

    names = [meal.name for d in plan.days for meal in d.meals if meal.type in ("lunch", "dinner")]
    m["meal_repeat_count"] = len(names) - len(set(names))

    budget = recompute_budget(plan, ctx["party"]["total"])
    m["recomputed_total"] = budget.total
    bc = ctx["budget_constraint"]
    if bc["amount"]:
        if bc["strictness"] == "hard":
            m["budget_ok"] = budget.total <= bc["amount"] and budget.total >= 0.4 * bc["amount"]
        else:
            m["budget_ok"] = budget.total <= 1.2 * bc["amount"]

    m["soft_pass"] = m["hard_pass"] and m["meal_repeat_count"] == 0 and m["budget_ok"]
    return m


def aggregate(metrics: list[dict]) -> dict:
    n = len(metrics)
    rate = lambda k: round(sum(1 for x in metrics if x[k]) / n * 100, 1) if n else 0.0
    return {"count": n, "json_ok": rate("json_ok"), "schema_ok": rate("schema_ok"),
            "hard_pass": rate("hard_pass"), "soft_pass": rate("soft_pass"),
            "budget_ok": rate("budget_ok"),
            "meal_repeat_avg": round(sum(x["meal_repeat_count"] for x in metrics) / n, 2) if n else 0}


async def _generate(llm: ChatOpenAI, record: dict, sem: asyncio.Semaphore) -> tuple[dict, str]:
    async with sem:
        resp = await llm.ainvoke(build_planner_messages(record["context"]))
        return record, resp.content


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    with open(args.records, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    llm = ChatOpenAI(base_url=args.base_url, api_key=os.environ.get(args.api_key_env, "EMPTY"),
                     model=args.model, temperature=args.temperature,
                     max_tokens=args.max_tokens, timeout=300)
    sem = asyncio.Semaphore(args.workers)
    results = await asyncio.gather(
        *[_generate(llm, r, sem) for r in records], return_exceptions=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    with open(out_dir / "generations.jsonl", "w", encoding="utf-8") as f:
        for item in results:
            if isinstance(item, Exception):
                print(f"❌ 生成失败: {item}")
                continue
            record, text = item
            m = evaluate_output(record, text)
            metrics.append(m)
            f.write(json.dumps({"record_id": record["record_id"], "output": text,
                                "metrics": m}, ensure_ascii=False) + "\n")

    report = {"model": args.model, "records": args.records, "summary": aggregate(metrics)}
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    s = report["summary"]
    md = (f"# Rule Eval: {args.model}\n\n| 指标 | 值 |\n| --- | ---: |\n"
          f"| 样本数 | {s['count']} |\n| json_ok | {s['json_ok']}% |\n"
          f"| schema_ok | {s['schema_ok']}% |\n| **hard_pass** | **{s['hard_pass']}%** |\n"
          f"| **soft_pass** | **{s['soft_pass']}%** |\n| budget_ok | {s['budget_ok']}% |\n"
          f"| 午晚餐平均重复 | {s['meal_repeat_avg']} |\n")
    (out_dir / "report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    asyncio.run(main())
