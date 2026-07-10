"""teacher（DeepSeek）生成 SFT 数据：子图快照 → teacher 生成 → 规则硬过滤。

运行（在 backend/ 下）：
  python -m ml.planner.data_gen --count 20 --seed 9000 --run-slug 260703_smoke20
节奏：smoke 20 → 人工审计 → 100 → 审计 → 1000。每个 run 独立目录 + manifest + usage。
"""
import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from app.config import get_settings
from app.planner.context import build_planner_messages
from app.services.amap_tools import init_amap_tools, close_amap_tools
from ml.planner.requestgen import iter_controlled_requests, eval_signature
from ml.planner.build_eval_set import snapshot_context
from ml.planner.rule_eval import evaluate_output, _strip_fences

EVAL_PATHS = ["ml/planner/eval/records.jsonl", "ml/planner/eval_hard/records.jsonl"]
RUNS_DIR = Path("ml/planner/data/runs")


def load_eval_signatures(paths: list[str]) -> set[str]:
    from app.models.schemas import TripRequest
    sigs = set()
    for p in paths:
        if not Path(p).exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sigs.add(eval_signature(TripRequest(**json.loads(line)["request"])))
    return sigs


def extract_usage(response) -> dict:
    u = getattr(response, "response_metadata", {}).get("token_usage", {})
    return {"prompt_tokens": u.get("prompt_tokens", 0),
            "completion_tokens": u.get("completion_tokens", 0)}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True, help="必须与评测集 seed(1000/2000) 不同")
    ap.add_argument("--run-slug", required=True, help="如 260703_smoke20")
    ap.add_argument("--hard-ratio", type=float, default=0.2)
    args = ap.parse_args()

    run_dir = RUNS_DIR / args.run_slug
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_sigs = load_eval_signatures(EVAL_PATHS)

    n_hard = int(args.count * args.hard_ratio)
    requests = (iter_controlled_requests(args.count - n_hard, "standard", seed=args.seed)
                + iter_controlled_requests(n_hard, "hard", seed=args.seed + 1))

    done = set()
    records_path = run_dir / "records.jsonl"
    if records_path.exists():
        with open(records_path, encoding="utf-8") as f:
            done = {json.loads(line)["record_id"] for line in f if line.strip()}

    stats = {"requested": len(requests), "eval_overlap_skipped": 0, "context_failed": 0,
             "teacher_failed": 0, "hard_pass": 0, "hard_fail": 0,
             "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

    # teacher = DeepSeek，专用长超时（生成完整行程比短解析慢，沿用全局 30s 链会频繁误超时）。
    s = get_settings()
    teacher = ChatOpenAI(base_url=s.deepseek_base_url, api_key=s.deepseek_api_key,
                         model=s.deepseek_model, temperature=0.2, max_tokens=8192, timeout=300)

    def write_manifest() -> dict:
        manifest = {"run_slug": args.run_slug, "created_at": datetime.now().isoformat(),
                    "seed": args.seed, "hard_ratio": args.hard_ratio, "stats": stats}
        (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        return manifest

    await init_amap_tools()
    try:
        with open(run_dir / "requests.jsonl", "w", encoding="utf-8") as freq, \
             open(records_path, "a", encoding="utf-8") as frec, \
             open(run_dir / "errors.jsonl", "a", encoding="utf-8") as ferr:
            for i, req in enumerate(requests):
                difficulty = "hard" if i >= len(requests) - n_hard else "standard"
                record_id = f"sft_{args.run_slug}_{i:04d}"
                freq.write(json.dumps({"record_id": record_id, "request": req.model_dump()},
                                      ensure_ascii=False) + "\n")
                if record_id in done:
                    continue
                if eval_signature(req) in eval_sigs:
                    stats["eval_overlap_skipped"] += 1
                    continue
                try:
                    context = await snapshot_context(req)
                except Exception as e:
                    stats["context_failed"] += 1
                    ferr.write(json.dumps({"record_id": record_id, "stage": "context",
                                           "error": str(e)}, ensure_ascii=False) + "\n")
                    continue

                # teacher 生成单条失败（超时/限流等）只记错误并跳过，绝不让整个 run 崩掉丢 manifest。
                try:
                    response = await teacher.ainvoke(build_planner_messages(context))
                except Exception as e:
                    stats["teacher_failed"] += 1
                    ferr.write(json.dumps({"record_id": record_id, "stage": "teacher",
                                           "error": str(e)}, ensure_ascii=False) + "\n")
                    continue
                usage = extract_usage(response)
                stats["usage"]["prompt_tokens"] += usage["prompt_tokens"]
                stats["usage"]["completion_tokens"] += usage["completion_tokens"]

                record = {"record_id": record_id, "difficulty": difficulty,
                          "request": req.model_dump(), "context": context}
                metrics = evaluate_output(record, response.content)
                if metrics["hard_pass"]:
                    stats["hard_pass"] += 1
                    record["teacher_output"] = _strip_fences(response.content)
                    frec.write(json.dumps(record, ensure_ascii=False) + "\n")
                    frec.flush()
                else:
                    stats["hard_fail"] += 1
                    ferr.write(json.dumps({"record_id": record_id, "stage": "audit",
                                           "violations": metrics["violations"][:10],
                                           "output": response.content},
                                          ensure_ascii=False) + "\n")
                stats["_processed"] = stats.get("_processed", 0) + 1
                if stats["_processed"] % 25 == 0:
                    write_manifest()  # 周期性落盘 manifest，长 run 中途也不丢进度
                print(f"[{i + 1}/{len(requests)}] {record_id} hard_pass={metrics['hard_pass']}")
    finally:
        await close_amap_tools()
        manifest = write_manifest()

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
