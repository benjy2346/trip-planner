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
    ap.add_argument("--workers", type=int, default=4, help="并发样本数（每个再并发3个子图，勿过高以免压垮 MCP）")
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

    # 写全量 requests.jsonl（含已完成/跳过的，做溯源），并挑出真正要处理的 todo。
    todo = []
    with open(run_dir / "requests.jsonl", "w", encoding="utf-8") as freq:
        for i, req in enumerate(requests):
            difficulty = "hard" if i >= len(requests) - n_hard else "standard"
            record_id = f"sft_{args.run_slug}_{i:04d}"
            freq.write(json.dumps({"record_id": record_id, "request": req.model_dump()},
                                  ensure_ascii=False) + "\n")
            if record_id not in done:
                todo.append((req, difficulty, record_id))

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

    sem = asyncio.Semaphore(args.workers)

    async def process(req, difficulty, record_id) -> dict:
        async with sem:
            if eval_signature(req) in eval_sigs:
                return {"kind": "overlap", "record_id": record_id}
            try:
                context = await snapshot_context(req)
            except Exception as e:
                return {"kind": "context_fail", "record_id": record_id, "error": str(e)}
            # teacher 生成单条失败（超时/限流等）只记错误并跳过，绝不让整个 run 崩掉丢 manifest。
            try:
                response = await teacher.ainvoke(build_planner_messages(context))
            except Exception as e:
                return {"kind": "teacher_fail", "record_id": record_id, "error": str(e)}
            record = {"record_id": record_id, "difficulty": difficulty,
                      "request": req.model_dump(), "context": context}
            metrics = evaluate_output(record, response.content)
            return {"kind": "pass" if metrics["hard_pass"] else "fail",
                    "record_id": record_id, "record": record, "response": response.content,
                    "metrics": metrics, "usage": extract_usage(response)}

    await init_amap_tools()
    n = 0
    try:
        with open(records_path, "a", encoding="utf-8") as frec, \
             open(run_dir / "errors.jsonl", "a", encoding="utf-8") as ferr:
            for coro in asyncio.as_completed([process(*t) for t in todo]):
                r = await coro
                n += 1
                kind = r["kind"]
                if kind == "overlap":
                    stats["eval_overlap_skipped"] += 1
                    continue
                if kind == "context_fail":
                    stats["context_failed"] += 1
                    ferr.write(json.dumps({"record_id": r["record_id"], "stage": "context",
                                           "error": r["error"]}, ensure_ascii=False) + "\n")
                    continue
                if kind == "teacher_fail":
                    stats["teacher_failed"] += 1
                    ferr.write(json.dumps({"record_id": r["record_id"], "stage": "teacher",
                                           "error": r["error"]}, ensure_ascii=False) + "\n")
                    continue
                stats["usage"]["prompt_tokens"] += r["usage"]["prompt_tokens"]
                stats["usage"]["completion_tokens"] += r["usage"]["completion_tokens"]
                if kind == "pass":
                    stats["hard_pass"] += 1
                    rec = r["record"]
                    rec["teacher_output"] = _strip_fences(r["response"])
                    frec.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    frec.flush()
                else:
                    stats["hard_fail"] += 1
                    ferr.write(json.dumps({"record_id": r["record_id"], "stage": "audit",
                                           "violations": r["metrics"]["violations"][:10],
                                           "output": r["response"]}, ensure_ascii=False) + "\n")
                if n % 25 == 0:
                    write_manifest()  # 周期性落盘 manifest，长 run 中途也不丢进度
                print(f"[{n}/{len(todo)}] {r['record_id']} hard_pass={kind == 'pass'}")
    finally:
        await close_amap_tools()
        manifest = write_manifest()

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
