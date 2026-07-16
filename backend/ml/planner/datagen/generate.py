"""Grounded teacher（DeepSeek）生成 SFT 数据驱动脚本。

参照 `ml/planner/data_gen.py` 的脚手架（manifest/resume/usage 统计/单样本容错/
asyncio.Semaphore 并发/周期性 manifest 落盘），但三段取数/生成/清洗全部换成 grounded
路径：

- 取数：`DataGenPlannerContextBuilder(amap_key, "open-meteo").collect(req)`（同步，
  用 `asyncio.to_thread` 包一层不阻塞事件循环）；`compact_for_planner(context)` 只用
  于喂给 teacher 的消息体，落盘的 `planner_context` 用未压缩的完整 context。
- 消息：`teacher.ainvoke(build_grounded_planner_messages(compact))`。
- 清洗：`is_clean()` 解析 `TripPlan(**json.loads(strip_fences(output)))` →
  `validate_grounded_trip_plan(plan, context)`，无违规才算干净；有违规写 errors.jsonl，
  不写 records.jsonl。

运行（在 backend/ 下）：
  python -m ml.planner.datagen.generate --count 20 --seed 9000 --run-slug 260714_smoke20
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.models.schemas import TripPlan
from app.planner.context import build_grounded_planner_messages
from app.planner.validation import validate_grounded_trip_plan
from ml.planner.datagen.context_builder import DataGenPlannerContextBuilder
from ml.planner.datagen.leakage import eval_signature, load_eval_signatures
from ml.planner.datagen.requests import iter_requests, to_trip_request

EVAL_PATHS = ["ml/planner/eval/records.jsonl", "ml/planner/eval_hard/records.jsonl"]
RUNS_DIR = Path("ml/planner/data/runs")


def _strip_fences(text: str) -> str:
    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def is_clean(plan_json: str, context: dict) -> tuple[bool, list[str]]:
    """解析 teacher 输出 → TripPlan → 硬校验，无违规即干净。

    解析/schema 失败不抛异常，统一收敛成 (False, ["<reason>"])。
    """
    try:
        data = json.loads(_strip_fences(plan_json))
        plan = TripPlan(**data)
    except Exception as e:  # noqa: BLE001 — 解析失败是正常的清洗结果，不是程序错误
        return False, [f"parse: {e}"]
    violations = validate_grounded_trip_plan(plan, context)
    return not violations, violations


def assemble_record(item: dict, context: dict, teacher_output: str) -> dict:
    """把请求素材 + 完整 context + teacher 原始输出组装成一条落盘记录。

    item 携带 record_id/control_spec（以及 iter_requests 产出的其余请求字段）；
    context 是 collect() 返回的完整快照；teacher_output 落盘前先 strip 代码围栏。
    """
    return {
        "record_id": item["record_id"],
        "request": item,
        "control_spec": item.get("control_spec", {}),
        "planner_context": context,
        "teacher_output": _strip_fences(teacher_output),
    }


def extract_usage(response) -> dict:
    u = getattr(response, "response_metadata", {}).get("token_usage", {})
    return {"prompt_tokens": u.get("prompt_tokens", 0),
            "completion_tokens": u.get("completion_tokens", 0)}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True, help="必须与评测集 seed 不同")
    ap.add_argument("--run-slug", required=True, help="如 260714_smoke20")
    ap.add_argument("--date-mode", default="mixed", choices=["mixed", "past", "future"])
    ap.add_argument("--workers", type=int, default=4, help="并发样本数，勿过高以免压垮高德/DeepSeek")
    args = ap.parse_args()

    run_dir = RUNS_DIR / args.run_slug
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_sigs = load_eval_signatures(EVAL_PATHS)

    items = iter_requests(args.count, seed=args.seed, date_mode=args.date_mode)
    for i, item in enumerate(items):
        item["record_id"] = f"sft_{args.run_slug}_{i:04d}"

    done: set[str] = set()
    records_path = run_dir / "records.jsonl"
    if records_path.exists():
        with open(records_path, encoding="utf-8") as f:
            done = {json.loads(line)["record_id"] for line in f if line.strip()}

    # 写全量 requests.jsonl（含已完成/跳过的，做溯源），并挑出真正要处理的 todo。
    todo = []
    with open(run_dir / "requests.jsonl", "w", encoding="utf-8") as freq:
        for item in items:
            freq.write(json.dumps(item, ensure_ascii=False) + "\n")
            if item["record_id"] not in done:
                todo.append(item)

    stats = {"requested": len(items), "eval_overlap_skipped": 0, "context_failed": 0,
             "teacher_failed": 0, "clean": 0, "dirty": 0,
             "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

    s = get_settings()
    builder = DataGenPlannerContextBuilder(s.amap_api_key, "open-meteo")
    # teacher = DeepSeek，专用长超时（生成完整行程比短解析慢，沿用全局 30s 链会频繁误超时）。
    teacher = ChatOpenAI(base_url=s.deepseek_base_url, api_key=s.deepseek_api_key,
                         model=s.deepseek_model, temperature=0.2, max_tokens=8192, timeout=300)

    def write_manifest() -> dict:
        manifest = {"run_slug": args.run_slug, "created_at": datetime.now().isoformat(),
                    "seed": args.seed, "date_mode": args.date_mode, "stats": stats}
        (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        return manifest

    sem = asyncio.Semaphore(args.workers)

    async def process(item: dict) -> dict:
        record_id = item["record_id"]
        async with sem:
            try:
                req = to_trip_request(item)
                if eval_signature(req) in eval_sigs:
                    return {"kind": "overlap", "record_id": record_id}
            except Exception as e:
                return {"kind": "context_fail", "record_id": record_id, "error": str(e), "stage": "request"}
            try:
                context = await asyncio.to_thread(builder.collect, req)
            except Exception as e:
                return {"kind": "context_fail", "record_id": record_id, "error": str(e)}
            compact = builder.compact_for_planner(context)
            # teacher 生成单条失败（超时/限流等）只记错误并跳过，绝不让整个 run 崩掉丢 manifest。
            try:
                response = await teacher.ainvoke(build_grounded_planner_messages(compact))
            except Exception as e:
                return {"kind": "teacher_fail", "record_id": record_id, "error": str(e)}
            ok, violations = is_clean(response.content, context)
            return {"kind": "clean" if ok else "dirty", "record_id": record_id,
                    "item": item, "context": context, "response": response.content,
                    "violations": violations, "usage": extract_usage(response)}

    n = 0
    try:
        with open(records_path, "a", encoding="utf-8") as frec, \
             open(run_dir / "errors.jsonl", "a", encoding="utf-8") as ferr:
            for coro in asyncio.as_completed([process(item) for item in todo]):
                r = await coro
                n += 1
                kind = r["kind"]
                if kind == "overlap":
                    stats["eval_overlap_skipped"] += 1
                    continue
                if kind == "context_fail":
                    stats["context_failed"] += 1
                    ferr.write(json.dumps({"record_id": r["record_id"], "stage": r.get("stage", "context"),
                                           "error": r["error"]}, ensure_ascii=False) + "\n")
                    continue
                if kind == "teacher_fail":
                    stats["teacher_failed"] += 1
                    ferr.write(json.dumps({"record_id": r["record_id"], "stage": "teacher",
                                           "error": r["error"]}, ensure_ascii=False) + "\n")
                    continue
                stats["usage"]["prompt_tokens"] += r["usage"]["prompt_tokens"]
                stats["usage"]["completion_tokens"] += r["usage"]["completion_tokens"]
                if kind == "clean":
                    stats["clean"] += 1
                    rec = assemble_record(r["item"], r["context"], r["response"])
                    frec.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    frec.flush()
                else:
                    stats["dirty"] += 1
                    ferr.write(json.dumps({"record_id": r["record_id"], "stage": "audit",
                                           "violations": r["violations"][:10],
                                           "output": r["response"]}, ensure_ascii=False) + "\n")
                if n % 25 == 0:
                    write_manifest()  # 周期性落盘 manifest，长 run 中途也不丢进度
                print(f"[{n}/{len(todo)}] {r['record_id']} clean={kind == 'clean'}")
    finally:
        manifest = write_manifest()

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
