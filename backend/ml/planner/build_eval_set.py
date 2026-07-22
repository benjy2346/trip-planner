"""构建冻结评测集：真跑三个子图拍候选快照。

运行（在 backend/ 下）：
  python -m ml.planner.build_eval_set --count 50 --difficulty standard --seed 1000 --output ml/planner/eval/records.jsonl
  python -m ml.planner.build_eval_set --count 50 --difficulty hard --seed 2000 --output ml/planner/eval_hard/records.jsonl
支持 --resume：跳过已有 record_id。
"""
import argparse
import asyncio
import json
from pathlib import Path

from app.agents.subgraphs.weather import weather_subgraph
from app.agents.subgraphs.hotel import hotel_subgraph
from app.agents.subgraphs.poi import poi_subgraph
from app.agents.supervisor import _date_range
from app.planner.context import build_planner_context
from app.services.amap_tools import init_amap_tools, close_amap_tools
from ml.planner.requestgen import iter_controlled_requests

# 单条样本拍快照的超时上限：AMAP MCP stdio 调用本身无超时，个别请求会 wedge，
# 用 wait_for 兜底，超时即抛出让调用方跳过该样本，避免整个串行循环被一个挂死的调用卡死。
SNAPSHOT_TIMEOUT = 120


async def snapshot_context(req) -> dict:
    weather, hotel, poi = await asyncio.wait_for(asyncio.gather(
        weather_subgraph.ainvoke({"city": req.city,
                                  "travel_dates": _date_range(req.start_date, req.travel_days),
                                  "raw_result": "", "weather_result": []}),
        hotel_subgraph.ainvoke({"city": req.city, "accommodation_pref": req.accommodation,
                                "budget_level": "mid", "raw_result": "", "hotel_result": []}),
        poi_subgraph.ainvoke({"city": req.city, "preferences": req.preferences,
                              "travel_days": req.travel_days, "raw_result": "", "poi_result": []}),
    ), timeout=SNAPSHOT_TIMEOUT)
    return build_planner_context(
        req, weather["weather_result"], hotel["hotel_result"], poi["poi_result"])


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--difficulty", choices=["standard", "hard"], default="standard")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="并发拍快照数（每个再并发3个子图，勿过高以免压垮 MCP）")
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out.exists():
        with open(out, encoding="utf-8") as f:
            done = {json.loads(line)["record_id"] for line in f if line.strip()}

    requests = iter_controlled_requests(args.count, args.difficulty, seed=args.seed)
    todo = [(i, req, f"{args.difficulty}_{args.seed}_{i:04d}")
            for i, req in enumerate(requests) if f"{args.difficulty}_{args.seed}_{i:04d}" not in done]
    total = len(todo)

    sem = asyncio.Semaphore(args.workers)

    async def snapshot_one(req, record_id):
        async with sem:
            try:
                return record_id, req, await snapshot_context(req), None
            except Exception as e:
                return record_id, req, None, str(e)

    await init_amap_tools()
    n = 0
    try:
        with open(out, "a", encoding="utf-8") as f:
            for coro in asyncio.as_completed([snapshot_one(req, rid) for _, req, rid in todo]):
                record_id, req, context, err = await coro
                n += 1
                if err is not None:
                    print(f"❌ {record_id} ({n}/{total}): {err}")
                    continue
                counts = context["tool_snapshot"]["candidate_counts"]
                if counts["hotels"] == 0 or counts["attractions"] == 0:
                    print(f"⚠️ {record_id} ({n}/{total}) 候选不足 {counts}，跳过")
                    continue
                f.write(json.dumps({
                    "record_id": record_id, "difficulty": args.difficulty,
                    "request": req.model_dump(), "context": context,
                }, ensure_ascii=False) + "\n")
                f.flush()
                print(f"✅ {record_id} ({n}/{total}) candidates={counts}")
    finally:
        await close_amap_tools()


if __name__ == "__main__":
    asyncio.run(main())
