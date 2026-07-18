"""grounded 生成 runner：对任意 OpenAI 兼容端点（vLLM /v1 或 DeepSeek）跑冻结评测集，
只写 generations.jsonl（record_id + output），不打分。打分见 rule_metrics.py。

与旧 rule_eval.py 的区别：用 build_grounded_planner_messages(compact_planner_context)
（train/serve parity），且生成与打分解耦（可换端点重跑生成，用同一 rule_metrics 打分）。

运行（在 backend/ 下）：
  # 微调/base 端点（vLLM）
  python -m ml.planner.eval.generate --records ml/planner/eval/records.jsonl \
    --base-url http://127.0.0.1:8000/v1 --model qwen-ft --api-key-env VLLM_API_KEY \
    --output-dir runs_eval/ft_standard
  # DeepSeek
  python -m ml.planner.eval.generate --records ml/planner/eval/records.jsonl \
    --base-url https://api.deepseek.com/v1 --model deepseek-chat \
    --api-key-env DEEPSEEK_API_KEY --output-dir runs_eval/deepseek_standard
"""
import argparse
import asyncio
import json
import os
from pathlib import Path

from langchain_openai import ChatOpenAI

from app.planner.context import build_grounded_planner_messages


def messages_for(record: dict) -> list:
    """该记录的 grounded 输入消息——直接用记录里已烤好的 compact_planner_context，
    不重新 compact（保证与训练/teacher 见到的输入同源）。"""
    return build_grounded_planner_messages(record["compact_planner_context"])


def write_generations(results: list, output_dir: str) -> int:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_dir / "generations.jsonl", "w", encoding="utf-8") as f:
        for item in results:
            if isinstance(item, Exception):
                print(f"❌ 生成失败: {item}", flush=True)
                continue
            record, text = item
            f.write(json.dumps({"record_id": record["record_id"], "output": text},
                               ensure_ascii=False) + "\n")
            n += 1
    return n


async def _generate(llm: ChatOpenAI, record: dict, sem: asyncio.Semaphore) -> tuple:
    async with sem:
        resp = await llm.ainvoke(messages_for(record))
        return record, resp.content


async def _run(args) -> None:
    with open(args.records, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    llm = ChatOpenAI(base_url=args.base_url, api_key=os.environ.get(args.api_key_env, "EMPTY"),
                     model=args.model, temperature=args.temperature,
                     max_tokens=args.max_tokens, timeout=600)
    sem = asyncio.Semaphore(args.workers)
    results = await asyncio.gather(
        *[_generate(llm, r, sem) for r in records], return_exceptions=True)
    n = write_generations(results, args.output_dir)
    print(f"✅ 写出 {n}/{len(records)} 条 generations 到 {args.output_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
