"""用 LLM 合成意图分类训练数据。

运行：cd backend && python -m ml.intent.data_gen
输出：backend/ml/intent/data/train.jsonl
"""
import asyncio
import json
from pathlib import Path

from app.agents.intent_labels import INTENT_LABELS
from app.agents.llm_router import acall_with_fallback
from langchain_core.messages import SystemMessage, HumanMessage

PER_LABEL = 300
BATCH_SIZE = 60  # 每次 LLM 请求生成的条数（单次请求 300 条会超过 30s 超时）
MAX_ROUNDS_PER_LABEL = 10  # 安全上限，避免模型持续返回重复而死循环
OUT_PATH = Path(__file__).resolve().parent / "data" / "train.jsonl"

_LABEL_DESC = {
    "query_weather": "查询行程期间的天气、气温、是否下雨、穿衣建议",
    "query_attraction": "查询景点、游玩安排、游览时长、餐饮/吃饭推荐",
    "query_hotel": "查询住宿/酒店信息、酒店位置评分、住宿费用与总预算",
    "plan_change": "生成新行程，或修改、增删、调整已有行程的任何内容",
    "other": "问候、感谢、闲聊，或与旅行行程完全无关的内容",
}


def build_prompt(label: str, n: int) -> str:
    return (
        f"你在为一个中文旅行助手构造意图分类训练数据。\n"
        f"意图类别「{label}」的含义：{_LABEL_DESC[label]}。\n"
        f"请生成 {n} 条属于该意图的、多样化的中文用户消息，"
        f"口语化、长短不一、涵盖不同问法，避免重复。\n"
        f"每行一条，只输出消息文本，不要编号、不要引号、不要多余说明。"
    )


def dedup(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"])
            out.append(r)
    return out


async def _gen_label(label: str, n: int) -> list[dict]:
    """分批向 LLM 请求，累积去重直到达到 n 条（单次请求 n 太大会超时）。"""
    seen: set[str] = set()
    rows: list[dict] = []
    rounds = 0
    while len(rows) < n and rounds < MAX_ROUNDS_PER_LABEL:
        rounds += 1
        resp = await acall_with_fallback([
            SystemMessage(content="你是数据标注助手，只输出要求的内容。"),
            HumanMessage(content=build_prompt(label, BATCH_SIZE)),
        ])
        for line in resp.content.splitlines():
            text = line.strip().strip("\"'　 ")
            if text and text not in seen:
                seen.add(text)
                rows.append({"text": text, "label": label})
    return rows[:n]


async def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    # 逐类写入并 flush，单类失败不会丢失已生成的其他类
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for label in INTENT_LABELS:
            rows = dedup(await _gen_label(label, PER_LABEL))
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            total += len(rows)
            print(f"{label}: 写入 {len(rows)} 条（累计 {total}）")
    print(f"完成，共写入 {total} 条到 {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
