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
    resp = await acall_with_fallback([
        SystemMessage(content="你是数据标注助手，只输出要求的内容。"),
        HumanMessage(content=build_prompt(label, n)),
    ])
    rows = []
    for line in resp.content.splitlines():
        text = line.strip().strip("\"'　 ")
        if text:
            rows.append({"text": text, "label": label})
    return rows


async def main() -> None:
    all_rows: list[dict] = []
    for label in INTENT_LABELS:
        rows = await _gen_label(label, PER_LABEL)
        print(f"{label}: 生成 {len(rows)} 条")
        all_rows.extend(rows)
    all_rows = dedup(all_rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"写入 {len(all_rows)} 条到 {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
