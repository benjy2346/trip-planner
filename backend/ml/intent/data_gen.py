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

# 说明中的示例为“风格示范”，用于引导覆盖面，均为改写措辞，切勿照抄评测集句子
_LABEL_DESC = {
    "query_weather": (
        "询问行程期间的天气、气温、冷热、是否下雨、要不要带伞或加衣。"
        "风格示范：出门要不要加件衣、白天晒不晒、这周末下不下雨、早晚温差大吗"
    ),
    "query_attraction": (
        "询问要去的景点、白天玩什么、游览/参观安排、有没有好玩的或值得打卡的地方，"
        "以及吃什么、去哪吃、有什么好吃的、餐厅/馆子推荐（吃与玩都算这一类）。"
        "风格示范：白天带我去哪逛、这地方值不值得玩、附近有啥好吃的小馆、"
        "哪家店味道正、第二天有啥可看的、有没有适合拍照的角落、想找点小吃"
    ),
    "query_hotel": (
        "询问住的酒店/住宿信息、酒店位置与评分、住得远不远，以及住宿花费和整趟行程的预算/总开销。"
        "风格示范：订的哪家宾馆、睡的地方贵不贵、离市中心多远、这趟大概得花多少、钱够不够花"
    ),
    "plan_change": (
        "要求系统‘改动行程’：生成新行程，或修改、增删、替换、调整已有安排"
        "（把某天改掉、去掉某个点、多加一天、换住处、行程排太满想调松）。"
        "关键是让系统动手改，而不是查询信息。"
        "风格示范：这天别排那么满、把那个点去掉、多留一天吧、帮我换个近点的住处、重新给我排一版"
    ),
    "other": (
        "问候、感谢、客套、闲聊，或与本次旅行行程完全无关的话题。"
        "风格示范：在忙啥呢、你叫什么、辛苦啦、随便聊两句、今天行情怎么样、给我讲个段子"
    ),
}


def build_prompt(label: str, n: int) -> str:
    return (
        f"你在为一个中文旅行助手构造意图分类训练数据。\n"
        f"意图类别「{label}」：{_LABEL_DESC[label]}\n\n"
        f"重要区分：query_* 是‘询问/查看已有行程里的信息’；plan_change 是‘要求改动行程’。"
        f"例如问‘第二天参观什么’属于 query_attraction（在问信息），"
        f"而说‘第二天改成爬山’属于 plan_change（在要求改动）。\n\n"
        f"请生成 {n} 条属于「{label}」意图的中文用户消息：口语化、长短不一，"
        f"多来一些很短的日常问法（如三五个字的），也要有完整句子；"
        f"覆盖不同问法与场景，句式不要雷同，避免重复。\n"
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
