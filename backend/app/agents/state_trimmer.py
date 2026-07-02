from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

WINDOW_SIZE = 4


def trim_state(state: dict, llm) -> dict:
    messages: list[BaseMessage] = state.get("messages", [])
    if len(messages) <= WINDOW_SIZE:
        return state

    to_summarize = messages[:-WINDOW_SIZE]
    kept = messages[-WINDOW_SIZE:]
    existing_summary = state.get("summary", "")

    summary_prompt = [
        SystemMessage(content=(
            "请将以下对话历史压缩为简洁摘要，保留关键行程偏好和修改意图，"
            "丢弃无关闲聊。摘要用中文，不超过 200 字。"
        )),
        HumanMessage(content=(
            f"已有摘要：{existing_summary}\n\n"
            "新增对话：\n" +
            "\n".join(f"{m.type}: {m.content}" for m in to_summarize)
        )),
    ]
    new_summary = llm.invoke(summary_prompt)

    return {
        **state,
        "messages": kept,
        "summary": new_summary.content,
    }
