from app.agents.supervisor import create_supervisor_graph
from app.agents.chat_graph import create_chat_graph

_supervisor_graph = None
_chat_graph = None


def init_supervisor_graph(checkpointer) -> None:
    global _supervisor_graph
    _supervisor_graph = create_supervisor_graph(checkpointer)


def get_supervisor_graph():
    if _supervisor_graph is None:
        raise RuntimeError("supervisor_graph not initialized")
    return _supervisor_graph


def init_chat_graph(checkpointer) -> None:
    global _chat_graph
    _chat_graph = create_chat_graph(checkpointer)


def get_chat_graph():
    if _chat_graph is None:
        raise RuntimeError("chat_graph not initialized")
    return _chat_graph
