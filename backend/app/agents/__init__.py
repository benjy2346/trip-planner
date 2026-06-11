from app.agents.supervisor import create_supervisor_graph

_graph = None


def init_supervisor_graph(checkpointer) -> None:
    global _graph
    _graph = create_supervisor_graph(checkpointer)


def get_supervisor_graph():
    if _graph is None:
        raise RuntimeError("supervisor_graph not initialized; call init_supervisor_graph() at startup")
    return _graph
