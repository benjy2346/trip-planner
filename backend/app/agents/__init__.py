from app.agents.graph import MODE_CHAT, MODE_GENERATE, create_trip_graph

_trip_graph = None


def init_trip_graph(checkpointer) -> None:
    global _trip_graph
    _trip_graph = create_trip_graph(checkpointer)


def get_trip_graph():
    if _trip_graph is None:
        raise RuntimeError("trip_graph not initialized")
    return _trip_graph


__all__ = [
    "MODE_CHAT",
    "MODE_GENERATE",
    "create_trip_graph",
    "get_trip_graph",
    "init_trip_graph",
]
