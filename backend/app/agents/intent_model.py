"""微调意图分类模型的进程内单例推理封装。"""
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "intent_classifier"

_pipeline = None


class IntentModelUnavailable(Exception):
    """模型目录不存在或加载失败。"""


def _load_pipeline():
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model=str(MODEL_DIR),
        tokenizer=str(MODEL_DIR),
        top_k=None,
    )


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not MODEL_DIR.exists():
            raise IntentModelUnavailable(f"model dir not found: {MODEL_DIR}")
        _pipeline = _load_pipeline()
    return _pipeline


def predict(text: str) -> tuple[str, float]:
    scores = get_pipeline()(text)[0]
    best = max(scores, key=lambda x: x["score"])
    return best["label"], float(best["score"])


def reset() -> None:
    """清空单例（测试用）。"""
    global _pipeline
    _pipeline = None
