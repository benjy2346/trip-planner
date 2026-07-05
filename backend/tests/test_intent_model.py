from unittest.mock import patch, MagicMock
import pytest
from app.agents import intent_model


def teardown_function():
    intent_model.reset()


def test_predict_returns_label_and_confidence():
    fake_pipe = MagicMock(return_value=[[
        {"label": "query_weather", "score": 0.1},
        {"label": "query_hotel", "score": 0.82},
        {"label": "other", "score": 0.08},
    ]])
    with patch("app.agents.intent_model.get_pipeline", return_value=fake_pipe):
        label, conf = intent_model.predict("住哪个酒店")
    assert label == "query_hotel"
    assert conf == pytest.approx(0.82)


def test_missing_model_dir_raises_unavailable(monkeypatch):
    from pathlib import Path
    intent_model.reset()
    # 指向不存在的路径，无需 mock Path.exists（pathlib.Path 是 __slots__，不可实例级打补丁）
    monkeypatch.setattr(intent_model, "MODEL_DIR", Path("/nonexistent/intent_classifier"))
    with pytest.raises(intent_model.IntentModelUnavailable):
        intent_model.get_pipeline()
