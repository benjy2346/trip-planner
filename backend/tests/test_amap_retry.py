"""高德客户端重试策略测试：错误分类 + 指数退避 + jitter。"""

import httpx
import pytest

from app.planner.amap import (
    AMAP_RETRY_ATTEMPTS,
    AMAP_RETRY_MAX_DELAY,
    AmapPermanentError,
    AmapTransientError,
    AmapPlannerClient,
    _retry_delay,
)


class _Resp:
    """最小 httpx.Response 替身。"""

    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)

    def json(self):
        return self._payload


@pytest.fixture
def client(tmp_path):
    return AmapPlannerClient(api_key="test-key", cache_dir=tmp_path)


def _patch(monkeypatch, responses):
    """让 httpx.get 依次返回 responses；返回调用计数器。"""
    calls = {"n": 0}

    def fake_get(*args, **kwargs):
        item = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("app.planner.amap.time.sleep", lambda _: None)
    monkeypatch.setattr(AmapPlannerClient, "_wait_for_amap_slot", lambda self: None)
    return calls


def test_permanent_business_error_fails_fast(monkeypatch, client):
    """KEY 无效属于永久错误：只调一次，不浪费重试。"""
    calls = _patch(monkeypatch, [_Resp({"status": "0", "info": "INVALID_USER_KEY", "infocode": "10001"})])

    with pytest.raises(AmapPermanentError):
        client.get("/place/text", {"keywords": "西湖"})

    assert calls["n"] == 1


def test_invalid_params_fails_fast(monkeypatch, client):
    """参数错误同样不该重试。"""
    calls = _patch(monkeypatch, [_Resp({"status": "0", "info": "INVALID_PARAMS", "infocode": "20000"})])

    with pytest.raises(AmapPermanentError):
        client.get("/place/text", {"keywords": "西湖"})

    assert calls["n"] == 1


def test_rate_limited_is_retried(monkeypatch, client):
    """10004 限频属于瞬时错误：重试到上限。"""
    calls = _patch(monkeypatch, [_Resp({"status": "0", "info": "ACCESS_TOO_FREQUENT", "infocode": "10004"})])

    with pytest.raises(RuntimeError, match="已重试"):
        client.get("/place/text", {"keywords": "西湖"})

    assert calls["n"] == AMAP_RETRY_ATTEMPTS


def test_timeout_is_retried_then_succeeds(monkeypatch, client):
    """超时可重试；恢复后应返回正常数据。"""
    ok = _Resp({"status": "1", "pois": []})
    calls = _patch(monkeypatch, [httpx.TimeoutException("timeout"), ok])

    data = client.get("/place/text", {"keywords": "西湖"})

    assert data["status"] == "1"
    assert calls["n"] == 2


def test_server_error_is_retried(monkeypatch, client):
    """5xx 属于瞬时错误。"""
    calls = _patch(monkeypatch, [_Resp({}, status_code=503)])

    with pytest.raises(RuntimeError, match="已重试"):
        client.get("/place/text", {"keywords": "西湖"})

    assert calls["n"] == AMAP_RETRY_ATTEMPTS


def test_client_error_fails_fast(monkeypatch, client):
    """404 这类 4xx 是请求本身的问题，不重试。"""
    calls = _patch(monkeypatch, [_Resp({}, status_code=404)])

    with pytest.raises(httpx.HTTPStatusError):
        client.get("/place/text", {"keywords": "西湖"})

    assert calls["n"] == 1


def test_retry_delay_grows_and_is_capped():
    """退避随重试次数指数增长，并被 MAX_DELAY 封顶。"""
    assert _retry_delay(0) < _retry_delay(3)
    assert all(_retry_delay(20) <= AMAP_RETRY_MAX_DELAY for _ in range(20))


def test_retry_delay_has_jitter():
    """同一 attempt 的延迟应有抖动，否则并发重试会撞在一起。"""
    delays = {_retry_delay(2) for _ in range(30)}
    assert len(delays) > 1
