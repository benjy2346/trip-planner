"""日志配置与应用启动冒烟测试。

应用启动冒烟这条尤其重要：此前 CI 全绿但 `app.api.main` 根本导入不了
（路由依赖已被移除的 hello_agents），因为没有任何测试导入过 FastAPI app。
"""

import logging

import pytest

from app.logging_config import (
    get_logger,
    get_request_id,
    new_request_id,
    set_request_id,
    setup_logging,
)


def test_app_imports_and_registers_expected_routes():
    """应用必须能被导入，且暴露前端实际调用的端点。"""
    from app.api.main import app

    paths = {r.path for r in app.routes if hasattr(r, "path")}
    for expected in {
        "/api/trip/plan",
        "/api/poi/photo",
        "/api/chat/modify",
        "/api/chat/history/{user_id}",
    }:
        assert expected in paths, f"缺少路由 {expected}"


def test_no_print_left_in_app_package():
    """生产代码不该用 print：日志要有级别、时间戳，且能被采集。"""
    from pathlib import Path

    app_dir = Path(__file__).resolve().parents[1] / "app"
    offenders = [
        f"{path.relative_to(app_dir.parent)}:{i}"
        for path in app_dir.rglob("*.py")
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if line.lstrip().startswith("print(")
    ]
    assert not offenders, f"仍有 print 调用: {offenders}"


def test_request_id_flows_into_log_records(caplog):
    """同一请求的日志应带上同一个 request id，便于串联链路。"""
    setup_logging("INFO")
    logger = get_logger("app.test")
    rid = new_request_id()
    set_request_id(rid)

    with caplog.at_level(logging.INFO, logger="app.test"):
        logger.info("处理中")

    assert get_request_id() == rid
    assert any(getattr(r, "request_id", None) == rid for r in caplog.records)


def test_new_request_id_is_unique():
    assert len({new_request_id() for _ in range(50)}) == 50


def test_setup_logging_respects_level():
    setup_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING
    setup_logging("INFO")
    assert logging.getLogger().level == logging.INFO


def test_setup_logging_is_idempotent():
    """重复调用不应堆积 handler。"""
    setup_logging("INFO")
    before = len(logging.getLogger().handlers)
    setup_logging("INFO")
    assert len(logging.getLogger().handlers) == before
