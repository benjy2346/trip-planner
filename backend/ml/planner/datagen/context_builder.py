"""训练数据生成专用 PlannerContextBuilder：历史天气 override + 限流。

线上系统只走 `app.planner.context.PlannerContextBuilder`（高德短期天气预报）。
本模块只给 `backend/ml/planner/datagen` 下的训练数据生成脚本使用：当行程完全
发生在过去时，改用 Open-Meteo Archive 历史天气覆盖 trip_weather，让训练样本
"有天气就遵循"；否则原样回退父类的高德天气逻辑。

行为对齐 helloagents 参考实现
（training/scripts/planner/data/generate_sft_data.py:380-425）。
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict

from app.models.schemas import TripRequest
from app.planner.context import PlannerContextBuilder

from .historical_weather import fetch_historical_trip_weather, is_past_trip

_open_meteo_lock = threading.Lock()
_last_open_meteo_call_at = 0.0


def throttle_open_meteo_call() -> None:
    """限制 Open-Meteo 历史天气请求速率，避免高并发造数触发 429。"""
    global _last_open_meteo_call_at
    min_interval = float(os.getenv("OPEN_METEO_MIN_INTERVAL_SECONDS", "0.4"))
    with _open_meteo_lock:
        now = time.monotonic()
        wait_seconds = min_interval - (now - _last_open_meteo_call_at)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        _last_open_meteo_call_at = time.monotonic()


class DataGenPlannerContextBuilder(PlannerContextBuilder):
    """训练数据专用 PlannerContextBuilder。

    线上系统不需要历史天气；这里仅在过去完整行程时绕开高德天气，
    直接使用 Open-Meteo Archive，避免浪费高德天气接口额度。
    """

    def __init__(self, amap_api_key: str, historical_weather_provider: str):
        super().__init__(amap_api_key)
        self.historical_weather_provider = historical_weather_provider

    def _collect_weather_snapshot(self, request: TripRequest) -> Dict[str, Any]:
        if self.historical_weather_provider == "open-meteo" and is_past_trip(request):
            for attempt in range(1, 4):
                try:
                    throttle_open_meteo_call()
                    rows = fetch_historical_trip_weather(request)
                    if rows:
                        return {
                            "tool_snapshot": {
                                "available_weather": rows,
                                "trip_weather": rows,
                            },
                            "status": self._tool_status(
                                True,
                                f"open_meteo_archive={len(rows)}, covered_trip_days={len(rows)}/{request.travel_days}",
                            ),
                        }
                except Exception as exc:  # noqa: BLE001
                    if attempt >= 3:
                        print(f"⚠️  Open-Meteo历史天气失败，回退默认天气链路: {exc}", flush=True)
                        break
                    time.sleep(1.5 * attempt)
        return super()._collect_weather_snapshot(request)
