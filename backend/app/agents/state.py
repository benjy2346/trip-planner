import operator
from typing import TypedDict, Annotated, Optional, NotRequired
from langgraph.graph.message import add_messages
from app.models.schemas import TripRequest, TripPlan, WeatherInfo, Hotel, Attraction


class SupervisorState(TypedDict):
    trip_request: TripRequest
    messages: Annotated[list, add_messages]
    trip_plan: Optional[TripPlan]
    summary: str
    weather_outputs: Annotated[list, operator.add]
    hotel_outputs: Annotated[list, operator.add]
    poi_outputs: Annotated[list, operator.add]
    intent: NotRequired[str]
    # 本次调用走生成还是对话分支。由调用方显式指定，不靠"有没有 trip_plan"去猜——
    # 已有行程时用户仍可能要求重新生成。
    mode: NotRequired[str]


class WeatherSubState(TypedDict):
    city: str
    travel_dates: list[str]
    raw_result: str
    weather_result: list[WeatherInfo]


class HotelSubState(TypedDict):
    city: str
    accommodation_pref: str
    budget_level: str
    raw_result: str
    hotel_result: list[Hotel]


class POISubState(TypedDict):
    city: str
    preferences: list[str]
    travel_days: int
    raw_result: str
    poi_result: list[Attraction]
