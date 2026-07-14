from unittest.mock import patch

from app.planner.amap import AmapPlannerClient


def _fake_amap_response():
    # 高德 place/text 结构化返回：pois[].location = "lng,lat"
    return {"status": "1", "pois": [
        {"name": "西湖", "address": "西湖区", "location": "120.15,30.25",
         "type": "风景名胜", "biz_ext": {"rating": "4.7"}},
    ]}


def test_search_keywords_parses_structured_location(tmp_path):
    client = AmapPlannerClient(api_key="TESTKEY", cache_dir=tmp_path)
    with patch.object(client, "get", return_value=_fake_amap_response()):
        rows = client.search_keywords("杭州", ["西湖"], source_role="scenic", limit=5)
    assert rows, "should return candidates"
    loc = rows[0]["location"]
    assert loc["longitude"] == 120.15 and loc["latitude"] == 30.25  # 真坐标，非 0,0
    assert rows[0]["name"] == "西湖"
