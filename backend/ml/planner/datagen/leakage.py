"""泄漏 guard：训练数据请求签名与冻结评测集比对，防止训练/评测重叠。"""
import hashlib
import json

from app.models.schemas import TripRequest


def eval_signature(req: TripRequest) -> str:
    """请求语义签名，与 ml/planner/requestgen.py::eval_signature 同算法。"""
    b = req.budget_constraint
    key = "|".join([
        req.city, req.start_date, str(req.travel_days),
        str(req.party.adults), str(req.party.children), str(req.party.elders),
        str(b.amount) if b else "none", b.strictness if b else "none",
        ",".join(sorted(req.preferences)),
    ])
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def load_eval_signatures(paths: list[str]) -> set[str]:
    """读取冻结评测记录文件（jsonl，每行含 request 字段），返回其请求签名集合。"""
    sigs: set[str] = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                req = TripRequest(**record["request"])
                sigs.add(eval_signature(req))
    return sigs
