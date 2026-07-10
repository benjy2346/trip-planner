"""受控请求分布生成器：评测集与训练数据共用，靠 seed 区分与复现。"""
import hashlib
import random
from datetime import date, timedelta
from app.models.schemas import TripRequest, PartyInfo, BudgetConstraint

CITIES = ["北京", "上海", "杭州", "成都", "西安", "广州", "南京", "重庆", "苏州", "厦门"]
PREFS = [["历史文化"], ["美食"], ["自然风光"], ["美食", "城市地标"],
         ["博物馆", "历史文化"], ["亲子"], ["购物", "美食"]]
TRANSPORT = ["公共交通", "打车", "自驾"]
ACCOM = ["经济型酒店", "舒适型酒店", "高档型酒店"]
HARD_FREE_TEXT = ["不吃辣，行程别太赶", "有老人同行，少爬山", "带孩子，需要亲子友好",
                  "素食为主", "不想去人太多的地方"]
_BASE_DATE = date(2026, 8, 1)


def iter_controlled_requests(count: int, difficulty: str = "standard", seed: int = 0) -> list[TripRequest]:
    rng = random.Random(seed)
    out = []
    for i in range(count):
        city = CITIES[(i + rng.randrange(len(CITIES))) % len(CITIES)]
        start = _BASE_DATE + timedelta(days=rng.randrange(60))
        if difficulty == "hard":
            days = rng.randint(4, 6)
            party = PartyInfo(adults=2, children=rng.randint(1, 2), elders=rng.randint(0, 1))
            budget = BudgetConstraint(
                amount=days * party.total * rng.choice([300, 400, 500]),
                budget_level="limited", strictness="hard")
            free_text = rng.choice(HARD_FREE_TEXT)
        else:
            days = rng.randint(2, 4)
            party = PartyInfo(adults=rng.randint(1, 2))
            budget = rng.choice([
                None,
                BudgetConstraint(amount=days * party.total * rng.choice([600, 800, 1000]),
                                 budget_level="comfortable", strictness="soft"),
            ])
            free_text = ""
        end = start + timedelta(days=days - 1)
        out.append(TripRequest(
            city=city,
            start_date=start.isoformat(), end_date=end.isoformat(), travel_days=days,
            transportation=rng.choice(TRANSPORT), accommodation=rng.choice(ACCOM),
            preferences=rng.choice(PREFS), free_text_input=free_text,
            party=party, budget_constraint=budget,
        ))
    return out


def eval_signature(req: TripRequest) -> str:
    """请求语义签名，用于训练数据与冻结评测集的防泄漏过滤。"""
    b = req.budget_constraint
    key = "|".join([
        req.city, req.start_date, str(req.travel_days),
        str(req.party.adults), str(req.party.children), str(req.party.elders),
        str(b.amount) if b else "none", b.strictness if b else "none",
        ",".join(sorted(req.preferences)),
    ])
    return hashlib.sha1(key.encode()).hexdigest()[:16]
