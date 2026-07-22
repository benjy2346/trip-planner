# DeepSeek 基线（Planner SFT 对比基准）

Teacher = 线上兜底模型 `deepseek-chat`，在两套**冻结评测集**上跑 `rule_eval`，作为后续 qwen-base / SFT 模型对比的基准线。评测集从此不再重采样。

- 生成日期：2026-07-11
- 命令：`python -m ml.planner.rule_eval --records <eval> --base-url https://api.deepseek.com/v1 --model deepseek-chat --api-key-env DEEPSEEK_API_KEY --output-dir <dir>`
- 参数：temperature 0.2 / max_tokens 8192 / workers 4

## 结果

| 指标 | standard (44) | hard (39) |
| --- | ---: | ---: |
| json_ok | 100.0% | 100.0% |
| schema_ok | 100.0% | 100.0% |
| **hard_pass** | **72.7%** | **79.5%** |
| **soft_pass** | **63.6%** | **20.5%** |
| budget_ok | 97.7% | 82.1% |
| 午晚餐平均重复 | 0.39 | 3.38 |

## 失败画像（hardpass 违规聚合）

| 违规类型 | standard | hard |
| --- | ---: | ---: |
| 餐饮占位词 | 31 | 42 |
| 景点数不在 1-3 | 0 | 3 |

- **主要失败 = 餐饮占位词**：DeepSeek 大量产出 `酒店早餐` / `X附近餐厅` / `X附近小吃店` 这类不具体的餐饮，正是 prompt/校验明令禁止的占位。这是微调要重点纠正的方向。
- **hard 集 soft_pass 崩到 20.5%、午晚餐重复 3.38**：多天行程里严重复用同一家餐厅（5-6 天几乎每天午/晚餐同名）。SFT 的核心增益点。
- **budget_ok**：standard 97.7% 基本无压力；hard 82.1% 有硬预算约束时偶尔超支。

## 已知校验边角（不影响基线结论）

- `MEAL_PLACEHOLDER_RE` 的 `^无` 分支误伤真实店名 `无味舒食(思明店)`（厦门素食连锁，1 例）。占位过滤对 SFT 数据/评测是保守偏严，可接受；如后续误伤增多再收窄该分支。

## 达标目标（对照本基线）

SFT 模型需：hardpass standard ≥ 85% 且 hard ≥ 70%；比 qwen-base 高 ≥ 20pp；**不低于本 DeepSeek 基线 −5pp**（即 standard ≥ 67.7%、hard ≥ 74.5%）。
