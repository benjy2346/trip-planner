# Teacher SFT 数据生成记录（T7）

DeepSeek(`deepseek-chat`) 作 teacher，子图快照 → 生成 → 规则**硬过滤**（结构/占位词/预算）落盘 `records.jsonl`；训练导出时再按 soft 口径**过滤午晚餐重复**（见 `export_llamafactory.meal_repeat_count`）。数据经 `eval_signature` 与冻结评测集零重叠。

## 各 run 汇总

| run (slug) | seed | 请求 | 处理 | hardpass | 高德失败 | 重复过滤后干净 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 260703_smoke20 | 9000 | 20 | 19 | 15 | 1 | **14** |
| 260704_batch100 | 9100 | 100 | 92 | 69 | 8 | **59** |
| 260705_batch1400 | 9200 | 1400 | ~1269 | 890 | ~132 | **710** |
| **合计** | | 1520 | | 974 | | **783** |

导出：`export_llamafactory --runs 260703_smoke20 260704_batch100 260705_batch1400` → **train 744 / val 39**（val_ratio 0.05，seed 42）。校验：train 中餐厅重复记录 0/744。

## 质量画像与决策

- **硬过滤主要失败**：`景点数 0 不在 1-3`（POI 快照候选不足的请求，teacher 无景点可选；`data_gen` 不像 `build_eval_set` 跳过空候选，故产出空景点天被硬过滤挡掉，仅损产出率）＋少量 `餐饮占位词`。
- **软口径过滤**：hardpass 记录里 batch100 有 14%、batch1400 有 20% 带**午晚餐店名重复**（DeepSeek 多天行程复用同餐厅，与基线 hard soft_pass 20.5% 一致）。这正是微调要治的病，**训练集一律剔除**（用户 2026-07-11 拍板）。
- **抽查**（standard + hard 各若干）：酒店连续同店 + 末日 null、景点 1-3、门票×人数、hard 预算不超 —— 均正确。

## 稳定性事故（记录备查）

- batch1400 首次用 Claude Code `run_in_background` 启动，会话空闲跨回合后被 harness 回收（~200/1400）。改用 `nohup caffeinate -is ... & disown` 脱钩重跑，靠同 slug 断点续跑补齐。
- 续跑跑到 ~1220/1270 后**挂死在最后 straggler**（某网络调用无有效超时，空转 ~20h，0% CPU）。数据已够（890 hardpass），SIGTERM 清理后落定。
- 教训：重活要么脱钩+防休眠本地跑，要么上远程机 tmux；`data_gen` 的子图/teacher 调用需要更硬的超时（后续可加）。

## 成本（teacher token，估）

累计约 prompt ~2.5M / completion ~2.9M（DeepSeek，成本约 1–2 美元量级）。各 run manifest 内含精确 usage。
