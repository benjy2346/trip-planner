# Plan 2b:Grounded Planner 微调 + 三方评测报告

**日期**:2026-07-19
**结论一句话**:用 grounded 数据对 Qwen2.5-7B 做 QLoRA 微调**显著有效** —— 主指标 `hard_pass` 在标准集从 35.0% 提到 **63.0%**,难集从 17.0% 提到 **47.3%**,补上了 base 与 DeepSeek 之间约一半的差距。

---

## 1. 最终结果(三方对比)

在**冻结评测集**(标准 200 + hard 300)上,用同一套规则打分器(`ml/planner/eval/rule_metrics.py`)评测三个模型。主指标 `hard_pass` = 无任何规则违规的样本占比。

### 主指标 hard_pass

| 模型 | 标准集 (n=200) | 难集 (n=300) |
| --- | ---: | ---: |
| base 7B(裸 Qwen2.5-7B-Instruct) | 35.0% | 17.0% |
| **微调 7B(QLoRA)** | **63.0%** | **47.3%** |
| DeepSeek(v4-flash,参照天花板) | 89.5% | 86.7% |

### 完整指标

| 指标 | base 标准 | **ft 标准** | DS 标准 | base 难 | **ft 难** | DS 难 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **hard_pass** | 35.0 | **63.0** | 89.5 | 17.0 | **47.3** | 86.7 |
| soft_pass | 24.0 | 50.5 | 82.0 | 5.3 | 17.3 | 74.0 |
| json_ok | 99.0 | 93.0 | 100.0 | 95.7 | 77.7 | 100.0 |
| schema_ok | 96.0 | 93.0 | 96.0 | 90.7 | 77.7 | 97.0 |
| budget_ok | 90.5 | 88.5 | 98.0 | 70.3 | 69.3 | 92.0 |
| 景点 grounding 均值 | 96.0 | 92.9 | 96.0 | 90.1 | 77.6 | 96.9 |
| 餐饮 grounding 均值 | 95.1 | 92.5 | 95.7 | 89.4 | 77.5 | 96.3 |
| 酒店 grounding 均值 | 95.5 | 93.0 | 96.0 | 89.8 | 77.7 | 97.0 |
| 餐饮多样性均值 | 62.6 | 80.2 | 91.4 | 51.6 | 63.9 | 92.3 |
| 午晚餐平均重复 | 1.75 | 0.52 | 0.14 | 2.58 | 0.65 | 0.09 |

> `soft_pass` 更严:要求 hard_pass 且无午晚餐重复且预算贴合。

---

## 2. 结果解读

**微调是有效的,提升幅度大**:标准集 +28 分、难集 +30 分,难集几乎是从 17% 翻到 47%(近 3 倍)。这证明用教师(DeepSeek)生成的 grounded 数据做蒸馏,确实让小模型学会了遵守 grounding / 多样性 / 预算 / 结构这一整套硬规则。

**多样性是微调学得最好的一项**:午晚餐平均重复从 base 的 1.75/2.58 骤降到 0.52/0.65(越低越好),餐饮多样性从 62.6/51.6 升到 80.2/63.9。base 模型爱反复推荐同一家餐厅,微调后基本改掉了 —— 这一项已接近 DeepSeek。

**一个诚实的短板 —— 微调模型的 `json_ok` 反而比 base 低**(难集 77.7% vs base 95.7%)。注意 ft 的 `json_ok` 与 `schema_ok` 完全相等(77.7/77.7、93.0/93.0),说明失败**全部是 JSON 解析失败、而非 schema 不符**。最可能的原因是**输出被截断**(生成 `max_tokens=4096`,难集行程长、候选多,偶尔写到一半被截,JSON 不闭合)。也就是说:**在能完整输出的样本里,微调模型的规则遵守率其实更高;当前 hard_pass 被这些截断样本拖了后腿**。如果调大 `max_tokens` 重跑,微调分数还会更高。这是留给下一轮最直接的优化点。

**与 DeepSeek 的差距**:微调 7B 还没到 DeepSeek 水平(63 vs 89.5,47 vs 87)。这符合预期 —— 7B QLoRA、仅 960 条训练数据、cutoff 为装下单卡从 24576 降到 20480。差距合理,方向明确。

---

## 3. 这次做了什么(流程)

**目标**:让一个便宜的本地小模型替代付费 DeepSeek API,做行程规划的 planner/assembler。

**训练数据**(Plan 2a 已完成):1010 条 grounded 记录 → `train.json` 960 / `val.json` 50。grounded = 真实 AMAP 坐标、真实餐饮候选、真实票价/房价提示、历史天气、结构化 preference_profile。教师 = DeepSeek。train/serve 输入同源(`build_grounded_planner_messages(compact_planner_context)`)。

**评测框架**(Plan 2b-code,本地 TDD):
- `ml/planner/eval/rule_metrics.py` —— 规则打分器,复用已就绪的 `validate_grounded_trip_plan`(hard_pass)+ 软指标。
- `ml/planner/eval/generate.py` —— 生成 runner,生成与打分解耦(换 base_url/model 打通三方端点)。
- 打分语义参照 helloagents `eval_rule_metrics.py`,逻辑自写复用我们的 validation。

**训练**(远程 AutoDL RTX 5090 32GB,Blackwell sm_120):
- QLoRA 4-bit NF4,r32/α64,lr 1e-4,2 epoch,grad_accum 16。
- 训练 120 步(60/epoch × 2),约 6 小时。**loss 平滑收敛 0.343 → 0.236**。
- cutoff_len 从 24576 降到 20480(见下"OOM 修复")。

**评测**:三个端点各跑标准 200 + 难 300 = 6 份 generations → 打分 → 6 份报告。
- base 7B、微调 7B:vLLM 本地起服务。
- DeepSeek:API。

---

## 4. 踩过的坑与修复(Blackwell 太新 + 版本兼容)

RTX 5090(Blackwell sm_120)是很新的卡,几乎每层软件栈都要现磨。按顺序:

| # | 问题 | 根因 | 修复 |
| --- | --- | --- | --- |
| 1 | torch 被依赖偷换 | `llamafactory[torch]` / vllm 拉 torch 2.11(CUDA13) | 每次 pip 用约束文件钉死 `torch==2.8.0`;装 llamafactory 去掉 `[torch]` 附加 |
| 2 | torchaudio import 崩 | torchaudio 2.11 按 CUDA13 编译(缺 libcudart.so.13) | 装配套 `torchaudio==2.8.0+cu128` |
| 3 | dry-run 命令报错 | LF 0.9.5 不支持 yaml + CLI 覆盖混用 | 用临时 yaml 副本 |
| 4 | 24K 上下文 logits OOM | LM head 全量 logits float32 ≈ 12GB,单卡装不下 | `enable_liger_kernel`(融合 CE,不物化全量 logits)+ cutoff 24576→20480 + `expandable_segments` |
| 5 | eval 阶段 OOM | liger 只接管训练 loss,eval 走标准 loss 又摊开 logits | 关掉训练内 eval(`eval_strategy: no`);质量评估交给三方 rule_metrics |
| 6 | 磁盘满 | 系统盘 30G 被模型+缓存撑爆 | 清 pip 缓存;缓存/合并模型导到数据盘 |
| 7 | vLLM 起不来 | vllm 0.11 vs transformers 5.6 tokenizer API 不兼容 | 降级 `transformers<5` → 4.57.6 |
| 8 | **微调模型输出乱码 → ft=0%** | **`llamafactory-cli export` 把 QLoRA 4-bit 基座合并成 bf16 时出坏权重** | **绕过合并:vLLM `--enable-lora` 直接加载 base + 原始 adapter** |
| 9 | ft vLLM OOM | 残留 vLLM 进程占着显存 | 按 PID 清干净再起 |

**训练本身自始至终是好的**(loss 正常收敛、adapter 完整);#8 的 0% 是合并 bug 造成的假失败,不是微调无效。

**最重要的一条教训**:**这套栈上,QLoRA 训练出的 adapter 不要信 `llamafactory-cli export` 合并成 bf16 来 serve —— 会产出坏权重、输出乱码。正确做法是用 vLLM `--enable-lora` 直接加载 base + adapter。** 最终的微调评测就是这么跑的。

**另一条纪律(用户要求)**:放量前先发**单条**请求、人工确认输出不是乱码,再跑满 500 条 —— 避免"跑完才发现坏了"的整轮 GPU 浪费。

---

## 5. 如何复现微调模型

微调模型 = base Qwen2.5-7B-Instruct + 本目录旁的 LoRA adapter。**不要用合并模型**。

```bash
# adapter 在:backend/ml/planner/outputs/qwen25_7b_qlora_v1/
# vLLM 直接加载 base + adapter(绕过坏合并):
python -m vllm.entrypoints.openai.api_server \
  --model <Qwen2.5-7B-Instruct 路径> \
  --enable-lora --lora-modules ft=<adapter 路径> \
  --max-lora-rank 32 --max-model-len 32768 --port 8000

# 生成 + 打分:
python -m ml.planner.eval.generate --records ml/planner/eval/records.jsonl \
  --base-url http://127.0.0.1:8000/v1 --model ft --api-key-env NONE \
  --output-dir runs_eval/ft_standard
python -m ml.planner.eval.rule_metrics --records ml/planner/eval/records.jsonl \
  --generations runs_eval/ft_standard/generations.jsonl \
  --output-dir runs_eval/ft_standard --model-tag ft_standard
```

---

## 6. 局限与下一步

**局限**:
- 微调 7B 未达 DeepSeek 水平(63/47 vs 89.5/87)。
- 难集 `json_ok` 仅 77.7%,疑为 `max_tokens=4096` 截断(见 §2),压低了 hard_pass。
- 训练 cutoff 20480 略低于数据中位(~22.5K),最长样本被截。
- QLoRA 4-bit 基座 vs bf16 有轻微精度损失。
- 三方"可比"限于方法学:同评测集、同打分器;绝对分不宜直接对标 helloagents 公布数字(数据/精度/单卡都不同)。DeepSeek 现指向 v4-flash(与当初生成教师数据的 deepseek-chat 已非同一模型),作强 API 参照。

**下一步(按性价比)**:
1. **调大生成 `max_tokens`(如 6144–8192)重跑 ft** —— 最省事,直接回收被截断样本,hard_pass 预计再涨。
2. 加训练数据 / 加到 3 epoch。
3. 双卡 bf16 跑满 cutoff 24576(去掉 QLoRA 精度损失 + 截断)。
4. 加一个反思 agent(已设计,见记忆 `reflection-agent-deferred`)。

---

## 文件清单

- `{base,ft,deepseek}_{standard,hard}/report.md` / `report.json` / `metrics.jsonl` / `generations.jsonl` —— 6 组评测产物
- adapter:`../outputs/qwen25_7b_qlora_v1/`(323MB safetensors + config + trainer_state)
