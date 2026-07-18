# Grounded Planner 训练 + 评测（Plan 2b）设计

> Plan 2 的后半：**LoRA 微调 + 三方评测**。依赖 Plan 2a（已产出 1010 条 grounded 数据 → `train.json` 960 / `val.json` 50）。需远程 GPU（AutoDL RTX 5090 D，PyTorch 2.8/CUDA 12.8）。

## 目标

用 Plan 2a 的 grounded 数据 LoRA 微调 Qwen2.5-7B（planner/assembler），在冻结评测集（标准 200 + hard 300）上量化提升，做 **base 7B / 微调 7B / DeepSeek** 三方对比，主指标 `hard_pass`。产出：一份三方对比报告 + 微调好的 LoRA adapter。

**成功判据**：微调 7B 的 hard_pass 显著高于 base 7B（证明微调有用），并向 DeepSeek 靠拢（蒸馏成功度）。

**source of truth**：打分器 = helloagents `training/scripts/eval/eval_rule_metrics.py::evaluate_output`。原则：参照他、写我们自己的（数据表/关键词逐字，逻辑自写）。

## 范围拆分（两个环境）

Plan 2b 分两半，分别在不同环境：

### A. 2b-code（本地，TDD 可测）
1. **移植打分器**：`backend/ml/planner/eval/rule_metrics.py` ← 他 `eval_rule_metrics.py` 的 `evaluate_output` + helpers + 聚合。计算 hard_pass（无违规）+ 软指标（景点/餐饮 grounding 率、餐饮多样性、预算贴合、meal_scale 等）。读冻结评测记录的 `planner_context`/`compact_planner_context`。可测：`--use-reference-output` 拿评测记录里的 teacher output 打分做 smoke（应高 hard_pass）。
2. **生成 runner**：`backend/ml/planner/eval/generate.py` ← 复用现有 `rule_eval.py` 的生成半部（`ChatOpenAI(base_url, model, api_key)` → OpenAI 兼容端点）。对每条评测记录，**直接用记录里已烤好的 `compact_planner_context`**（不重新 compact——保证与训练/teacher 见到的输入同源）走 `build_grounded_planner_messages(record["compact_planner_context"])` 生成 → 写 `generations.jsonl`（`record_id` + `output`）。**同一 runner 靠换 base_url/model/key 打通三方**（vLLM 的 `/v1` + DeepSeek）。
3. **注册数据集**：`ml/planner/llamafactory/dataset_info.json` 注册 `trip_planner_sft`（train.json）/ `trip_planner_sft_val`（val.json），sharegpt 格式。更新 LoRA 配置数据集路径。

### B. 2b-run（远程 AutoDL 运行手册，非单测；交互执行 + 看产物）
在 5090 实例上按序执行：传输 → 下模型 → 训练 → 起服务 → 生成 → 打分 → 报告。见下「远程工作流」。

## 训练配置

**QLoRA 单卡**（1×32G）。基于 Plan 1 的 `ml/planner/configs/qwen25_7b_lora_sft.yaml`，但有两处关键修正：

- **`cutoff_len: 24576`**（不是 8192）。实测我们的 grounded 上下文中位 ~22.5K token、最大 ~31.5K token，**96% 样本 >8192 token**；8192 会截断掉绝大多数候选，训练输入残缺。24576 与 helloagents 一致（他也用 24576），覆盖中位、只截最长 ~10-20%（可接受）。
- **QLoRA（4-bit NF4 base）**：`quantization_bit: 4` + `quantization_method: bitsandbytes`(nf4)。4-bit base 省 ~10G 显存，让 24K 上下文在单张 32G 卡装得下（bf16 LoRA 在 24K 需 2 卡序列并行，见下）。代价：4-bit base 轻微降质，SFT 下通常可接受。

其余沿用：LoRA r32 / α64 / dropout 0.05 / target all，micro_batch 1，bf16 计算，gradient/activation checkpointing，flash-attention。**超参起点**：lr 1e-4（QLoRA 常用比 bf16 略高；helloagents bf16 用 1e-5，我们 QLoRA 起 1e-4），2–3 epochs，global_batch ~16-32（grad_accum 调）。**均为首轮起点，看 train/val loss 再调**。

**只改**：dataset 指向传上去的 train/val + `dataset_info.json` 注册、`output_dir` 到数据盘、加 quantization 与 cutoff 字段。

> 为何不是 bf16 单卡：24K 上下文 bf16 LoRA 在单张 32G 上 ~26G 峰值、临界易 OOM；helloagents 靠 2 卡 Ulysses 序列并行才装下。QLoRA 是单卡长上下文的务实解。若首轮结果接近但不够，可升级到 2 卡 bf16（对齐他、质量更高）。

## 评测数据流（他的解耦结构）

```
冻结评测记录(planner_context) ──┐
                               ▼
  generate.py(换 base_url/model) → generations.jsonl(三份：base / 微调 / deepseek)
                               ▼
  rule_metrics.py(evaluate_output) → 逐条打分 → 聚合
                               ▼
  三方报告：standard-200 / hard-300 分别报 hard_pass + 软指标
```

**三方端点**：
- base 7B、微调 7B → 5090 上 **vLLM** 起 OpenAI 兼容服务。微调侧：QLoRA 训练产出的是 4-bit base + adapter，但 **serve 时用 bf16**——`llamafactory-cli export` 把 LoRA 合并进反量化后的 bf16 base，得到一个完整 bf16 模型给 vLLM serve（评测最简单、少一层变量）。vLLM 不吃训练时的 4-bit 格式，故必须先合并成 bf16。
- DeepSeek → API（有 key）。

**指标**：`hard_pass`（主，standard/hard 分开）+ 软指标（grounding/多样性/预算贴合率）。可选二次层：LLM-as-judge 成对比较（他有 `eval_pairwise_judge`），非必需，先不做。

## 远程工作流（2b-run）

1. **传输**：AutoDL 上 `git clone` 本仓库（含代码 + committed 的 500 条评测记录）；`scp` gitignored 的 `train.json`/`val.json` 上去；下 Qwen2.5-7B-Instruct（**ModelScope**，国内快）。
2. **环境校验**：`torch.zeros(1).cuda()` 通过（确认 PyTorch 2.8/cu128 支持 Blackwell sm_120）；装 LLaMA-Factory + vLLM。
3. **训练**：`llamafactory-cli train qwen25_7b_lora_sft.yaml` → LoRA adapter。看 loss 曲线、val loss。
4. **serve + 生成**：vLLM 起 base → 跑 generate.py 得 base 的 generations；起微调 → 得微调的 generations；DeepSeek API → 得 deepseek 的 generations。（各 500 条）
   - **vLLM 必须 `--max-model-len 32768`**：我们的输入 compact context 中位 ~22.5K token、最大 ~31.5K，加上生成的 plan 输出，需要接近 Qwen2.5-7B 原生 32K 上限。默认 max-model-len 若偏小，长输入会被拒/截断，评测就废了。generate.py 侧 `max_tokens`（输出）设 ~4096 足够一份行程。
5. **打分 + 报告**：rule_metrics.py 对三份 generations 打分 → 三方对比报告（committed 到 `backend/ml/planner/reports/`）。
6. **回收**：LoRA adapter + 报告拉回本地；关机止损。

## 明确不做（YAGNI）
- DPO / rerank / best-of-N（他有，属更后期）。
- LLM-judge（先只做规则分；需要再加）。
- 全参微调（LoRA 足够，32G 也放不下全参）。
- 多卡（1 卡够）。

## 风险
- **⚠️ QLoRA/bitsandbytes 在 Blackwell 上的兼容（可能卡死 QLoRA 方案）**：QLoRA 4-bit 靠 `bitsandbytes`，而 bnb 对 sm_120/CUDA 12.8 的支持是近期才有的。**开机第一件事：装好后先跑一个极小的 4-bit 加载冒烟**（`BitsAndBytesConfig(load_in_4bit=True)` 载入 Qwen2.5-7B 一层/整模型不报错）。若 bnb 不支持 5090 → QLoRA 走不通，退路：① 降 cutoff 到能装下的 bf16 单卡（截断部分候选），② 或加租 1 卡走 bf16+Ulysses（对齐他）。**别等训练启动才发现 bnb 挂了。**
- **Blackwell 通用兼容**：PyTorch 2.8/cu128 支持 sm_120，但 vLLM / LLaMA-Factory / flash-attn 也要跟上；开机先 `torch.zeros(1).cuda()` + 上面的 bnb 冒烟。
- **24K 上下文显存**：即便 QLoRA，24576 token 的激活也不小。确认 flash-attention + gradient checkpointing 开启；若 OOM，先降 `cutoff_len`（如 16384，仍远好于 8192）再考虑其他。
- **数据集注册**：LLaMA-Factory 的 `dataset_info.json` 里 sharegpt 映射要对——我们导出的是 `conversations`(from/value) + 顶层 `system` 字段，需正确配 `formatting: sharegpt` + `columns`(messages/system) + `tags`。训练前跑一次 preprocessing dry-run 确认能加载、且截断比例合理。
- **评测可比性（勿高估）**：冻结评测集 + 他的打分器让**方法学**一致、可对照；但我们的微调用的是**我们自己的数据 + QLoRA 单卡**（他是 bf16 双卡、数据不同），所以**绝对分数不宜直接等同他公布的数字**。真正内部有效的结论是 base vs 我们微调 vs DeepSeek 这一组三方对比。
- **成本/时长（保守估）**：24K 长上下文训练慢，~960 样本 ×2–3 epoch 可能 3–5 小时；加装环境+评测，全流程约 **6–12 小时、¥20–40**。DeepSeek 三方那份是 500 条 API 调用。**不用即关机**（含调试卡住时）。
