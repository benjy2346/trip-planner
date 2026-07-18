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

复用 Plan 1 的 `ml/planner/configs/qwen25_7b_lora_sft.yaml`（已验证合理）：Qwen2.5-7B-Instruct，LoRA rank 32 / α 64 / dropout 0.05 / target all，cutoff 8192，3 epochs，lr 5e-5，batch 1 × grad_accum 16，bf16，gradient_checkpointing。**只改**：`dataset_dir` 指向传上去的数据、确认 `dataset_info.json` 注册、`output_dir` 到数据盘。首轮用默认超参，评测不理想再调（rank/lr/epochs）。

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
- base 7B、微调 7B → 5090 上 **vLLM** 起 OpenAI 兼容服务。微调侧默认 **LLaMA-Factory export 合并 LoRA 后再 serve**（评测最简单、少一层变量）；如需省去合并步骤可退而用 vLLM `--enable-lora` 直挂 adapter。
- DeepSeek → API（有 key）。

**指标**：`hard_pass`（主，standard/hard 分开）+ 软指标（grounding/多样性/预算贴合率）。可选二次层：LLM-as-judge 成对比较（他有 `eval_pairwise_judge`），非必需，先不做。

## 远程工作流（2b-run）

1. **传输**：AutoDL 上 `git clone` 本仓库（含代码 + committed 的 500 条评测记录）；`scp` gitignored 的 `train.json`/`val.json` 上去；下 Qwen2.5-7B-Instruct（**ModelScope**，国内快）。
2. **环境校验**：`torch.zeros(1).cuda()` 通过（确认 PyTorch 2.8/cu128 支持 Blackwell sm_120）；装 LLaMA-Factory + vLLM。
3. **训练**：`llamafactory-cli train qwen25_7b_lora_sft.yaml` → LoRA adapter。看 loss 曲线、val loss。
4. **serve + 生成**：vLLM 起 base → 跑 generate.py 得 base 的 generations；起微调 → 得微调的 generations；DeepSeek API → 得 deepseek 的 generations。（各 500 条）
5. **打分 + 报告**：rule_metrics.py 对三份 generations 打分 → 三方对比报告（committed 到 `backend/ml/planner/reports/`）。
6. **回收**：LoRA adapter + 报告拉回本地；关机止损。

## 明确不做（YAGNI）
- DPO / rerank / best-of-N（他有，属更后期）。
- LLM-judge（先只做规则分；需要再加）。
- 全参微调（LoRA 足够，32G 也放不下全参）。
- 多卡（1 卡够）。

## 风险
- **Blackwell 兼容**：PyTorch 2.8/cu128 支持 sm_120，但 vLLM / LLaMA-Factory 也要跟上；开机先 `.cuda()` 冒烟，别训到一半才发现。
- **数据集注册**：LLaMA-Factory 认 `dataset_info.json` 的 sharegpt 字段（我们导出的是 `conversations`/`system`）；训练前先 `--use-reference-output` 之外单跑一个 preprocessing dry-run 确认能加载。
- **评测可比性**：用他的打分器 + 他的冻结评测集 → 分数可与他公布数字比；但高德 POI 已烤进他的评测记录，我们不重拍，保持冻结。
- **成本**：GPU ~¥3/时，全流程 4–8 小时；DeepSeek 三方那份是 500 条 API 调用。不用即关机。
