# Plan 2b 远程运行手册（AutoDL RTX 5090 D）

环境：AutoDL F01 机，RTX 5090 32GB，基础镜像 **PyTorch 2.8.0 / CUDA 12.8 / Ubuntu22.04**，¥2.93/时。
执行方式：本地通过 `ssh`/`scp` 驱动；长任务（训练）在远程 `nohup ... &` 后台跑 + 轮询。
**用不即关机**（含调试卡住时）。全流程约 6–12 小时、¥20–40。

## 0. 连接（一次性）
- 推荐 SSH key 免密：把本地公钥加进 AutoDL 实例 `~/.ssh/authorized_keys`，或配 `~/.ssh/config` 别名 `autodl`。
- 之后所有步骤都能 `ssh autodl 'cmd'` 非交互执行。

## 1. ⚠️ bnb 4-bit 冒烟（开机第一件事，别跳过）
QLoRA 依赖 bitsandbytes，其对 Blackwell(sm_120) 的支持是近期才有的。**先验证再训练**：
```bash
pip install -U bitsandbytes
python - <<'PY'
import torch, transformers
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
print("cuda ok:", torch.zeros(1).cuda().is_cuda, "| cap:", torch.cuda.get_device_capability())
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
        quantization_config=bnb, device_map="cuda", trust_remote_code=True)
print("4-bit load OK, layers:", m.config.num_hidden_layers)
PY
```
- **通过** → 继续 QLoRA。
- **失败**（bnb 报 sm_120 不支持）→ 退路二选一：① sft 配置去掉 `quantization_*`、`cutoff_len` 降到能装下的 bf16 单卡值（先试 16384）；② 加租第二张卡走 bf16 + Ulysses 序列并行。**别硬扛，先决定路线再往下。**

## 2. 传输 + 环境
```bash
# 远程 clone 本仓库（含代码 + committed 的 500 条评测记录）
ssh autodl 'cd ~ && git clone <repo-url> trip-planner && cd trip-planner && git checkout feat/planner-finetune'
# 上传 gitignored 的训练数据
scp backend/ml/planner/llamafactory/generated/train.json autodl:~/trip-planner/backend/ml/planner/llamafactory/generated/
scp backend/ml/planner/llamafactory/generated/val.json   autodl:~/trip-planner/backend/ml/planner/llamafactory/generated/
# 装依赖
ssh autodl 'pip install -U "llamafactory[torch,metrics,bitsandbytes]" vllm langchain-openai'
```
下模型（ModelScope，国内快）：
```bash
ssh autodl 'pip install -U modelscope && modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ~/models/Qwen2.5-7B-Instruct'
```
> 若用本地已下模型，把两个 yaml 里的 `model_name_or_path` 指到 `~/models/Qwen2.5-7B-Instruct`。

## 3. 预处理 dry-run（先验证数据集加载 + 截断比例）
```bash
ssh autodl 'cd ~/trip-planner/backend && llamafactory-cli train ml/planner/configs/qwen25_7b_lora_sft.yaml \
  --max_steps 1 --output_dir /tmp/dryrun'
```
- 确认能加载 `trip_planner_sft` / `_val`、sharegpt 映射无报错、截断比例合理（>24576 的样本约 10–20%）。
- 若报 `quantization_*` 未知字段，按该 LLaMA-Factory 版本 `examples/train_qlora/*.yaml` 修正字段名，改完重跑本步。

## 4. 训练（后台 + 轮询）
```bash
ssh autodl 'cd ~/trip-planner/backend && nohup llamafactory-cli train ml/planner/configs/qwen25_7b_lora_sft.yaml \
  > ~/train.log 2>&1 &'
# 轮询
ssh autodl 'tail -n 30 ~/train.log; nvidia-smi --query-gpu=memory.used --format=csv'
```
- 看 train/val loss 下降。若 OOM：先把 `cutoff_len` 降到 16384 重训（仍远好于 8192）。
- 产出 LoRA adapter 在 `ml/planner/outputs/qwen25_7b_qlora_v1`。

## 5. 合并成 bf16 完整模型（给 vLLM）
```bash
ssh autodl 'cd ~/trip-planner/backend && llamafactory-cli export ml/planner/configs/qwen25_7b_lora_merge.yaml'
# 产出：ml/planner/outputs/qwen25_7b_qlora_v1_merged
```

## 6. 三方生成（3 端点 × 2 评测集 = 6 份 generations）
每个 vLLM 端点**必须** `--max-model-len 32768`（输入中位 ~22.5K、最大 ~31.5K token）。
```bash
# --- base 7B ---
ssh autodl 'cd ~/trip-planner/backend && nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/Qwen2.5-7B-Instruct --max-model-len 32768 --port 8000 > ~/vllm_base.log 2>&1 &'
# 起来后（tail 日志看到 "Uvicorn running"）：
ssh autodl 'cd ~/trip-planner/backend && \
  python -m ml.planner.eval.generate --records ml/planner/eval/records.jsonl \
    --base-url http://127.0.0.1:8000/v1 --model base --api-key-env NONE --output-dir runs_eval/base_standard && \
  python -m ml.planner.eval.generate --records ml/planner/eval_hard/records.jsonl \
    --base-url http://127.0.0.1:8000/v1 --model base --api-key-env NONE --output-dir runs_eval/base_hard'
# 停 base vLLM，再起 merged（换 --model 路径，同 --port 8000），跑 ft_standard / ft_hard：
#   --model ~/trip-planner/backend/ml/planner/outputs/qwen25_7b_qlora_v1_merged --output-dir runs_eval/ft_{standard,hard}
# --- DeepSeek（API，无需 GPU）---
ssh autodl 'cd ~/trip-planner/backend && export DEEPSEEK_API_KEY=<key> && \
  python -m ml.planner.eval.generate --records ml/planner/eval/records.jsonl \
    --base-url https://api.deepseek.com/v1 --model deepseek-chat --output-dir runs_eval/deepseek_standard && \
  python -m ml.planner.eval.generate --records ml/planner/eval_hard/records.jsonl \
    --base-url https://api.deepseek.com/v1 --model deepseek-chat --output-dir runs_eval/deepseek_hard'
```

## 7. 打分（6 份 generations → 6 份报告）
```bash
ssh autodl 'cd ~/trip-planner/backend && for tag in base ft deepseek; do \
  python -m ml.planner.eval.rule_metrics --records ml/planner/eval/records.jsonl \
    --generations runs_eval/${tag}_standard/generations.jsonl --output-dir runs_eval/${tag}_standard --model-tag ${tag}_standard; \
  python -m ml.planner.eval.rule_metrics --records ml/planner/eval_hard/records.jsonl \
    --generations runs_eval/${tag}_hard/generations.jsonl --output-dir runs_eval/${tag}_hard --model-tag ${tag}_hard; \
done'
```
成功判据：微调 7B 的 `hard_pass` 显著高于 base 7B，并向 DeepSeek 靠拢。

## 8. 回收 + 关机
```bash
scp -r autodl:~/trip-planner/backend/runs_eval ./backend/ml/planner/reports_2b/     # 拉回 6 份报告
scp -r autodl:~/trip-planner/backend/ml/planner/outputs/qwen25_7b_qlora_v1 ./         # 拉回 adapter（可选）
# 确认无残留后台任务后，AutoDL 控制台关机止损。
```
本地把 6 份报告整理成一份三方对比，committed 到 `backend/ml/planner/reports_2b/`。
