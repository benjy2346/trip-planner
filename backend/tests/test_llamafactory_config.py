"""训练配置与数据集注册的静态校验（不训练，只查关键字段没写错）。"""
import json
from pathlib import Path

import yaml

CFG = Path("ml/planner/configs")
DS = Path("ml/planner/llamafactory/dataset_info.json")


def test_sft_config_is_qlora_long_context():
    cfg = yaml.safe_load((CFG / "qwen25_7b_lora_sft.yaml").read_text())
    assert cfg["cutoff_len"] == 24576                 # 不是 8192
    assert cfg["quantization_bit"] == 4               # QLoRA
    assert cfg["finetuning_type"] == "lora"
    assert cfg["lora_rank"] == 32 and cfg["lora_alpha"] == 64
    assert cfg["dataset"] == "trip_planner_sft"
    assert cfg["eval_dataset"] == "trip_planner_sft_val"
    assert cfg["bf16"] is True                        # 计算精度仍 bf16
    assert cfg["gradient_checkpointing"] is True


def test_merge_config_points_at_adapter():
    cfg = yaml.safe_load((CFG / "qwen25_7b_lora_merge.yaml").read_text())
    assert cfg["adapter_name_or_path"] == cfg_expected_adapter(cfg)
    assert cfg["finetuning_type"] == "lora"
    assert "export_dir" in cfg
    # 合并导出不能再带量化（要导出 bf16 完整模型给 vLLM）
    assert "quantization_bit" not in cfg


def cfg_expected_adapter(cfg):
    # 合并配置的 adapter 路径应等于 sft 配置的 output_dir
    sft = yaml.safe_load((CFG / "qwen25_7b_lora_sft.yaml").read_text())
    return sft["output_dir"]


def test_dataset_info_sharegpt_mapping_intact():
    ds = json.loads(DS.read_text())
    for name in ("trip_planner_sft", "trip_planner_sft_val"):
        entry = ds[name]
        assert entry["formatting"] == "sharegpt"
        assert entry["columns"]["messages"] == "conversations"
        assert entry["columns"]["system"] == "system"
