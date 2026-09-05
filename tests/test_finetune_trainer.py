import json
import sys
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from aether_scientist.cli import app
from aether_scientist.finetune.config import get_lora_config
from aether_scientist.finetune.trainer import run_finetune


def test_get_lora_config_mocked():
    mock_peft = MagicMock()
    with patch.dict(sys.modules, {"peft": mock_peft}):
        get_lora_config("fast", r=8, lora_alpha=16)
        mock_peft.LoraConfig.assert_called_once()
        kwargs = mock_peft.LoraConfig.call_args[1]
        assert kwargs["r"] == 8
        assert kwargs["lora_alpha"] == 16
        assert "q_proj" in kwargs["target_modules"]


def test_get_lora_config_offline_profile():
    mock_peft = MagicMock()
    with patch.dict(sys.modules, {"peft": mock_peft}):
        get_lora_config("offline")
        kwargs = mock_peft.LoraConfig.call_args[1]
        assert kwargs["target_modules"] == ["c_attn"]


def test_get_lora_config_missing_peft():
    with (
        patch.dict(sys.modules, {"peft": None}),
        pytest.raises(ImportError, match="peft is required for fine-tuning"),
    ):
        get_lora_config("fast")


def test_run_finetune_missing_deps(tmp_path):
    data_file = tmp_path / "sample.json"
    data_file.write_text(json.dumps([{"instruction": "Q", "output": "A"}]))
    with (
        patch.dict(sys.modules, {"trl": None}),
        pytest.raises(ImportError, match="Fine-tuning dependencies missing"),
    ):
        run_finetune(str(data_file))


def test_run_finetune_empty_data(tmp_path):
    data_file = tmp_path / "empty.json"
    data_file.write_text(json.dumps([]))
    mock_modules = {
        "peft": MagicMock(),
        "trl": MagicMock(),
        "transformers": MagicMock(),
        "datasets": MagicMock(),
        "torch": MagicMock(),
    }
    with (
        patch.dict(sys.modules, mock_modules),
        pytest.raises(ValueError, match="No valid records found"),
    ):
        run_finetune(str(data_file))


def test_run_finetune_cpu_abort_large_dataset(tmp_path):
    data_file = tmp_path / "large.json"
    data = [{"instruction": f"Q{i}", "output": f"A{i}"} for i in range(105)]
    data_file.write_text(json.dumps(data))

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_modules = {
        "peft": MagicMock(),
        "trl": MagicMock(),
        "transformers": MagicMock(),
        "datasets": MagicMock(),
        "torch": mock_torch,
    }
    with patch.dict(sys.modules, mock_modules):
        res = run_finetune(str(data_file), profile_name="fast")
        assert res["status"] == "aborted"
        assert "exceeds 100 rows" in res["reason"]


def test_run_finetune_mocked_success(tmp_path):
    data_file = tmp_path / "small.json"
    data_file.write_text(json.dumps([
        {"instruction": "Explain gravity", "output": "Attraction of masses"},
        {"instruction": "Explain photons", "output": "Light quanta"},
    ]))
    out_dir = tmp_path / "lora_adapter"

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    mock_tok = MagicMock()
    mock_tok.pad_token = None
    mock_tok.eos_token = "[EOS]"

    mock_model = MagicMock()
    mock_peft_model = MagicMock()

    mock_peft = MagicMock()
    mock_peft.get_peft_model.return_value = mock_peft_model

    mock_trainer = MagicMock()
    mock_trl = MagicMock()
    mock_trl.SFTTrainer.return_value = mock_trainer

    mock_tf = MagicMock()
    mock_tf.AutoTokenizer.from_pretrained.return_value = mock_tok
    mock_tf.AutoModelForCausalLM.from_pretrained.return_value = mock_model

    mock_datasets = MagicMock()
    mock_datasets.Dataset.from_list.return_value = MagicMock()

    mock_modules = {
        "torch": mock_torch,
        "peft": mock_peft,
        "trl": mock_trl,
        "transformers": mock_tf,
        "datasets": mock_datasets,
    }

    with (
        patch.dict(sys.modules, mock_modules),
        patch("aether_scientist.finetune.trainer.get_lora_config") as mock_get_cfg,
    ):
        mock_get_cfg.return_value = MagicMock()
        res = run_finetune(str(data_file), profile_name="fast", output_dir=str(out_dir))

        assert res["status"] == "completed"
        assert res["profile"] == "fast"
        assert res["samples"] == 2
        mock_trainer.train.assert_called_once()


def test_cli_finetune_missing_file():
    runner = CliRunner()
    result = runner.invoke(app, ["finetune", "--data", "missing_file_xyz.json"])
    assert result.exit_code != 0


def test_cli_finetune_success(tmp_path):
    runner = CliRunner()
    data_file = tmp_path / "data.json"
    data_file.write_text(json.dumps([{"instruction": "Q", "output": "A"}]))

    mock_res = {
        "status": "completed",
        "profile": "fast",
        "output_dir": str(tmp_path / "out"),
        "epochs": 1,
        "samples": 1,
    }
    with patch("aether_scientist.finetune.trainer.run_finetune", return_value=mock_res):
        res = runner.invoke(app, ["finetune", "--data", str(data_file), "--json"])
        assert res.exit_code == 0
        parsed = json.loads(res.output)
        assert parsed["status"] == "completed"
        assert parsed["samples"] == 1
