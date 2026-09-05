import json

import pytest

from aether_scientist.core.profiles import get_profile
from aether_scientist.finetune.data import (
    SYSTEM_PROMPT,
    format_chatml_entry,
    format_dataset,
    load_scientific_data,
)


def test_format_chatml_entry():
    entry = format_chatml_entry("Explain quantum entanglement.", "Quantum entanglement is...")
    assert "messages" in entry
    messages = entry["messages"]
    assert len(messages) == 3
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": "Explain quantum entanglement."}
    assert messages[2] == {"role": "assistant", "content": "Quantum entanglement is..."}


def test_format_dataset_raw_text():
    prof = get_profile("fast")
    docs = ["Attention mechanisms allow neural networks to focus on specific sequence parts."]
    formatted = format_dataset(docs, prof)
    assert len(formatted) == 1
    msgs = formatted[0]["messages"]
    assert msgs[0]["role"] == "system"
    assert "Attention mechanisms" in msgs[1]["content"]
    assert "Attention mechanisms" in msgs[2]["content"]


def test_format_dataset_qa_format():
    prof = get_profile("fast")
    docs = ["Question: What is CRISPR?\n\nAnswer: CRISPR is a gene editing technology."]
    formatted = format_dataset(docs, prof)
    assert len(formatted) == 1
    msgs = formatted[0]["messages"]
    assert msgs[1]["content"] == "What is CRISPR?"
    assert msgs[2]["content"] == "CRISPR is a gene editing technology."


def test_format_dataset_json_records():
    prof = get_profile("fast")
    records = [
        json.dumps({
            "instruction": "Define entropy.",
            "input": "In thermodynamics",
            "output": "Entropy is a measure of molecular disorder.",
        }),
        json.dumps({
            "question": "What is ATP?",
            "answer": "Adenosine triphosphate is cellular energy currency.",
        }),
    ]
    formatted = format_dataset(records, prof)
    assert len(formatted) == 2
    assert "Define entropy." in formatted[0]["messages"][1]["content"]
    assert "In thermodynamics" in formatted[0]["messages"][1]["content"]
    assert formatted[0]["messages"][2]["content"] == "Entropy is a measure of molecular disorder."
    assert formatted[1]["messages"][1]["content"] == "What is ATP?"
    assert "Adenosine triphosphate" in formatted[1]["messages"][2]["content"]


def test_format_dataset_empty_skips():
    prof = get_profile("fast")
    assert format_dataset(["", "   \n\n  "], prof) == []


def test_load_scientific_data_json(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps([{"instruction": "Q1", "output": "A1"}]), encoding="utf-8")
    loaded = load_scientific_data(p)
    assert len(loaded) == 1
    assert "Q1" in loaded[0]


def test_load_scientific_data_jsonl(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"instruction": "Q1"}\n{"instruction": "Q2"}\n', encoding="utf-8")
    loaded = load_scientific_data(p)
    assert len(loaded) == 2


def test_load_scientific_data_txt(tmp_path):
    p = tmp_path / "data.txt"
    p.write_text("Paragraph one.\n\nParagraph two.", encoding="utf-8")
    loaded = load_scientific_data(p)
    assert len(loaded) == 2
    assert loaded[0] == "Paragraph one."


def test_load_scientific_data_missing():
    with pytest.raises(FileNotFoundError):
        load_scientific_data("non_existent_data_file.json")
