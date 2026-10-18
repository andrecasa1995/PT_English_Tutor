"""Validate small supervised-edit seed datasets before any training."""
import json
from pathlib import Path

DATASETS = Path(__file__).with_name("datasets")
FILES = [DATASETS / "seed_examples.json", DATASETS / "heldout_eval.json"]
CATEGORIES = {"grammar", "spelling", "translation", "style"}


def validate(examples):
    ids = set()
    for item in examples:
        ident = item["id"]
        assert ident not in ids, f"Duplicate id: {ident}"
        ids.add(ident)
        assert item["group"], f"{ident}: missing group"
        text = item["text"]
        assert text.strip(), f"{ident}: empty text"
        spans = []
        for edit in item["edits"]:
            original = edit["original"]
            assert original and original != edit["replacement"], f"{ident}: no-op edit"
            assert edit["category"] in CATEGORIES, f"{ident}: unknown category"
            assert edit["explanation_pt"].strip(), f"{ident}: missing explanation"
            assert text.count(original) == 1, f"{ident}: ambiguous original {original}"
            start = text.index(original)
            end = start + len(original)
            assert all(end <= a or start >= b for a, b in spans), f"{ident}: overlap"
            spans.append((start, end))
    return len(ids)


if __name__ == "__main__":
    all_groups = set()
    for file in FILES:
        data = json.loads(file.read_text(encoding="utf-8"))
        count = validate(data["examples"])
        groups = {item["group"] for item in data["examples"]}
        assert not (all_groups & groups), f"{file.name}: train/eval group overlap"
        all_groups.update(groups)
        print(f"Dataset valid: {file.name} ({count} examples)")
