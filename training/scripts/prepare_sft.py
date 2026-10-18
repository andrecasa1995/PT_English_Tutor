"""Export reviewed seed examples to local JSONL for future supervised fine-tuning.

Only seed_examples.json is eligible. The held-out evaluation file is never loaded.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "training" / "datasets" / "seed_examples.json"
DEST = ROOT / "training" / "data" / "sft_train.jsonl"
SYSTEM = (
    "You are a careful American English writing tutor for Brazilian Portuguese speakers. "
    "Return JSON with a single edits array. Each edit has original, replacement, category, "
    "and explanation_pt. Only propose necessary, minimal edits; explain in Brazilian Portuguese."
)


def convert(record):
    edits = record["edits"]
    # Canonical targets contain ONLY the intended changes.
    target = json.dumps({"edits": edits}, ensure_ascii=False, separators=(",", ":"))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": record["text"]},
            {"role": "assistant", "content": target},
        ]
    }


def main():
    records = json.loads(SOURCE.read_text(encoding="utf-8"))["examples"]
    # Structural validation before producing a training artifact.
    import sys
    sys.path.insert(0, str(ROOT / "training"))
    from validate_dataset import validate
    validate(records)
    assert len({r["group"] for r in records}) == len(records), "Review grouped examples before export"
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with DEST.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(convert(record), ensure_ascii=False) + "\n")
    print(f"Exported {len(records)} supervised training examples to {DEST}")


if __name__ == "__main__":
    main()
