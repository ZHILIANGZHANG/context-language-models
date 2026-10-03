"""Append subagent answers to a pilot: python rec.py <pilot_dir> <model> <blind:shelf> ..."""
import json, sys
from pathlib import Path

pilot, model = Path(sys.argv[1]), sys.argv[2]
(pilot / "responses").mkdir(parents=True, exist_ok=True)
with open(pilot / "responses" / f"{model}.jsonl", "a") as f:
    for tok in sys.argv[3:]:
        b, s = tok.split(":")
        f.write(json.dumps({"blind": b, "model": model, "response": json.dumps({"shelf": f"S{s}"})}) + "\n")
print("recorded", len(sys.argv) - 3)
