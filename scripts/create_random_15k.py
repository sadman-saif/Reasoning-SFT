import json
import random
from pathlib import Path

INPUT_FILE = Path("raw_data/data/acereason11_100k.json")
OUTPUT_FILE = Path("LLaMA-Factory/data/acereason_random_15k.json")
N_SAMPLES = 15_000
SEED = 42

with INPUT_FILE.open("r", encoding="utf-8") as f:
    data = json.load(f)

valid = []
for row in data:
    instruction = str(row.get("instruction", "")).strip()
    output = str(row.get("output", "")).strip()
    if instruction and output:
        valid.append(row)

if len(valid) < N_SAMPLES:
    raise RuntimeError(f"Only {len(valid)} valid rows remain; need {N_SAMPLES}.")

rng = random.Random(SEED)
selected = rng.sample(valid, N_SAMPLES)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    json.dump(selected, f, ensure_ascii=False, indent=2)

print(f"Source rows: {len(data):,}")
print(f"Valid rows: {len(valid):,}")
print(f"Selected: {len(selected):,}")
print(f"Seed: {SEED}")
print(f"Saved to: {OUTPUT_FILE}")
