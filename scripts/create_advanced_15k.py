
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = Path(
    "raw_data/data/acereason11_100k.json"
)

OUTPUT_FILE = Path(
    "LLaMA-Factory/data/acereason_advanced_15k.json"
)

METADATA_FILE = Path(
    "advanced_selection_metadata.csv"
)

N_SELECT = 15000
N_CLUSTERS = 300
SEED = 42

# Prefer examples around the 70th percentile of solution length
# within each semantic cluster.
TARGET_DIFFICULTY = 0.70


# ============================================================
# 1. LOAD DATA
# ============================================================

with INPUT_FILE.open("r", encoding="utf-8") as f:
    raw = json.load(f)

print("Original examples:", len(raw))


# ============================================================
# 2. REMOVE INVALID + DUPLICATE PROMPTS
# ============================================================

records = []
seen_prompts = set()

for i, row in enumerate(raw):

    instruction = str(
        row.get("instruction", "")
    ).strip()

    output = str(
        row.get("output", "")
    ).strip()

    if not instruction or not output:
        continue

    normalized = re.sub(
        r"\s+",
        " ",
        instruction.lower()
    ).strip()

    if normalized in seen_prompts:
        continue

    seen_prompts.add(normalized)

    records.append(
        {
            "original_index": i,
            "instruction": instruction,
            "solution_words": len(output.split()),
            "record": row,
        }
    )

print("After filtering:", len(records))

if len(records) < N_SELECT:
    raise RuntimeError("Not enough valid examples.")


# ============================================================
# 3. EMBED THE QUESTIONS
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Embedding device:", device)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device
)

prompts = [
    x["instruction"]
    for x in records
]

embeddings = model.encode(
    prompts,
    batch_size=256,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Embedding matrix:", embeddings.shape)


# ============================================================
# 4. SEMANTIC CLUSTERING
# ============================================================

print("Running MiniBatchKMeans...")

kmeans = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    random_state=SEED,
    batch_size=2048,
    n_init=10
)

labels = kmeans.fit_predict(embeddings)

clusters = {
    i: []
    for i in range(N_CLUSTERS)
}

for record, label in zip(records, labels):

    record["cluster"] = int(label)

    clusters[int(label)].append(record)

print(
    "Clusters created:",
    len(clusters)
)


# ============================================================
# 5. CALCULATE RELATIVE DIFFICULTY
# ============================================================

for cluster_id, items in clusters.items():

    items.sort(
        key=lambda x: x["solution_words"]
    )

    n = len(items)

    for rank, item in enumerate(items):

        if n == 1:
            percentile = 0.5
        else:
            percentile = rank / (n - 1)

        item["difficulty_percentile"] = percentile

        item["difficulty_score"] = (
            1.0
            - abs(
                percentile
                - TARGET_DIFFICULTY
            )
        )


# ============================================================
# 6. BALANCED SELECTION ACROSS CLUSTERS
# ============================================================

base_quota = N_SELECT // N_CLUSTERS
remainder = N_SELECT % N_CLUSTERS

print(
    "Base examples per cluster:",
    base_quota
)

cluster_order = sorted(
    clusters.keys(),
    key=lambda c: len(clusters[c]),
    reverse=True
)

selected = []
selected_ids = set()

for position, cluster_id in enumerate(cluster_order):

    quota = base_quota

    if position < remainder:
        quota += 1

    candidates = sorted(
        clusters[cluster_id],
        key=lambda x: x["difficulty_score"],
        reverse=True
    )

    chosen = candidates[:quota]

    for item in chosen:

        selected.append(item)
        selected_ids.add(
            item["original_index"]
        )


# ============================================================
# 7. FILL ANY MISSING SLOTS
# ============================================================

needed = N_SELECT - len(selected)

print(
    "Selected after cluster quotas:",
    len(selected)
)

print(
    "Additional examples needed:",
    needed
)

if needed > 0:

    remaining = [
        x
        for x in records
        if x["original_index"]
        not in selected_ids
    ]

    remaining.sort(
        key=lambda x: x["difficulty_score"],
        reverse=True
    )

    selected.extend(
        remaining[:needed]
    )

selected = selected[:N_SELECT]

assert len(selected) == N_SELECT


# ============================================================
# 8. SAVE ADVANCED DATASET
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with OUTPUT_FILE.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        [x["record"] for x in selected],
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 9. SAVE SELECTION METADATA
# ============================================================

metadata = pd.DataFrame(
    [
        {
            "original_index":
                x["original_index"],

            "cluster":
                x["cluster"],

            "solution_words":
                x["solution_words"],

            "difficulty_percentile":
                x["difficulty_percentile"],

            "difficulty_score":
                x["difficulty_score"],
        }
        for x in selected
    ]
)

metadata.to_csv(
    METADATA_FILE,
    index=False
)


# ============================================================
# 10. SUMMARY
# ============================================================

print()
print("========== COMPLETE ==========")
print("Selected:", len(selected))
print(
    "Clusters represented:",
    metadata["cluster"].nunique()
)
print(
    "Median solution words:",
    metadata["solution_words"].median()
)
print(
    "Mean solution words:",
    round(
        metadata["solution_words"].mean(),
        2
    )
)
print("Dataset:", OUTPUT_FILE)
print("Metadata:", METADATA_FILE)
