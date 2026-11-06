"""Print bench/results.json as the markdown table used in bench/REPORT.md."""

import json
import sys

COLS = [
    ("dim", "dim", "{:.0f}"),
    ("infer_ms_per_track", "infer ms/track", "{:.0f}"),
    ("pair_auc_no_shared_artist", "pair AUC (no shared artist)", "{:.3f}"),
    ("centroid_recall@3", "LOO recall@3 centroid", "{:.3f}"),
    ("top3_recall@3", "LOO recall@3 top-3", "{:.3f}"),
    ("fixture_style_knn5_acc", "fixture style 5-NN", "{:.2f}"),
    ("fixture_style_pair_auc", "fixture style AUC", "{:.3f}"),
]

results = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "bench/results.json"))
print("| model | " + " | ".join(h for _, h, _ in COLS) + " |")
print("|---|" + "---:|" * len(COLS))
for name, r in sorted(results.items(), key=lambda kv: -kv[1]["pair_auc_no_shared_artist"]):
    print(f"| {name} | " + " | ".join(f.format(r[k]) if k in r else "–" for k, _, f in COLS) + " |")
