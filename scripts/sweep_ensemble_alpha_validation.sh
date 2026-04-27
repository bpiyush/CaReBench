#!/usr/bin/env bash
# Parallel ensemble evaluation over alpha on validation embeddings; log one metric per alpha.
#
# Summary metric: mean of 7 heads —
#   time_v2t-ssv2: chiral/static/all R@1;
#   negation-msrvtt: standard & negation R@5;
#   multimodal_covr: covr R@1 and R@5.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TARA_FEAT="${TARA_FEAT:-/work/piyush/experiments/CaRe/Tarsier2-7b-0115/covr/chiral10k-covr10k/merged_checkpoint//embs/tarsier2_7b_nuanced_retrieval_data-validation-v1_embeddings.pt}"
QWEN_FEAT="${QWEN_FEAT:-/work/piyush/pretrained_checkpoints/Qwen3-VL-Embedding-8B/embs/qwen3vlemb_nuanced_retrieval_data-validation-v1_embeddings.pt}"
CSV_PATH="${CSV_PATH:-${REPO_ROOT}/data/nuanced_retrieval_data-validation-v1.csv}"
LAB_PATH="${LAB_PATH:-${REPO_ROOT}/data/nuanced_retrieval_labels-validation-v1.json}"
ALPHA_STEP="${ALPHA_STEP:-0.05}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/_ensemble_alpha_sweep_validation}"
LOG_TXT="${LOG_TXT:-${OUTDIR}/ensemble_alpha_sweep_summary.txt}"
CSV_BASENAME="$(basename "$CSV_PATH" .csv)"

# shellcheck disable=SC2155
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')"
if [[ -z "$NGPU" || "$NGPU" -lt 1 ]]; then
  NGPU=1
fi

mkdir -p "$OUTDIR"
rm -f "${OUTDIR}/.sweep_errors"

echo "Repo: $REPO_ROOT"
echo "GPUs (concurrent jobs): $NGPU"
echo "Alpha step: $ALPHA_STEP | Output: $OUTDIR"
echo "TARA: $TARA_FEAT"
echo "QWEN: $QWEN_FEAT"

i=0
while read -r alpha; do
  [[ -z "$alpha" ]] && continue
  gpu=$((i % NGPU))
  (
    set +e
    export CUDA_VISIBLE_DEVICES="$gpu"
    python "${REPO_ROOT}/scripts/ensemble_tara_q3vle.py" \
      --alpha "$alpha" \
      --tara_feat_path "$TARA_FEAT" \
      --qwen_feat_path "$QWEN_FEAT" \
      --csv_path "$CSV_PATH" \
      --lab_path "$LAB_PATH" \
      --output_dir "$OUTDIR"
    ec=$?
    if [[ $ec -ne 0 ]]; then
      echo "alpha=$alpha gpu=$gpu exit=$ec" >>"${OUTDIR}/.sweep_errors"
    fi
    exit "$ec"
  ) &
  i=$((i + 1))
  if (( i % NGPU == 0 )); then
    wait
  fi
done < <(LC_NUMERIC=C seq 0 "$ALPHA_STEP" 1)
wait

if [[ -s "${OUTDIR}/.sweep_errors" ]]; then
  echo "Some runs failed; see ${OUTDIR}/.sweep_errors" >&2
fi

python3 - "$OUTDIR" "$LOG_TXT" "$CSV_BASENAME" <<'PY'
import json
import os
import re
import sys
from datetime import datetime, timezone

outdir, log_path, csv_stem = sys.argv[1:4]


def avg_metric(X: dict) -> float:
    """Mean of 7 validation scalars (SSv2 v2t R@1 x3, MSRVTT negation R@5 x2, CoVR R@1/R@5)."""
    vals = []
    t = X.get("time_v2t-ssv2") or {}
    for mode in ("chiral", "static", "all"):
        v = (t.get(mode) or {}).get("R@1")
        if v is None:
            return float("nan")
        vals.append(float(v))
    n = X.get("negation-msrvtt") or {}
    v_s = ((n.get("standard") or {}).get("standard") or {}).get("R@5")
    v_n = ((n.get("negation") or {}).get("negation") or {}).get("R@5")
    if v_s is None or v_n is None:
        return float("nan")
    vals.extend([float(v_s), float(v_n)])
    c = (X.get("multimodal_covr") or {}).get("covr") or {}
    for k in ("R@1", "R@5"):
        v = c.get(k)
        if v is None:
            return float("nan")
        vals.append(float(v))
    return sum(vals) / len(vals)


# metrics_ensemble_tara_q3vle_a<alpha>_<csv_stem>.json — alpha uses 'p' for '.'
pat = re.compile(
    re.escape(f"metrics_ensemble_tara_q3vle_a") + r"(.+)_" + re.escape(csv_stem) + r"\.json$"
)
rows = []
for name in os.listdir(outdir):
    if not name.endswith(".json") or not name.startswith("metrics_ensemble_tara_q3vle_a"):
        continue
    m = pat.match(name)
    if not m:
        continue
    alpha_s = m.group(1).replace("p", ".")
    try:
        alpha = float(alpha_s)
    except ValueError:
        continue
    path = os.path.join(outdir, name)
    with open(path, encoding="utf-8") as f:
        X = json.load(f)
    rows.append((alpha, avg_metric(X), path))

rows.sort(key=lambda t: t[0])
finite_rows = [t for t in rows if t[1] == t[1]]  # exclude NaN averages
best = max(finite_rows, key=lambda t: t[1]) if finite_rows else None

lines = [
    f"# Written {datetime.now(timezone.utc).isoformat()}",
    "# alpha\tavg_metric\tjson_path",
    "# avg_metric = mean( time_v2t-ssv2 chiral/static/all R@1, negation-msrvtt std/neg R@5, covr R@1 R@5 )",
]
for alpha, avg, path in rows:
    lines.append(f"{alpha}\t{avg:.6f}\t{path}")
if best is not None:
    lines.append(f"# best_alpha\t{best[0]}\tbest_avg_metric\t{best[1]:.6f}")

os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Wrote summary ({len(rows)} alphas) to {log_path}")
if best:
    print(f"Best alpha (by avg metric): {best[0]}  avg={best[1]:.6f}")
PY

echo "Done. Summary: $LOG_TXT"
