split=$1
if [ -z "$split" ]; then
    split="nli-9k+ego4d-1k"
fi

device_map=$2
if [ -z "$device_map" ]; then
    device_map="cuda:0"
fi

BASE_MODEL=/work/piyush/pretrained_checkpoints/Tarsier-7b
base_model_name=$(basename $BASE_MODEL)
OUTPUT_DIR="/work/piyush/experiments/CaRe/${base_model_name}/${split}"
echo "Using output directory: $OUTPUT_DIR"

# Merge weights
echo "Merging weights..."
python tasks/merge_weights.py \
    -b $BASE_MODEL \
    -f $OUTPUT_DIR

# Evaluate on SSv2
echo "[::::::] Evaluating SSv2 [::::::]"
python notebooks/eval_care_retrieval.py \
    --model_id $OUTPUT_DIR/merged_checkpoint \
    --device_map $device_map \
    --dataset ssv2
echo "[:::::::::::::::::::::::::::::::]"

# # Evaluate on epic
# echo "[::::::] Evaluating EPIC [::::::]"
# python notebooks/eval_care_retrieval.py \
#     --model_id $OUTPUT_DIR/merged_checkpoint \
#     --device_map $device_map \
#     --dataset epic
# echo "[:::::::::::::::::::::::::::::::]"

# # Evaluate on charades
# echo "[::::::] Evaluating Charades [::::::]"
# python notebooks/eval_care_retrieval.py \
#     --model_id $OUTPUT_DIR/merged_checkpoint \
#     --device_map $device_map \
#     --dataset charades
# echo "[:::::::::::::::::::::::::::::::]"
