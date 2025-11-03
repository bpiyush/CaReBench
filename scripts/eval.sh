split=$1
if [ -z "$split" ]; then
    split="nli-27k+ego4d-3k"
fi

device_map=$2
if [ -z "$device_map" ]; then
    device_map="cuda:0"
fi

# Merge weights
echo "Merging weights..."
python tasks/merge_weights.py \
    -b /work/piyush/pretrained_checkpoints/Tarsier-7b \
    -f /work/piyush/experiments/CaRe/Tarsier-7b/${split}

# Evaluate
echo "Evaluating..."
python notebooks/eval_care_retrieval.py \
    --model_id /work/piyush/experiments/CaRe/Tarsier-7b/${split}/merged_checkpoint \
    --device_map $device_map

echo "Done!"