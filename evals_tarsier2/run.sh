split=$1
if [ -z "$split" ]; then
    split="nli-9k+ego4d-1k"
fi

device_map="cuda:0"

model_name=$2
if [ -z "$model_name" ]; then
    model_name="tarsier2_7b"
fi

BASE_MODEL=/work/piyush/pretrained_checkpoints/Tarsier2-7b-0115
base_model_name=$(basename $BASE_MODEL)
OUTPUT_DIR="/work/piyush/experiments/CaRe/${base_model_name}/${split}"
echo "Using output directory: $OUTPUT_DIR"

# Merge weights

# # Check if the merged checkpoint exists
# if [ ! -f "$OUTPUT_DIR/merged_checkpoint" ]; then
#     echo "Merging weights..."
#     python tasks/merge_weights_tarsier2.py \
#         -b $BASE_MODEL \
#         -f $OUTPUT_DIR
# fi


# Compute embeddings
echo "Computing embeddings..."
python evals_tarsier2/compute_embeddings.py \
    --model_path $OUTPUT_DIR/merged_checkpoint \
    --model_name $model_name 