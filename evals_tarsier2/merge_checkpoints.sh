ROOT_DIR=$1
if [ -z "$ROOT_DIR" ]; then
    echo "Usage: bash evals_tarsier2/merge_checkpoints.sh <root_dir>"
    echo "Example: bash evals_tarsier2/merge_checkpoints.sh /work/piyush/experiments/CaRe/Tarsier2-7b-0115/ablations"
    exit 1
fi

if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: $ROOT_DIR is not a valid directory"
    exit 1
fi

BASE_MODEL=/work/piyush/pretrained_checkpoints/Tarsier2-7b-0115
MIN_SIZE_GB=15

for folder in "$ROOT_DIR"/*/; do
    [ -d "$folder" ] || continue
    folder_name=$(basename "$folder")
    echo "=========================================="
    echo "Processing: $folder_name"

    num_files=$(ls -A "$folder" 2>/dev/null | wc -l)
    if [ "$num_files" -eq 0 ]; then
        echo "  Skipping: folder is empty"
        continue
    fi

    merged_dir="$folder/merged_checkpoint"
    if [ -d "$merged_dir" ]; then
        size_bytes=$(du -sb "$merged_dir" 2>/dev/null | cut -f1)
        size_gb=$(echo "scale=2; $size_bytes / 1073741824" | bc)
        if [ "$(echo "$size_gb >= $MIN_SIZE_GB" | bc)" -eq 1 ]; then
            echo "  Skipping: merged_checkpoint/ exists (${size_gb} GB)"
            continue
        else
            echo "  merged_checkpoint/ exists but is only ${size_gb} GB, re-merging..."
        fi
    fi

    echo "  Merging weights for $folder_name..."
    python tasks/merge_weights_tarsier2.py \
        -b "$BASE_MODEL" \
        -f "$folder"
done

echo "=========================================="
echo "Done."
