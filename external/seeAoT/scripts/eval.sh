DATA_FILES=(
    "data/data_files/input/ReverseFilm.json"
    "data/data_files/input/UCF101.json"
    "data/data_files/input/Rtime_t2v.json"
    "data/data_files/input/Rtime_v2t.json"
    "data/data_files/input/AoTBench_QA.json"
) 

CKPTS=(
    # "/work/piyush/pretrained_checkpoints/Qwen2.5-VL-7B-Instruct"
    # "/work/piyush/pretrained_checkpoints/ArrowRL-Qwen2.5-VL-7B"
    # "/work/piyush/experiments/CaRe/special_milestones/care-stage2-nli90k-ego4d-10k"
    "/work/piyush/pretrained_checkpoints/CaRe-7B/"
)
EVAL_SCRIPTS=(
    # "eval/run_qwen25.py"
    # "eval/run_qwen25.py"
    # "eval/run_qwen2.py"
    "eval/run_qwen2.py"
)


for DATA_FILE in "${DATA_FILES[@]}"; do
    for CKPT in "${CKPTS[@]}"; do
        echo "Evaluating $CKPT on $DATA_FILE with script ${EVAL_SCRIPTS[$i]}"
        python ${EVAL_SCRIPTS[$i]} --data_json "$DATA_FILE" --ckpt $CKPT
    done
done