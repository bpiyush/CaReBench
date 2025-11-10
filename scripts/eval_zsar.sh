MODEL_PATH=$1
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="/work/piyush/experiments/CaRe/Tarsier-7b/nli-9k+ego4d-1k/merged_checkpoint"
fi

python tasks/eval_zsar.py \
    --model_path $MODEL_PATH \
    --dataset ucf101 \
    --model tarsier7b+tara

python tasks/eval_zsar.py \
    --model_path $MODEL_PATH \
    --dataset hmdb51 \
    --model tarsier7b+tara

python tasks/eval_zsar.py \
    --model_path $MODEL_PATH \
    --dataset epic \
    --model tarsier7b+tara

python tasks/eval_zsar.py \
    --model_path $MODEL_PATH \
    --dataset kinetics-verbs \
    --model tarsier7b+tara