si=$1
ei=$2

# Calculate number of GPUs available
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs: $num_gpus"

# Calculate number of samples per GPU
total_samples=$((ei - si))
samples_per_gpu=$((total_samples / num_gpus))
remainder=$((total_samples % num_gpus))
echo "Total samples: $total_samples"
echo "Samples per GPU: $samples_per_gpu"
echo "Remainder: $remainder"

# Run parallelly across all GPUs
current_si=$si
for ((gpu=0; gpu<num_gpus; gpu++)); do
    # Calculate end index for this GPU
    current_ei=$((current_si + samples_per_gpu))
    # Distribute remainder across first few GPUs
    if [ $gpu -lt $remainder ]; then
        current_ei=$((current_ei + 1))
    fi
    
    echo "GPU $gpu: processing samples $current_si to $current_ei"
    CUDA_VISIBLE_DEVICES=$gpu python tasks/compute_features.py --si $current_si --ei $current_ei &
    
    current_si=$current_ei
done

wait
echo "All GPU processes completed."