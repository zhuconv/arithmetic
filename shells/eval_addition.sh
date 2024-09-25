#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=3

case $NGPUS in
  2)
    device_list=(0 1 0 1 0 1 0 1)
    ;;
  4)
    device_list=(0 1 2 3 0 1 2 3)
    ;;
  8)
    device_list=(0 1 2 3 4 5 6 7)
    ;;
  *)
    echo "Unsupported NGPUS value. Please use 2, 4, or 8."
    exit 1
    ;;
esac

echo "Device list: ${device_list[@]}"

for i in {0..7}; do
  CUDA_VISIBLE_DEVICES=${device_list[$i]} \
    python arithmetic_eval_quicker.py name=$name base_dir=$cramming_base_dir data=arithmetic max_rec=1 token_limit=105 \
    big_eval_step_$((i+1))=True reverse_inputs=True remove_padding=True data.sources.arithmetic.tokenizer_type="pad" &
done

wait
