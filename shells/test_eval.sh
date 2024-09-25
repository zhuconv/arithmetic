#!/bin/bash
NGPUS=${NGPUS}
echo "NGPUS ${NGPUS}"

# Get the list of GPUs allocated to this job
GPUS=$SLURM_JOB_GPUS
echo "GPUS ${SLURM_JOB_GPUS}"

# Split the GPUS into an array
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"

# Loop over NGPUS tasks
for ((i=0; i<${NGPUS}; i++)); do
  # Compute the overall task index
  task_index=$(( SLURM_ARRAY_TASK_ID * NGPUS + i ))
  big_eval_step_num=$((task_index + 1))

  # Set the GPU for this task
  # echo ${GPU_ARRAY[$i]}

  # Run the command in the background
  CUDA_VISIBLE_DEVICES=$i
  python arithmetic_eval_quicker.py name=$name base_dir=$cramming_base_dir data=arithmetic max_rec=1 token_limit=105 \
    big_eval_step_${big_eval_step_num}=True reverse_inputs=True remove_padding=True \
    data.sources.arithmetic.tokenizer_type="pad" &

done

# Wait for all background tasks to finish
wait