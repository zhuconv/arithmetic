export NGPUS=${NGPUS:-4}
export name=${name:-add_fire_abacus}
export OUTPUT=eval_${name}

TOTAL_TASKS=8
# Compute array length
ARRAY_LENGTH=$((TOTAL_TASKS / NGPUS))

# Submit the job
sbatch -t 1:00:00 \
    --nodes=1 \
    --gres=gpu:$NGPUS \
    --array=0-$(($ARRAY_LENGTH - 1)) \
    --mail-type=end \
    --mail-user=zhuconv@gmail.com \
    -o log/${OUTPUT}.log \
    -J ${OUTPUT} \
    shells/test_eval.sh