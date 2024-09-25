name=${name:-add_fire_abacus}
OUTPUT=eval_${name}
NGPUS=${NGPUS:-4}
# sbatch  shells/eval_addition.sh
if [[ "$SLURM_JOB_NAME" == "interactive" ]]; then
  bash shells/eval_addition.sh # > log/train_llama_$TYPE.log 2>&1 &
else
  sbatch -t 1:00:00 \
    --nodes=1 \
    --gres=gpu:$NGPUS \
    --mail-type=end \
    --mail-user=zhuconv@gmail.com \
    -o log/${OUTPUT}.log \
    -J ${OUTPUT} \
    shells/eval_addition.sh
fi