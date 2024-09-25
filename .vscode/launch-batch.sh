#!/bin/bash

JOB=$RANDOM
TEMP_FILE="../tmp/vscode-sbatch-script.$JOB.slurm"

echo "Temporary job file: $TEMP_FILE" # show file name
# echo "Current running file: $1" # show which script is being run

# Cancel job on Ctrl+C Keyboard Interrupt
JOBNAME="debugpy${JOB}"
trap "scancel -n $JOBNAME; exit 130" SIGINT

cat >$TEMP_FILE << EOL
#!/bin/bash
#SBATCH --job-name=$JOBNAME       
#SBATCH -o $JOBNAME.log                               
#SBATCH -e $JOBNAME.log  
#SBATCH --get-user-env  

unset TORCH_DISTRIBUTED_DEBUG
python -m debugpy --wait-for-client --listen 0.0.0.0:3000 -m torch.distributed.launch --nproc_per_node auto --use-env \
    pretrain.py \
    impl.fullgraph=false \
    name=add_debug \
    wandb=none \
    arch=crammed-adadepthrecurrent \
    data=arithmetic \
    base_dir=/scratch/gpfs/pw4811/arithmetic/cramming-data \
    impl.microbatch_size=256 \
    budget=1 \
    impl.compile_torch=False \
    arch.local_compilation=False \
    arch.objective_layout=TBPTT \
    arch.layers_in_recurrent_block=16 \
    arch.maximal_recurrence=1 \
    arch.hidden_size=1024 \
    arch.intermed_size=2048 \
    impl.forbid_dataset_preprocessing=False \
    impl.save_intermediate_checkpoints=True \
    impl.save_final_model=True \
    data.sources.arithmetic.tokenized_dataset_path="+_bucket_method_n_20_m_20_20000000_p_00_reverse_all/hf_tokenized_dataset" \
    train.optim.lr=0.0001 \
    data.sources.arithmetic.tokenizer_type="pad" \
    arch.mask_before_equals=True \
    arch.embedding.pos_embedding=None \
    arch.attention.max_length=64 \
    arch.attention.type="self-attention" \
    arch.attention.rotary_embedding="adape"
EOL
# python -m debugpy --wait-for-client --listen 0.0.0.0:3000 -m torch.distributed.launch --nproc_per_node auto --use-env  train_longlora.py \
#   --ddp_timeout 18000 \
#   --model_name_or_path meta-llama/Llama-2-7b-hf \
#   --peft_type lora \
#   --bf16 True \
#   --resume_from_checkpoint false \
#   --output_dir ./output/longlora_llama \
#   --model_max_length 8192 \
#   --use_flash_attn True \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --gradient_accumulation_steps 8 \
#   --evaluation_strategy no --save_strategy steps \
#   --save_steps 200 --save_total_limit 2 \
#   --learning_rate 2e-5 --weight_decay 0.0 \
#   --warmup_steps 20 --lr_scheduler_type constant_with_warmup \
#   --logging_steps 1 \
#   --deepspeed config/ds_configs/stage2.json \
#   --tf32 True \
#   --max_steps 1000 \
#     --report_to none
# EOL
 
# display the job submission file
echo " ///////////////////////////////////////// "
echo "   Job submission file : $TEMP_FILE "
echo " ///////////////////////////////////////// "
cat $TEMP_FILE
echo " ///////////////////////////////////////// "

# submit the job
# SLURM_SBATCH_FLAGS=$2
sbatch --nodes=1 --ntasks=1 --time=1:00:00 --gres=gpu:2 --cpus-per-task=16 $TEMP_FILE
# sbatch --get-user-env $SLURM_SBATCH_FLAGS $TEMP_FILE

echo 'Waiting for Slurm job to begin..'
while true; do
 export JOB_STATUS=$(squeue -n $JOBNAME --format="%.2t" | tail -n1 | xargs)
 echo "Job Status : $JOB_STATUS"
 if [ "$JOB_STATUS" == "R" ]; then
   echo "Job started!"
   break
 else
   sleep 2
   tput cuu 1
 fi
done

ln -sf $JOBNAME.log debugpy_latest.log
echo $JOBNAME > debugpy-jobid

# Give the script some time to install and start debugpy server
sleep 15