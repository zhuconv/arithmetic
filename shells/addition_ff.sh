## FF
# norotary + nope
# rope,fire need arch.attention.type="self-attention" 
python pretrain.py name=add_bucket_20_20_reverse_all_pad_00_depthrec_16_1_TBPTT_1024_batch_size_512_mask_before_equals_true_start_emb_nope_attn_emb_nope_run_1 wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir impl.microbatch_size=256 budget=24 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=16 arch.maximal_recurrence=1 arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_intermediate_checkpoints=True impl.save_final_model=True data.sources.arithmetic.tokenized_dataset_path="+_bucket_method_n_20_m_20_20000000_p_00_reverse_all/hf_tokenized_dataset" train.optim.lr=0.0001 data.sources.arithmetic.tokenizer_type="pad" arch.mask_before_equals=True arch.embedding.pos_embedding=None

# fire + nope
python pretrain.py name=add_bucket_20_20_reverse_all_pad_00_depthrec_16_1_TBPTT_1024_batch_size_512_mask_before_equals_true_start_emb_nope_attn_emb_fire_run_1 wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir impl.microbatch_size=256 budget=24 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=16 arch.maximal_recurrence=1 arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_intermediate_checkpoints=True impl.save_final_model=True data.sources.arithmetic.tokenized_dataset_path="+_bucket_method_n_20_m_20_20000000_p_00_reverse_all/hf_tokenized_dataset" train.optim.lr=0.0001 data.sources.arithmetic.tokenizer_type="pad" arch.mask_before_equals=True arch.embedding.pos_embedding=None arch.attention.type="self-attention" arch.attention.rotary_embedding="fire" 

# norotary + abacus
python pretrain.py \
    name=add_bucket_20_20_reverse_all_pad_00_depthrec_16_1_TBPTT_1024_batch_size_512_mask_before_equals_true_start_emb_abacus_attn_emb_nope_run_1 \
    wandb=none \
    arch=crammed-depthrecurrent \
    data=arithmetic \
    base_dir=$cramming_base_dir \
    impl.microbatch_size=256 \
    budget=24 \
    impl.compile_torch=False \
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
    arch.embedding.pos_embedding=abacus

## FF w/ II
# nope
python pretrain.py name=add_bucket_20_20_reverse_all_pad_00_depthrec_16_1_TBPTT_1024_batch_size_512_mask_before_equals_true_start_emb_nope_attn_emb_nope_with_skip_connections_run_1 wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir impl.microbatch_size=512 budget=24 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=16 arch.maximal_recurrence=1 arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_intermediate_checkpoints=True impl.save_final_model=True data.sources.arithmetic.tokenized_dataset_path="+_bucket_method_n_20_m_20_20000000_p_00_reverse_all/hf_tokenized_dataset" train.optim.lr=0.0001 data.sources.arithmetic.tokenizer_type="pad" arch.mask_before_equals=True arch.embedding.pos_embedding=None arch.forward_only_model_with_skip=True
# fire
python pretrain.py name=add_bucket_20_20_reverse_all_pad_00_depthrec_16_1_TBPTT_1024_batch_size_512_mask_before_equals_true_start_emb_nope_attn_emb_fire_with_skip_connections_run_1 wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir impl.microbatch_size=512 budget=24 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=16 arch.maximal_recurrence=1 arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_intermediate_checkpoints=True impl.save_final_model=True data.sources.arithmetic.tokenized_dataset_path="+_bucket_method_n_20_m_20_20000000_p_00_reverse_all/hf_tokenized_dataset" train.optim.lr=0.0001 data.sources.arithmetic.tokenizer_type="pad" arch.mask_before_equals=True arch.embedding.pos_embedding=None arch.attention.type="self-attention" arch.attention.rotary_embedding="fire"  arch.forward_only_model_with_skip=True

# abacus
python pretrain.py \
    name=add_bucket_20_20_reverse_all_pad_00_depthrec_16_1_TBPTT_1024_batch_size_512_mask_before_equals_true_start_emb_abacus_attn_emb_nope_with_skip_connections_run_1 \wandb=none \
    arch=crammed-depthrecurrent \
    data=arithmetic \
    base_dir=$cramming_base_dir \
    impl.microbatch_size=512 \
    budget=24 \
    impl.compile_torch=False \
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
    arch.embedding.pos_embedding=abacus \
    arch.forward_only_model_with_skip=True


## FF w/ II
# Abacus + FIRE
python pretrain.py \
    name=add_bucket_20_20_reverse_all_pad_00_depthrec_16_1_TBPTT_1024_batch_size_256_mask_before_equals_true_start_emb_abacus_attn_emb_fire_with_skip_connections_run_1 \
    wandb=none \
    arch=crammed-depthrecurrent \
    data=arithmetic \
    base_dir=$cramming_base_dir \
    impl.microbatch_size=256 \
    budget=24 \
    impl.compile_torch=False \
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
    arch.attention.type="self-attention" \
    arch.attention.rotary_embedding="fire" \
    arch.forward_only_model_with_skip=True \
    arch.embedding.pos_embedding=abacus

## standard transformer
# RoPE + nope
torchrun --nproc_per_node=auto pretrain.py \
    impl.fullgraph=false \
    name=add_ronope \
    wandb=none \
    arch=crammed-depthrecurrent \
    data=arithmetic \
    base_dir=$cramming_base_dir \
    impl.microbatch_size=256 \
    budget=0.2 \
    impl.compile_torch=False \
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
    arch.attention.rotary_embedding=true

# FIRE + NoPE
torchrun --nproc_per_node=auto --master_port 25012 pretrain.py \
    impl.fullgraph=false \
    name=add_fire_no \
    wandb=none \
    arch=crammed-depthrecurrent \
    data=arithmetic \
    base_dir=$cramming_base_dir \
    impl.microbatch_size=256 \
    budget=0.5 \
    impl.compile_torch=False \
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
    arch.attention.rotary_embedding="fire"

# FIRE + Abacus
torchrun --nproc_per_node=auto --master_port 25012 pretrain.py \
    impl.fullgraph=false \
    name=add_fire_abacus \
    wandb=none \
    arch=crammed-depthrecurrent \
    data=arithmetic \
    base_dir=$cramming_base_dir \
    impl.microbatch_size=256 \
    budget=0.5 \
    impl.compile_torch=False \
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
    arch.embedding.pos_embedding="abacus" \
    arch.attention.max_length=64 \
    arch.attention.type="self-attention" \
    arch.attention.rotary_embedding="fire"

#  RoPE + Abacus
torchrun --nproc_per_node=auto pretrain.py \
    impl.fullgraph=false \
    name=add_ropecus \
    wandb=none \
    arch=crammed-depthrecurrent \
    data=arithmetic \
    base_dir=$cramming_base_dir \
    impl.microbatch_size=256 \
    budget=0.2 \
    impl.compile_torch=False \
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
    arch.embedding.pos_embedding=abacus \
    arch.attention.max_length=64 \
    arch.attention.type="self-attention" \
    arch.attention.rotary_embedding=true

## our method
torchrun --nproc_per_node=auto pretrain.py \
    impl.fullgraph=false \
    name=add_adayarn \
    wandb=none \
    arch=crammed-adadepthrecurrent \
    data=arithmetic \
    base_dir=$cramming_base_dir \
    impl.microbatch_size=256 \
    budget=0.8 \
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
    arch.embedding.pos_embedding=abacus \
    arch.attention.max_length=64 \
    arch.attention.type="self-attention" \
    arch.attention.rotary_embedding="adape"