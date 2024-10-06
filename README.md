<h1 align="center">Rethinking Addressing in Language Models via Contextualized Equivariant Positional Encoding (TAPE)</h1>
<p align="center">
    <a href=""><img src="https://img.shields.io/badge/-arXiv-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
    <a href=""><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
    <a href=""> <img alt="License" src="https://img.shields.io/static/v1?label=UR&message=ICLR%2725&color=blue"> </a>
</p>

This repository contains code to replicate the arithmetic learning experiments in our paper: [Rethinking Addressing in Language Models via Contextualized Equivariant Positional Encoding](). It is inherited from [mcleish7/arithmetic](https://github.com/mcleish7/arithmetic) and is a fork of the language model training framework [cramming](https://github.com/JonasGeiping/cramming) edited to for a next token prediction objective.


## Getting Started
We developed in Python 3.10.4, to install run:
```
git clone git@github.com:zhuconv/arithmetic.git
cd arithmetic
pip install .
```
On some machines you will need to run:
1. `pip install multiprocess -U`
2. `pip install dill -U`
3. `pip install apache-beam -U`


## File Structure
We recommend creating another directory `cramming-data` inside of arithmetic. This is where the models, logs and data will be stored.

You can either export you cramming base directory path to your `.bashrc` or you can replace `$cramming_base_dir` manually in the provided shells. For example, 
```
cd arithmetic
mkdir cramming-data
echo 'export cramming_base_dir=cramming-data' >> ~/.bashrc
source ~/.bashrc
```

Then our file system may looks like:
```
arithmetic
└── cramming-data
    ├── addition-train-one
    │    ├── pretrain/<DATE>/<TIME>
    │    │    ├── .hydra
    │    │    │   ├── config.yaml
    │    │    │   ├── hydra.yaml
    │    │    │   └── overrides.yaml
    │    │    └── addition-train-one_pretrain.log
    │    ├── checkpoints/FINAL_<LOSS_VAL>
    │    │    ├── model_config.json
    │    │    ├── model.safetensors
    │    │    └── state_dict.pth
    │    └── downstream
    └── data
        ├── +_grid_eval_dataset_reverse_all_tokenized
        └── ... other datasets ...
```

## Data
We use this addition dataset[+_bucket_method_n_20_m_20_20000000_p_00_reverse_all](https://drive.google.com/file/d/1xhPgxfJ96qFWrx6JTfJXUALSXRSDmaTq/view?usp=drive_link) hosted by [mcleish7/arithmetic](https://github.com/mcleish7/arithmetic) on Google Drive in zipped format.

## Training
We use commands under the [script](script) directory, for our method TAPE (`adape` in code),
```shell
TYPE=adape ARCH=crammed-adadepthrecurrent bash script/train_add.sh
```
For other baseline methods in `['rope', 'randomized', 'fire']`,
```shell
TYPE=fire ARCH=crammed-depthrecurrent bash script/train_add.sh
```

## Evaluation
Evaluating multiple lengths can be time-consuming, so we split the task into 8 sub-jobs by setting `big_eval_step_${num}=True`, where `num` ranges from 1 to 8. You can find an example script in [script/eval_add.sh](script/eval_add.sh). After running the evaluations, you can merge and visualize the overall results using [assets/draw_one.py](assets/draw_one.py).
