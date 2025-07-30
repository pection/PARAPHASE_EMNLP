1. Create and activate a Conda environment. 

```
conda create -n feb python=3.10.12
conda activate feb
```

2. Download the requirements. 

```
pip install -r requirements.txt
pip install --upgrade deepspeed
python -m spacy download en_core_web_sm

wandb offline
```

## Run evaluation

### T5/UnfifiedQA

The command below will train and evaluate given models on chosen datasets with 60 random seeds: 

```
python scripts/exp.py --exp_root <path_to_checkpoints_folder> --not_dryrun --model_vals <a string of models to evaluate separated by comma> --dataset_vals <a string of datasets to evaluate on, separated by comma> --n_gpus <number of available GPUs>
```

By default these experiments will be done with IO formats (prompts) that find to work the best (according to the experiments in the paper), but you can play around with different values in `format_dict` in `scripts/exp.py`.

The same command with concrete values: 

```
mkdir checkpoints
python scripts/exp.py --exp_root checkpoints --not_dryrun --model_vals t5-base,t5-large,t5-3b --dataset_vals esnli --n_gpus 4 --model_class t5
python scripts/exp.py --exp_root checkpoints --not_dryrun --model_vals allenai/unifiedqa-t5-base,allenai/unifiedqa-t5-large,allenai/unifiedqa-t5-3b --dataset_vals ecqa,sensemaking,sbic --n_gpus 4 --model_class t5
```


### Collect results 

After you're doing with training/eval with 60 seeds, you can collect results (mean, stddev) by running this: 

```
mkdir out
python scripts/exp.py --exp_root <path_to_checkpoints_folder>  --collect_results
```

If you get the assertion error, check which runs have not been trained properly, repeat evaluating only those seeds, and run the above command again. 

### Human evaluation
We use the NLEs associated with the first 30 correctly predicted samples in each validation set in the training for human evaluation. To make the
evaluation more robust, 30 samples were chosen to be balanced in the number of classes.

To get the generations from which to sample from, run:

```
bash get_generations.sh
```

# PARAPHASE_EMNLP
