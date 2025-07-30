# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This code is based on https://github.com/allenai/label_rationale_association/blob/main/input_to_label_and_rationale.py

import logging
import math
import os
import torch
import torch.nn as nn
from typing import List, Dict, Any, NewType
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import Seq2SeqLMOutput
from custom_loss_class import SPLINT_T5
from transformers import TrainingArguments as HFTrainingArguments
from dataclasses import dataclass, field
from typing import Optional
InputDataClass = NewType("InputDataClass", Any)
from itertools import product
from transformers import HfArgumentParser
from custom_args import ModelArguments, DataTrainingArguments
from datetime import datetime
import os
import matplotlib.pyplot as plt


from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback
)
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, AdaLoraConfig, IA3Config
from transformers.trainer_utils import EvaluationStrategy
from transformers.integrations import TensorBoardCallback
import transformers
from transformers import Trainer, AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, DataCollatorForLanguageModeling

from feature_conversion_methods import format_instance

from custom_args import (
    DataTrainingArguments,
    ModelArguments
)
from metrics_custom_loss import evaluate
import torch
import datasets
import time
from datetime import datetime
import sys
from tqdm import trange
import random 
import pandas as pd 
import jsonlines
from copy import deepcopy
from typing import List, Dict
from accelerate import Accelerator

logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_info()
import re
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import json

def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
set_global_logging_level(logging.ERROR, ["datasets"])


CONFIG_MAPPING = {"t5": T5Config, "llama": LlamaConfig}
MODEL_MAPPING = {"t5": T5ForConditionalGeneration, "llama": LlamaForCausalLM}
TOKENIZER_MAPPING = {"t5": T5Tokenizer, "llama": LlamaTokenizer}
model_class = "llama"


def set_other_seeds(seed):
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

# inspired by DefaultDataCollator from:
# https://github.com/huggingface/transformers/blob/master/src/transformers/data/data_collator.py
# modified to perform batch-level padding.


class DebugTrainer(Trainer):
    def __init__(self, *args, tokenizer=None,json_dir="./", **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer  # Store tokenizer
        self.json_dir = json_dir   # Store json_dir

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        # loss = outputs['loss']
        loss = outputs.loss
        logits = outputs.logits
        # logits = outputs['logits']

        if loss.item() > 0.0:
            input_ids = inputs.get("input_ids")
            labels = inputs.get("labels")

            decoded_inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            predicted_ids = torch.argmax(logits, dim=-1)
            decoded_preds = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)


            json_file_name =os.path.join(self.json_dir,"high_loss_samples.jsonl")
            with open(json_file_name, "a") as f:
                for i in range(min(3, len(decoded_inputs))):
                    f.write(json.dumps({
                        "loss": float(loss.item()),
                        "input": decoded_inputs[i],
                        "label": decoded_labels[i],
                        "prediction": decoded_preds[i]
                    }) + "\n")


        return (loss, outputs) if return_outputs else loss

class SequenceCollator:
    def __init__(self, model, pad_token):
        self.model = model
        self.pad_token_mapping = {
            "labels": -100,
            "attention_mask": 0,
            "decoder_attention_mask": 0,
            "input_ids": pad_token,
        }

        self.columns = [
            "input_ids",
            "attention_mask",
            "labels",
            "decoder_attention_mask",
        ]

    def __call__(self, examples: List[Dict[str, InputDataClass]]) -> Dict[str, torch.Tensor]:
        # re-format inputs for training
        batch = {}
        for key in examples[0].keys():
            if key in self.columns:
                tmp_list = []
                for item in examples:
                    tmp_list.append(item[key])

                # pad lists to max length
                if isinstance(tmp_list[0], list):
                    max_length = max(map(len, tmp_list))
                    tmp_list = [
                        el + [self.pad_token_mapping[key]] * (max_length - len(el))
                        for el in tmp_list
                    ]

                batch[key] = torch.tensor(tmp_list, dtype=torch.long)
        return batch

class CausalLMCollator:
    def __init__(self, model, pad_token):
        self.model = model
        self.pad_token_mapping = {
            "labels": -100,
            "attention_mask": 0,
            "input_ids": pad_token,
        }

        self.columns = [
            "input_ids",
            "attention_mask",
            "labels"
        ]

    def __call__(self, examples: List[Dict[str, InputDataClass]]) -> Dict[str, torch.Tensor]:
        # re-format inputs for training
        batch = {}

        max_length = None
        for example in examples:
            for key in example.keys():
                if key in self.columns:
                    if max_length is None or max_length < len(example[key]):
                        max_length = len(example[key])

        for key in examples[0].keys():
            if key in self.columns:
                tmp_list = []
                for item in examples:
                    tmp_list.append(item[key])

                # pad lists to max length
                if isinstance(tmp_list[0], list):
                    # max_length = max(map(len, tmp_list))
                    tmp_list = [
                        el + [self.pad_token_mapping[key]] * (max_length - len(el))
                        for el in tmp_list
                    ]

                batch[key] = torch.tensor(tmp_list, dtype=torch.long)
        return batch

@dataclass
class TrainingArguments(HFTrainingArguments):
    save_weight: bool = field(
        default=False,
        metadata={"help": "Save model weights after training"}
    )
@dataclass
class DataTrainingArguments(HfArgumentParser):
    data_path: Optional[str] = field(default=None)  

def train_model(model_args, data_args, training_args):

    accelerator = Accelerator()
    og_start_time = time.time()

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if model_args is None or data_args is None or training_args is None:
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        unused_args = []
    else:
        unused_args = []
    # model_args, data_args, training_args, unused_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    print("*********************************")
    print(f"model_args = {model_args}")
    print(f"data_args = { data_args}")
    print(f"training_args = {training_args}")
    print(f"unused_args = {unused_args}")
    print("*********************************")

    data_args.dev_predict = True
    if data_args.generations_filepath is not None:
        training_args.do_train = False
        training_args.do_eval = False
        if "train" in data_args.generations_filepath:
            data_args.train_predict = True
            data_args.test_predict = False
            data_args.dev_predict = False
        elif "test" in data_args.generations_filepath:
            data_args.train_predict = False
            data_args.test_predict = True
            data_args.dev_predict = False
        elif "validation" in data_args.generations_filepath:
            data_args.train_predict = False
            data_args.test_predict = False
            data_args.dev_predict = True

    if not training_args.do_train and data_args.generations_filepath is None:
        if not model_args.pretrained_model_file:
            raise Exception(
                "if not training a model from scratch, must specify a trained model to load for evaluation"
            )
        

    if training_args.do_train:
        training_args.output_dir = os.path.join(
            training_args.output_dir, datetime.now().strftime("%m%d%y_%H%M%S")
        )
        training_args.logging_dir = training_args.output_dir
        os.makedirs(training_args.output_dir, exist_ok=True)

        handlers = [
            logging.FileHandler(os.path.join(training_args.output_dir, "logger.log")),
            logging.StreamHandler(),
        ]
    else:
        # don't overwrite existing logfile or create new directory
        training_args.output_dir = model_args.pretrained_model_file
        handlers = [logging.StreamHandler()]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=handlers,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.bf16,
    )
    logger.info("Save path: %s" % training_args.output_dir)


    # model_class = "llama"
    model_class = model_args.model_class
    assert data_args.task_name in {"cos_e", "esnli", "sbic", "sensemaking", "ecqa"}
    if training_args.do_train:
        with open(
                os.path.join(training_args.output_dir, "commandline_args.txt"), "w"
        ) as f:

            f.write("Command:\n")
            f.write("\n".join(sys.argv[1:]))

    # Set seed
    set_seed(training_args.seed)
    set_other_seeds(training_args.seed)

    if "t5" in model_args.model_class:
        model_class = "t5"

    if model_class == "t5":
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        model = SPLINT_T5.from_pretrained("google/flan-t5-large")
    model.resize_token_embeddings(len(tokenizer))        

    

    data_splits = {'train': None, 'validation': None, 'test': None}
    original_data_splits = {'train': None, 'validation': None, 'test': None}  

    # Data loading from huggingface's datasets
    if data_args.task_name in {"cos_e", "esnli"}:
        version_arg = None
        if data_args.task_name == "cos_e":
            assert data_args.version_name in {"v1.11", "v1.0"}
            version_arg = data_args.version_name
            print(f"version_arg = {version_arg}")

        load_train = True
        if (not training_args.do_train
            and not training_args.do_eval
            and not data_args.train_predict
        ):
            # don't load training dataset
            dataset = {}
            dataset["train"] = None
            dataset["validation"] = datasets.load_dataset(
                data_args.task_name, version_arg, split="validation"
            )
            data_splits['validation'] = dataset["validation"]

            if data_args.task_name == "esnli":
                dataset["test"] = datasets.load_dataset(data_args.task_name, split="test")
                data_splits['test'] = dataset["test"]
            load_train = False
        else:
            dataset = datasets.load_dataset(data_args.task_name, version_arg)
            print(f"LOAD DATASET SUCCESS")
            print(data_args.task_name, version_arg)
            if data_args.n_shots > 0: # Shots = number of training examples **per label** 
                if data_args.task_name == 'esnli': # Construct a *balanced* random sample of the size `data_args.n_shots*len(labels)` (for train) or `data_args.fewshot_eval_size` (for eval)
                    for split in ["train", "validation", "test"]:
                        split_data = dataset[split]
                        label_subsets = []
                        labels = split_data.features['label'].names
                        sample_size = data_args.n_shots if split == "train" else int(data_args.fewshot_eval_size/len(labels))
                        if data_args.gpt3_max_eval_size is not None and split != 'train':
                            assert len(labels) == 3
                            sample_size = data_args.gpt3_max_eval_size // len(labels)
                        for label in labels:
                            label_int = split_data.features['label'].str2int(label)
                            label_set = split_data.filter(lambda example: example['label'] == label_int).shuffle() # all instances of labeled as `label`
                            label_subset = label_set.select(range(sample_size)) #select `sample_size` random instances labeled as `label`
                            label_subsets.append(label_subset)
                        dataset[split] = datasets.concatenate_datasets(label_subsets) #merge all label-specific instances
                elif data_args.task_name == 'cos_e': 
                    for split in ["train", "validation"]: 
                        split_data = dataset[split]
                        sample_size = data_args.n_shots if split == "train" else int(data_args.fewshot_eval_size) #Shots for QA are not label-specific, i.e., `n_shots` is the training data size
                        if data_args.gpt3_max_eval_size is not None and split != 'train':
                            sample_size = data_args.gpt3_max_eval_size
                        dataset[split] = split_data.shuffle().select(range(sample_size)) # select `sample_size` random instances
                else: 
                    raise ValueError('Only cos_e and esnli are supported by Huggingface datasets.')
        for split in dataset.keys():
            if dataset[split] is not None:
                dataset[split] = dataset[split].map(
                    lambda x: format_instance(
                        x,
                        tokenizer,
                        data_args.explanation_sep,
                        datasource=data_args.task_name,
                        io_format=data_args.io_format
                    ),
                    batched=False,
                    load_from_cache_file=False,
                )
        data_splits["train"] = deepcopy(dataset["train"])
        data_splits["validation"] = deepcopy(dataset["validation"])
        if data_args.task_name == "esnli":
            data_splits["test"] = deepcopy(dataset["test"])

        original_data_splits["train"] = deepcopy(dataset["train"])
        original_data_splits["validation"] = deepcopy(dataset["validation"])
        if data_args.task_name == "esnli":
            original_data_splits["test"] = deepcopy(dataset["test"])


    elif data_args.task_name == "sbic":
        split_mapping = {'trn': 'train', 'dev': 'validation', 'tst': 'test'}
        splits = ['trn', 'dev', 'tst'] if training_args.do_train else ['dev', 'tst']
        load_train = True if training_args.do_train else False 
        n_labels = 2 # two labels: offensive, not offensive 
        for split in splits:
            data_splits[split_mapping[split]] = []
            if not training_args.do_train:
                continue
            data_path = os.path.join(os.getcwd(), data_args.data_path.lstrip('../'), f"SBIC.v2.{split}.modified.csv")
            df = pd.read_csv(data_path)

            if data_args.n_shots > 0: # This condition could probably be removed; we used n_shots=0 to experiment with training with the entire train set
                # Here we create a balanced training set with `data_args.n_shots` examples per label
                not_offensive_df = df.loc[df["offensiveYN"]=="not offensive"]
                frac1 = data_args.n_shots/len(not_offensive_df) if split == 'trn' else int(data_args.fewshot_eval_size/n_labels)/len(not_offensive_df)
                offensive_df = df.loc[df["offensiveYN"]=="offensive"]
                frac2 = data_args.n_shots/len(offensive_df) if split == 'trn' else int(data_args.fewshot_eval_size/n_labels)/len(offensive_df)
                label1_data = not_offensive_df.sample(frac=frac1, replace=False)
                label2_data = offensive_df.sample(frac=frac2, replace=False)
                if data_args.gpt3_max_eval_size is not None and split != 'trn':
                    label1_data = label1_data[:data_args.gpt3_max_eval_size // 2]
                    label2_data = label2_data[:data_args.gpt3_max_eval_size // 2]
                df = pd.concat([label1_data, label2_data])

            for i in trange(len(df["targetStereotype"])):
                new_encoded = format_instance(
                    df.iloc[i],
                    tokenizer,
                    data_args.explanation_sep,
                    datasource=data_args.task_name,
                    io_format=data_args.io_format
                )
                data_splits[split_mapping[split]].append({**df.iloc[i], **new_encoded})
            original_data_splits[split_mapping[split]] = deepcopy(data_splits[split_mapping[split]])
    

    elif data_args.task_name == "sensemaking":
        split_mapping = {'Training': 'train', 'Dev': 'validation', 'Test': 'test'}
        splits = ['Training', 'Dev', 'Test'] if training_args.do_train else ['Dev', 'Test']
        load_train = True if training_args.do_train else False 
        n_labels = 2 # two labels: choice1, choice2
        for split in splits:
            data_splits[split_mapping[split]] = []
            if not training_args.do_train:
                continue
            data_path = os.path.join(os.getcwd(), data_args.data_path.lstrip('../'), f"SenMaking.{split}.csv")
            df = pd.read_csv(data_path)

            if data_args.n_shots > 0: # This condition could probably be removed; we used n_shots=0 to experiment with training with the entire train set
                # Here we create a balanced training set with `data_args.n_shots` examples per label
                choice1_df = df.loc[df["label"]==0]
                frac1 = data_args.n_shots/len(choice1_df) if split == 'Training' else int(data_args.fewshot_eval_size/n_labels)/len(choice1_df)
                choice2_df = df.loc[df["label"]==1]
                frac2 = data_args.n_shots/len(choice2_df) if split == 'Training' else int(data_args.fewshot_eval_size/n_labels)/len(choice2_df)
                label1_data = choice1_df.sample(frac=frac1, replace=False)
                label2_data = choice2_df.sample(frac=frac2, replace=False)
                if data_args.gpt3_max_eval_size is not None and split != 'Training':
                    label1_data = label1_data[:data_args.gpt3_max_eval_size // 2]
                    label2_data = label2_data[:data_args.gpt3_max_eval_size // 2]
                df = pd.concat([label1_data, label2_data])
            
            for i in trange(len(df)):
                new_encoded = format_instance(
                    df.iloc[i],
                    tokenizer,
                    data_args.explanation_sep,
                    datasource=data_args.task_name,
                    io_format=data_args.io_format
                )
                data_splits[split_mapping[split]].append({**df.iloc[i], **new_encoded})
            original_data_splits[split_mapping[split]] = deepcopy(data_splits[split_mapping[split]])

    elif data_args.task_name == 'ecqa': 
        for split in ["train", "validation"]: 
            ecqa_data_split = []
            data_path = os.path.join(os.getcwd(), data_args.data_path.lstrip('../'), f"ecqa_{split}.jsonl")
            with jsonlines.open(data_path) as ecqa_split_reader:
                for item in ecqa_split_reader: 
                    formatted_instance = format_instance(item,
                                                         tokenizer,
                                                         data_args.explanation_sep,
                                                         datasource=data_args.task_name,
                                                         io_format=data_args.io_format)
                    ecqa_data_split.append(formatted_instance)
            sample_size = data_args.n_shots if split == "train" else int(data_args.fewshot_eval_size)
            if data_args.gpt3_max_eval_size is not None and split != 'train':
                sample_size = data_args.gpt3_max_eval_size
            data_splits[split] = random.sample(ecqa_data_split, sample_size)
            original_data_splits[split] = deepcopy(data_splits[split])
    else: 
        raise ValueError("Unknown task. Currently supported: esnli, cos_e, sbic, sensemaking, ecqa.")

    logger.info("****LOG****")
    for split in ['train', 'validation', 'test']:
        if data_splits[split]:
            logger.info(split)
            logger.info(len(data_splits[split]))

    
    '''
    if data_args.n_shots > 0: 
        import jsonlines 
        with jsonlines.open(os.path.join(training_args.output_dir,'train.json'), 'w') as writer:
            for item in original_data_splits['train']:
                writer.write(item)
        with jsonlines.open(os.path.join(training_args.output_dir,'validation.json'), 'w') as writer:
            for item in original_data_splits['validation']:
                writer.write(item)
    '''

    if data_args.generations_filepath is None:
        callbacks = [TensorBoardCallback()]
        if data_args.early_stopping_patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=data_args.early_stopping_patience))
            training_args.load_best_model_at_end = True
        else:
            training_args.load_best_model_at_end = False  # use the last model state
        training_args.metric_for_best_model = 'eval_loss'
        training_args.greater_is_better = False
        if training_args.eval_steps is None:
            training_args.evaluation_strategy = EvaluationStrategy.EPOCH
        else:
            training_args.evaluation_strategy = EvaluationStrategy.STEPS
        
        
        # # SPARSEFIT CHANGES
        # Make trainable only key terms in self-attention layers.
        # *** LMHEAD FREEZE ***
        # for param in model.parameters():
        #     param.requires_grad = True


        # for name, param in model.named_parameters():            
        #     if 'SelfAttention.q' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        # model.lm_head.weight.requires_grad = True
        # for param in model.lm_head.parameters():
        #     param.requires_grad = True

        # for name, param in model.named_parameters():
        #     if "dec" in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False


        trainable = [name for name, param in model.named_parameters() if param.requires_grad]
        print("Trainable parameters:")
        for name in trainable:
            print(name)


        # for name, param in model.named_parameters():
        #     if name.startswith("encoder"):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        # for name, param in model.named_parameters():
        #     if 'self_attn.q_proj' in name:
        #         param.requires_grad = True

        # for name, param in model.named_parameters():
        #     if 'layernorm' in name:
        #         param.requires_grad = True


        peft_config = LoraConfig(
            peft_type="LORA",
            task_type="SEQ_2_SEQ_LM",
            r=4,  # LoRA rank
            lora_alpha=32,
            target_modules=[
                "q", "k", "v", "o", "wi_0", "wi_1", "wo"  # T5 decoder layers
            ],
            lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)

        training_args.bf16=True
        training_args.bf16_full_eval=True

        def process_data_split(data_split):
          sample_input_ids = data_split["input_ids"]
          label_input_ids = data_split["labels"]
          data_split['input_ids'] = sample_input_ids + label_input_ids
          data_split["labels"] = [tokenizer.pad_token_id] * len(sample_input_ids) + label_input_ids
          data_split['attention_mask'] = [1] * len(data_split['input_ids'])
          return data_split

        trainer = DebugTrainer(
            model=model,
            args=training_args,
            train_dataset=data_splits['train'],
            eval_dataset=data_splits['validation'],
            tokenizer=tokenizer,  
            json_dir = training_args.output_dir,
            callbacks=callbacks,
            data_collator=SequenceCollator(
                model=model_class, pad_token=tokenizer.pad_token_id
            ),
        )
    
        
    # Training. Don't train if it is use_gpt3
    if training_args.do_train and not model_args.use_gpt3:
        start_time = time.time()
        trainer.train()
        train_time = time.time() - start_time
        model = trainer.model
        train_loss = []
        eval_loss = []
        train_steps = []
        eval_steps = []
        for record in trainer.state.log_history:
            if 'loss' in record and 'eval_loss' not in record:
                train_loss.append(record['loss'])
                train_steps.append(record['step'])
            elif 'eval_loss' in record:
                eval_loss.append(record['eval_loss'])
                eval_steps.append(record['step'])
        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, train_loss, label='Training Loss', marker='o')
        plt.plot(eval_steps, eval_loss, label='Validation Loss', marker='x')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss Curve')
        plt.legend()
        plt.grid(True)
        loss_plot_path = os.path.join(training_args.output_dir, 'loss_curve.png')
        plt.savefig(loss_plot_path)
        print(f"Loss curve saved to {loss_plot_path}")
        if training_args.save_weight:
            print(f"-----Save model weight-----")
            trainer.save_model(training_args.output_dir)  # Saves model, config, and tokenizer (if assigned to Trainer)
            tokenizer.save_pretrained(training_args.output_dir)

    else:
        start_time = time.time()
        train_time = time.time() - start_time

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        unwrap_model = accelerator.unwrap_model(model)
    else:
        unwrap_model = accelerator.unwrap_model(model)

    results = {}
    if training_args.do_eval:
        start_time = time.time()
        logger.info("*** Evaluate on train set***")
        logger.info(len(data_splits['train']))
        train_output = trainer.evaluate(data_splits['train'])
        perplexity = math.exp(train_output["eval_loss"])
        results["perplexity_train"] = perplexity

        logger.info("*** Evaluate on dev set***")
        logger.info(len(data_splits['validation']))
        eval_output = trainer.evaluate(data_splits['validation'])
        perplexity = math.exp(eval_output["eval_loss"])
        results["perplexity_validation"] = perplexity

        if data_args.task_name in {"esnli", "sbic"}:
            # also evaluate on test
            logger.info("*** Evaluate on test set***")
            logger.info(len(data_splits["test"]))
            test_output = trainer.evaluate(data_splits["test"])
            logger.info("test loss @ best dev epoch: %0.4f" % test_output["eval_loss"])
            perplexity = math.exp(test_output["eval_loss"])
            results["perplexity_test"] = perplexity

        eval_time = time.time() - start_time

    if data_args.generations_filepath is None:
        save_path = trainer.state.best_model_checkpoint
        if save_path is None:  
            save_path = training_args.output_dir
        model.eval()
    else:
        save_path = os.path.dirname(data_args.generations_filepath)

    start_time = time.time()
    # Storing predictions & computing BLEUscore. Don't predict on the training set if `use_gpt3`
    # `data_args.train_predict` is NOT used for experiments in the paper
    if data_args.train_predict :
        logger.info("*** Predict on train set***")
        if data_args.generations_filepath is not None:
            assert "train" in data_args.generations_filepath

        results = evaluate(
                            save_path,
                            original_data_splits['train'],
                            model,
                            tokenizer,
                            "train",
                            data_args.task_name,
                            training_args.device,
                            data_args.explanation_sep,
                            rationale_only=model_args.rationale_only,
                            generations_file=data_args.generations_filepath,
                            io_format=data_args.io_format
                            )
    # `data_args.test_predict` is NOT used for experiments in the paper
    if data_args.test_predict and data_args.task_name in {"esnli", "sbic"}:
        logger.info("*** Predict on test set***")
        print(f"************************ MODEL WILL EVALUATE TEST ************************")
        results = evaluate(
                            save_path,
                            original_data_splits['test'],
                            model,
                            tokenizer,
                            "test",
                            data_args.task_name,
                            training_args.device,
                            data_args.explanation_sep,
                            rationale_only=model_args.rationale_only,
                            generations_file=data_args.generations_filepath,
                            io_format=data_args.io_format
                            )
    if data_args.dev_predict:
        logger.info("*** Predict on dev set***")
        if data_args.generations_filepath is not None:
            assert "validation" in data_args.generations_filepath
        if model_args.pretrained_model_file and not training_args.do_train:
            save_path = model_args.pretrained_model_file
        print(f"************************ MODEL WILL EVALUATE VALIDATION ************************")
        results = evaluate(
                            save_path,
                            original_data_splits["validation"],
                            model,
                            unwrap_model,
                            tokenizer,
                            "validation",
                            data_args.task_name,
                            training_args.device,
                            data_args.explanation_sep,
                            rationale_only=model_args.rationale_only,
                            generations_file=data_args.generations_filepath,
                            io_format=data_args.io_format
                        )

    if data_args.generations_filepath is None:
        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
    else:
        output_eval_file = os.path.join(
            os.path.dirname(os.path.dirname(data_args.generations_filepath)),
            "eval_results_lm.txt",
        )
    # if data_args.generations_filepath or trainer.is_world_process_zero():


    if data_args.generations_filepath is None:
        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
    else:
        output_eval_file = os.path.join(
            os.path.dirname(os.path.dirname(data_args.generations_filepath)),
            "eval_results_lm.txt",
        )
    print(f"KEYS = = == =")
    for key in results.keys():
        print(key)
    if data_args.generations_filepath or trainer.is_world_process_zero():
        with open(output_eval_file, "a+") as writer:
            for key in results.keys():
                if results[key] is not None:
                    logger.info("  %s = %s", key, str(results[key])) # logging this is important for collecting results later
                    writer.write("%s = %s\n" % (key, str(results[key])))

    predict_time = time.time() - start_time
    logger.info("Save path: %s" % training_args.output_dir)
    if training_args.do_train:
        logger.info("total train time: %.4f hours" % (train_time / 60.0 / 60.0))
    if training_args.do_eval:
        logger.info("total eval time: %.4f hours" % (eval_time / 60.0 / 60.0))
    if (
            data_args.train_predict
            or data_args.dev_predict
            or (data_args.test_predict and data_args.task_name in {"esnli", "sbic"})
    ):
        logger.info("total predict time: %.4f hours" % (predict_time / 60.0 / 60.0))
    logger.info(
        "TOTAL SCRIPT TIME: %.4f hours" % ((time.time() - og_start_time) / 60.0 / 60.0)
    )



if __name__ == "__main__":
    project = "PARAPHASE_SPLINT"
    exp_name = "Lora_POC"
    seeds = [9599]
    model_name = "t5-base"
    explanation_sep = " because "
    warmup_step = 0
    learning_rate_variable = 3e-5
    max_step = 10
    eval_steps = 5
    fewshot_eval_size = 350
    num_train_epochs = 2
    per_device_train_batch_size = 1
    format_dict = {
        'esnli': ['unifiedqa_snli_mix_what_with_choices_v2'],
        'cos_e': ['unifiedqa_matching'],
        'ecqa': ['unifiedqa_matching'],
        'sensemaking': ['unifiedqa_what'],
        'sbic': ['t5_fewshot_infilling_more_natural']
    }

    n_shots_dict = {
        'esnli': [16],
        'cos_e': [48],
        'ecqa': [48],
        'sensemaking': [24],
        'sbic': [24]
    }

    data_path_dict = {
        'esnli': None,
        'cos_e': None,
        'ecqa': '../data/ECQA-Dataset',
        'sensemaking': '../data/SenseMaking/',
        'sbic': '../data/SBIC/'
    }
    for task in ['esnli', 'cos_e', 'ecqa', 'sensemaking']:
        io_format = format_dict[task][0]
        n_shots = n_shots_dict[task][0]
        data_path = data_path_dict[task]

        for seed in seeds:
            run_name = (
                f"{task}-{seed}-{model_name}-1-{max_step}-{num_train_epochs}-"
                f"{warmup_step}-3e-05-{per_device_train_batch_size}-{eval_steps}-"
                f"{fewshot_eval_size}-{explanation_sep.strip()}-"
                f"{model_name.replace('/', '')}{io_format}-{n_shots}"
            ).replace(" ", "")

            output_dir = os.path.join(exp_name, run_name)
            os.makedirs(output_dir, exist_ok=True)

            os.environ["WANDB_PROJECT"] = project
            os.environ["WANDB_NAME"] = run_name
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

            model_args = ModelArguments(
                model_type=model_name,
                model_class=model_name,
                tokenizer_name=model_name,
                pretrained=True,
                rationale_only=False
            )

            data_args = DataTrainingArguments(
                task_name=task,
                version_name="v1.0",
                n_shots=n_shots,
                fewshot_eval_size=fewshot_eval_size,
                io_format=io_format,
                explanation_sep=explanation_sep,
                data_path=data_path
            )

            training_args = TrainingArguments(
                output_dir=output_dir,
                do_train=True,
                do_eval=True,
                seed=seed,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=8,
                learning_rate=learning_rate_variable,
                warmup_steps=warmup_step,
                max_steps=max_step,
                eval_steps=eval_steps,
                logging_steps=1,
                logging_first_step=True,
                save_total_limit=1,
                save_strategy="steps",
                lr_scheduler_type="constant",
                bf16=True,
                bf16_full_eval=True,
                save_weight=False,
                report_to=["all"]
            )

            print(f"Running: {run_name}")
            train_model(model_args, data_args, training_args)