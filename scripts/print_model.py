
import gpt3
import logging
import math
import os
import torch
import torch.nn as nn
from typing import List, Dict, Any, NewType

InputDataClass = NewType("InputDataClass", Any)

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
from metrics import evaluate
import torch
import datasets
import git
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
from transformers import AutoTokenizer, AutoModel

# model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-large")
# model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-large")

####
# tokenizer = AutoTokenizer.from_pretrained("KETI-AIR/ke-t5-base")
# model = AutoModelForSeq2SeqLM.from_pretrained("KETI-AIR/ke-t5-base")


# Summarization
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

# print(model)

## text-generation
# tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
# model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")


for name, param in model.named_parameters():
    if "decoder" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

trainable = [name for name, param in model.named_parameters() if param.requires_grad]
print("Trainable parameters:")
for name in trainable:
    print(name)

# model.lm_head.weight.requires_grad = True

trainable = [name for name, param in model.named_parameters() if param.requires_grad]


trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

# Calculate the percentage
percentage = 100 * trainable_params / total_params

print(f"Trainable parameters: {trainable_params}")
print(f"Total parameters: {total_params}")
print(f"Trainable %: {percentage:.2f}%")
