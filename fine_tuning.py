# Databricks notebook source
# MAGIC %md
# MAGIC ## Fine Tuning LLama3.2 3B using QLoRA
# MAGIC
# MAGIC Databricks hosts Llama models in the marketplace. This notebook provides an example of fine tuning them on a dataset using a V100 GPU and 15.4ML runtime.
# MAGIC
# MAGIC We leverage the PEFT library from Hugging Face, as well as QLoRA for more memory efficient finetuning.
# MAGIC
# MAGIC Run the cells below to setup and install the required libraries. For our experiment we will need `accelerate`, `peft`, `transformers`, `datasets` and TRL to leverage the recent [`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer). We will use `bitsandbytes` to [quantize the base model into 8bit](https://huggingface.co/blog/4bit-transformers-bitsandbytes).

# COMMAND ----------

# MAGIC %pip install -r requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset Generation
# MAGIC Our first task is to take a 3B parameter Llama 3.2 model and fine-tune it to predict activities based on a long and short description, as well as the equipment tag.

# COMMAND ----------

from utils import print_gpu_utilization
print_gpu_utilization()

# COMMAND ----------

import mlflow
from mlflow.artifacts import download_artifacts
mlflow.set_registry_uri("databricks-uc")
uc_model_path = "models:/system.ai.llama_v3_2_3b_instruct/2"
base_model_local_path = "/local_disk0/llama_v3_2_3b_instruct/"
ft_model_local_path = "/local_disk0/llama_v3_2_3b_ft_activity/"
artifact_path = download_artifacts(
  artifact_uri=uc_model_path, 
  dst_path=base_model_local_path
  )

# COMMAND ----------

import os
from transformers import AutoTokenizer

max_length = 4096

tokenizer_path = os.path.join(artifact_path, "components", "tokenizer")
tokenizer = AutoTokenizer.from_pretrained(
  tokenizer_path, 
  padding_side='left',
  model_max_length = max_length,
  add_eos_token=True
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Load a quantized Llama 3B model. We will attempt PEFT with this.

# COMMAND ----------

from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_path = os.path.join(artifact_path, "model")
model = AutoModelForCausalLM.from_pretrained(
  model_path, 
  quantization_config=bnb_config
  )
model.config.use_cache = False
model.config.pad_token_id = model.config.eos_token_id

# COMMAND ----------

print_gpu_utilization()

# COMMAND ----------

import bitsandbytes as bnb

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

linear_layers = find_all_linear_names(model)
print(f"Linear layers in the model: {linear_layers}")

# COMMAND ----------

from peft import LoraConfig, TaskType, get_peft_model

peft_config = LoraConfig(
  task_type="CAUSAL_LM",
  bias="none",
  inference_mode=False, 
  r=32, 
  lora_alpha=32, 
  lora_dropout=0.1,
  target_modules=linear_layers
  )

# COMMAND ----------

peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

# COMMAND ----------


