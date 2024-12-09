# Databricks notebook source
# MAGIC %md
# MAGIC Huggingface uses the datasets library to generate a prompt. We can generate a simple text column for supervised fine tuning using this library and then leverage the huggingface library in downstream tasks. We will save the dataset into volumes.

# COMMAND ----------

import pyspark.sql.functions as F
import datasets

# COMMAND ----------

from pyspark.sql import functions as F

incidents = spark.table("shm.process_safety.incident_descriptions") \
    .filter(F.col('Critical Risk').isNotNull()) \
    .filter(F.col('Description').isNotNull()) \
    .withColumn("Is Test", (F.rand() < 0.2))

display(incidents)

# COMMAND ----------

# Setup the configuration
import mlflow
from mlflow.models import ModelConfig
config = ModelConfig(development_config="./config.yml")

# COMMAND ----------

dataset = datasets.Dataset.from_spark(incidents, cache_dir="/local_disk0/")

# COMMAND ----------



# COMMAND ----------

prompt = config.get("prompt_template")

def combine_columns(example):
    prompt.format(
        valid_categories=config.get("valid_categories"),
        industry_sector=example['Industry Sector'],
        description=example['Description']
        )
    return {
      "input": prompt, 
      "label": f"{example['Critical Risk']}",
      "text": prompt + example['Critical Risk']
      }

dataset = dataset.map(combine_columns)

# COMMAND ----------

dataset[0]

# COMMAND ----------

# now let's consistently split our dataset into train and test

train_dataset = dataset.filter(lambda x: x['Is Test'] == False)
test_dataset = dataset.filter(lambda x: x['Is Test'] == True)
split_dataset = split_dataset = datasets.DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

split_dataset['train'][0]

split_dataset.save_to_disk("/Volumes/shm/process_safety/datasets/risk_dataset")
