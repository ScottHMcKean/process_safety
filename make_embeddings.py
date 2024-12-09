# Databricks notebook source
# MAGIC %md
# MAGIC This notebook provides an example of using traditional classification on embeddings to classify safety incidents. It takes a dataset from kaggle (https://www.kaggle.com/datasets/ihmstefanini/industrial-safety-and-health-analytics-database/data).

# COMMAND ----------

# MAGIC %md
# MAGIC First we need to get the embeddings. We will use the BGE-Large Endpoint from Databricks

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow-skinny langchain==0.2.16 langgraph-checkpoint==1.0.12 langchain_core langchain-community==0.2.16 langgraph==0.2.16 pydantic faiss-cpu langchain-openai databricks-langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Setup the configuration
import mlflow
from mlflow.models import ModelConfig
mlflow.langchain.autolog()
config = ModelConfig(development_config="./config.yml")

# COMMAND ----------

from databricks_langchain import DatabricksEmbeddings
embeddings = DatabricksEmbeddings(endpoint=config.get("embedding_endpoint"))
embeddings.embed_query("What is LangChain?")[0:5]

# COMMAND ----------

incidents = spark.table("shm.process_safety.incident_descriptions").filter(F.col("description").isNotNull())
display(incidents)

# COMMAND ----------

# MAGIC %md
# MAGIC We are going to use a spark UDF to get the embeddings for every description 

# COMMAND ----------

import pyspark.sql.functions as F
import pyspark.sql.types as T

@F.udf(T.ArrayType(T.FloatType()))
def compute_embeddings_udf(description):
  from databricks_langchain import DatabricksEmbeddings
  embeddings = DatabricksEmbeddings(endpoint=config.get("embedding_endpoint"))
  return embeddings.embed_query(description)

# COMMAND ----------

# MAGIC %md
# MAGIC I ran into an issue where one of our serving endpoints was limited to 2 QPM per user and it took over 4 hours. I reran with a non-limited endpoint and it took 43 seconds for ~400 queries. So watch out for the query limits!

# COMMAND ----------

incidents_w_embeddings = incidents.withColumn(
  'embedding',
  compute_embeddings_udf(F.col('description'))
  )

# COMMAND ----------

display(incidents_w_embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC Worth noting that this will rerun every time due to the UDF, so be careful. Let's benchmark the entire table.

# COMMAND ----------

# MAGIC %md
# MAGIC Let's use PySpark's explode function to transform the array column and make it useable as features, as well as convert it into Pandas to utilize sci-kit learn and keep things simple.

# COMMAND ----------

#write out table for later
for col in incidents_w_embeddings.columns:
  incidents_w_embeddings = incidents_w_embeddings.withColumnRenamed(col, col.lower().replace(" ","_"))

incidents_w_embeddings.write.format("delta").mode("overwrite").saveAsTable("shm.process_safety.incidents_w_embeddings")

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's jump into scitkit learn and do some standard classification.
