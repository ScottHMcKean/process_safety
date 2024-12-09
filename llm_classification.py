# Databricks notebook source
# MAGIC %md
# MAGIC This notebook provides an example of using large language models (LLMs) classify safety incidents. It takes a dataset from kaggle (https://www.kaggle.com/datasets/ihmstefanini/industrial-safety-and-health-analytics-database/data).

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow-skinny langchain==0.2.16 langgraph-checkpoint==1.0.12 langchain_core langchain-community==0.2.16 langgraph==0.2.16 pydantic faiss-cpu langchain-openai databricks-langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

incidents = spark.table("shm.process_safety.incident_descriptions")
display(incidents)

# COMMAND ----------

# MAGIC %md
# MAGIC We can immediately see why classifying critical risks is fraught with problems - even the distinct categories become difficult due to different spellings and added information. We use DBSQL to query the distinct values.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ARRAY_JOIN(ARRAY_DISTINCT(collect_list(`Critical Risk`)), ', ') AS `Critical Risks`
# MAGIC FROM shm.process_safety.incident_descriptions
# MAGIC WHERE `Critical Risk` IS NOT NULL
# MAGIC AND len(`Critical Risk`)>3

# COMMAND ----------

# Setup the configuration
import mlflow
from mlflow.models import ModelConfig
mlflow.langchain.autolog()
config = ModelConfig(development_config="./config.yml")

# COMMAND ----------

# MAGIC %md
# MAGIC The next section covers the bare minimum to get a custom prompt working at scale in Databricks.

# COMMAND ----------

# Setup the Foundation Model
from langchain_community.chat_models import ChatDatabricks
llm = ChatDatabricks(endpoint=config.get("foundation_endpoint"))
llm.invoke("Write a haiku about process safety")

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's put a prompt and foundation model together

# COMMAND ----------

import mlflow
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

valid_categories = "Pressed, Pressurized Systems, Manual Tools, Other, Chemical substances, Liquid Metal, Electrical, Confined space, Pressurized Systems, Stored Energy, Suspended Loads, Poll, Fall, Bees, Traffic, Projection of Fragments, Venomous Animals, Plates, Vehicles and Mobile Equipment, Choco, Power, Burn, Individual Protection Equipment, Electrical Shock."

# Define a prompt template
prompt = PromptTemplate(
    template=config.get("prompt_template"),
    input_variables=config.get("input_variables")
)

# Define the basic chain
chain = (
    prompt
    | llm
    | StrOutputParser()
)

# COMMAND ----------

#Let's try our zero-shot prompt with the first row of our table
chain.invoke({
    "valid_categories": valid_categories,
    "industry_sector": incidents.select("Industry Sector").first()["Industry Sector"],
    "description": incidents.select("Description").first()["Description"]
})

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's scale our example across the entire dataframe using a spark user defined function (UDF). User defined functions are run on a cluster and need to be fully self contained, so we define most of the above code again.

# COMMAND ----------

import pyspark.sql.functions as F
import pyspark.sql.types as T

prompt_template_str = config.get("prompt_template")
input_variables = config.get("input_variables")

@F.udf(T.StringType())
def get_incident_class_llm(industry_sector, description):
    from langchain_community.chat_models import ChatDatabricks
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    llm = ChatDatabricks(endpoint=config.get("foundation_endpoint"))
    
    prompt = PromptTemplate(
        template=config.get("prompt_template"),
        input_variables=config.get("input_variables")
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "valid_categories": valid_categories,
        "industry_sector": industry_sector,
        "description": description
    })

# COMMAND ----------

# MAGIC %md
# MAGIC We can benchmark the parallelism using 100 rows. The cluster used here has up to 8 workers with 16 cores, so should in theory be able to run all these processes at once and overwhelm our PPT endpoint. But the parallelism seems to work quite nicely, with a total time of 2.6 minutes for 100 70B queries.

# COMMAND ----------

incidents_w_classification = incidents.limit(100).withColumn(
  'llm_classification', 
  get_incident_class_llm(F.col("Industry Sector"), F.col("Description"))
  )

# COMMAND ----------

display(incidents_w_classification)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's store that dataframe for future investigation and evaluation.

# COMMAND ----------

for col in incidents_w_classification.columns:
  incidents_w_classification = incidents_w_classification.withColumnRenamed(col, col.lower().replace(" ","_"))

incidents_w_classification.write.mode('overwrite').saveAsTable(
  'shm.process_safety.incidents_w_llm_classifications'
  )
