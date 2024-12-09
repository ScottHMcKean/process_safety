# Databricks notebook source
display(spark.table('shm.process_safety.incidents_w_embeddings'))

# COMMAND ----------

import pyspark.sql.functions as F
incidents_embedded_pd = (
  spark.table('shm.process_safety.incidents_w_embeddings')
  .toPandas()
)

# COMMAND ----------

import pandas as pd
expl_embed = pd.DataFrame(
  incidents_embedded_pd['embedding'].to_list(), 
  columns=[f'emb{x}' for x in range(len(incidents_embedded_pd['embedding'][0]))]
)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's try the naive approach and only use the embeddings and labels
# MAGIC - We don't need to scale because embeddings are already normalized
# MAGIC - Let's test PCA and non PCA
# MAGIC - The Naive model without PCA isn't awful, with an accuracy of 66% before any tuning

# COMMAND ----------

X = expl_embed.fillna(0)
y = incidents_embedded_pd['critical_risk'].fillna('Other')

# COMMAND ----------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

from lightgbm import LGBMClassifier

model = LGBMClassifier(
    objective='multiclass',
    num_class=y.unique().shape,
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42
)

model.fit(X_train, y_train)

# COMMAND ----------

from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC PCA reduces dimensionality from 1024 to 135, not bad and at least features < obs now. But doesn't substantially improve accuracy. Might need to look at some better normalization techniques.

# COMMAND ----------

from sklearn.decomposition import PCA

pca = PCA(n_components=0.90)  # Retain 90% of variance
X_pca = pca.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# COMMAND ----------

from lightgbm import LGBMClassifier

model = LGBMClassifier(
    objective='multiclass',
    num_class=y.unique().shape,
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
