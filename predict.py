# Databricks notebook source
import mlflow

model = mlflow.sklearn.load_model("models:/dota-unesp/production")

sdf = spark.table("sandbox_apoiadores.abt_dota_pre_match_new")
df = sdf.toPandas()

# COMMAND ----------

target_column = 'radiant_win'
id_column = 'match_id'

features = list(set(df.columns.tolist()) - set([target_column, id_column]))

X = df[features]

# COMMAND ----------

score = model.predict_proba(X)
df["proba_radiant_win"] = score[:, 1]
df[[id_column, target_column, "proba_radiant_win"]]

# COMMAND ----------


