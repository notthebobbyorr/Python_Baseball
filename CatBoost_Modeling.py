# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 11:18:49 2025

@author: orrro
"""
#%%
import pandas as pd
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, r2_score
from catboost import CatBoostClassifier, Pool
import optuna


df = pl.read_parquet(r"C:\Users\orrro\Documents\Python_Data\seasons_df_lwt.parquet")

#%%
import random

# ========================================
# 2. Sampling function (equivalent to slice_sample)
# ========================================
def group_sample(df: pl.DataFrame, group_cols: list, prop: float, seed: int = 18):
    """
    Sample proportionally within each group (like dplyr::slice_sample in R)
    """
    np.random.seed(seed)
    groups = []
    for _, group_df in df.group_by(group_cols):
        n = int(len(group_df) * prop)
        sampled_idx = np.random.choice(len(group_df), n, replace=False)
        groups.append(group_df[sampled_idx])
    return pl.concat(groups)

#%%
# removing position player pitching & Coors Field from set
condition = ((pl.col('pitch_velo') > 60) & (pl.col('home_team') != 'COL'))

working_df = df.filter(condition)

# filtering to swings (smaller denominator and leads to more accurate results)
condition2 = (pl.col('is_swing') == 1)

working_df = df.filter(condition2)

working_df = (
    working_df.filter(~pl.col("pi_pitch_group").is_in(['KN', 'XX']))
    .filter(~((pl.col('balls') == 3) & (pl.col('strikes') == 0)))
    )

# using 20/80 rule, take .2 proportion from sample, preserving the conditions in the data
sampled_df = group_sample(working_df, ["stands", "throws", "game_year", "gameday_zone",
                                       "balls", "strikes", "pi_pitch_group"],
                          prop=0.2, seed=18)


#%%
# Define your target and features
# this is for the big data set - will create a much smaller set to test first
target = "is_whiff"

numerical_features = ["avg_release_z", "avg_release_x", "avg_ext", 
            "pitch_velo", "rpm", "vbreak", "hbreak", "axis", "spin_efficiency",
            "z_angle_release", "x_angle_release", "vaa", "haa",
            "primary_velo", "primary_loc_adj_vaa",  "primary_z_release", "primary_x_release",
            "primary_rpm", "primary_axis"]

categorical_features = ["balls", "strikes", "throws", "stands"]

all_features = numerical_features + categorical_features

# Filter and clean data
pdf = sampled_df.to_pandas().replace([np.inf, -np.inf], np.nan)
pdf = pdf.dropna(subset=all_features + [target])
pdf = pdf[~pdf["event_type"].isin(["null", "pitchout", "hit_by_pitch"])]

# Ensure categoricals are strings (CatBoost prefers this)
pdf[categorical_features] = pdf[categorical_features].astype(str)

X = pdf[all_features]
y = pdf[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)

# Create Pools
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
test_pool = Pool(X_test, y_test, cat_features=categorical_features)

#%%
# this is the small test set for the tuning logic
# UPDATE::logic checks out so we can skip this cell

# pdf_sample = pdf.sample(n=5000, random_state=42)
# X_sample = pdf_sample[all_features]
# y_sample = pdf_sample[target]

# X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=18)

# train_pool = Pool(X_train, y_train, cat_features=categorical_features)
# test_pool = Pool(X_test, y_test, cat_features=categorical_features)

#%%
from optuna.integration import CatBoostPruningCallback

# setting up hyperparameter tuning with Optuna

def objective(trial):
    params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": 500,  
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 64, 255),
        "early_stopping_rounds": 50,
        "random_seed": 42,
        "verbose": True
    }

    # Create a model with early stopping
    model = CatBoostClassifier(**params)

    pruning_callback = CatBoostPruningCallback(trial, "Logloss")
   
    model.fit(
        train_pool,
        eval_set=test_pool,
        verbose=1,
        early_stopping_rounds=50,
        callbacks=[pruning_callback],
    )

    # evoke pruning manually.
    pruning_callback.check_pruned()

    preds = model.predict_proba(X_test)[:, 1]
    return log_loss(y_test, preds)

#%%
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, n_jobs=1)

print("Best Logloss:", study.best_value)
print("Best Params:", study.best_params)

#%%
best_params = study.best_params
best_params.update({
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "iterations": 1000,
    "early_stopping_rounds": 25,
    "random_seed": 42,
    "verbose": 100,
})

final_model = CatBoostClassifier(**best_params)
final_model.fit(train_pool, eval_set=test_pool, use_best_model=True)

#%%
y_pred = final_model.predict_proba(X_test)[:, 1]
print("Log loss:", log_loss(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
print("Correlation:", np.corrcoef(y_test, y_pred)[0, 1])

#%%
final_model.save_model(f"{target}_catboost_model.cbm")

#%%
import matplotlib.pyplot as plt
import seaborn as sns

importances = final_model.get_feature_importance(prettified=True)
plt.figure(figsize=(8, 4))
sns.barplot(data=importances, x="Importances", y="Feature Id")
plt.title("CatBoost Feature Importance")
plt.tight_layout()
plt.show()

#%%
from catboost import CatBoostClassifier

whiff_model = CatBoostClassifier()
whiff_model.load_model(f"is_whiff_catboost_model.cbm")

#%%
# Features used for prediction
numerical_features = ["avg_release_z", "avg_release_x", "avg_ext", 
            "pitch_velo", "rpm", "vbreak", "hbreak", "axis", "spin_efficiency",
            "z_angle_release", "x_angle_release", "vaa", "haa",
            "primary_velo", "primary_loc_adj_vaa",  "primary_z_release", "primary_x_release",
            "primary_rpm", "primary_axis"]

categorical_features = ["balls", "strikes", "throws", "stands"]
all_features = numerical_features + categorical_features

# Convert Polars to pandas for prediction
df_for_preds = df.select(all_features).to_pandas()

# Fix dtypes for categorical features
df_for_preds[categorical_features] = df_for_preds[categorical_features].astype(str)

#%%
# Predict
preds = whiff_model.predict_proba(df_for_preds)[:, 1]

#%%
import polars as pl

# Add predictions to original df
df = df.with_columns([
    pl.Series(name="pred_whiff", values=preds)
])

#%%
#  testing against actual results
df_whiff = (
    df.filter(pl.col('is_swing') == 1)
    .group_by(['pitcher_mlbid', 'game_year', 'pitch_tag'])
    .agg([pl.first('pitcher_name').alias('pitcher_name'),
          pl.mean('is_whiff').alias('whiff_pct'),
          pl.mean('pred_whiff').alias('pred_whiff_pct'),
          pl.count().alias('swings')
    ])
    .filter(pl.col('swings') > 20)
    .sort(by = ['game_year', 'pred_whiff_pct'], descending = [True, True])
    )
