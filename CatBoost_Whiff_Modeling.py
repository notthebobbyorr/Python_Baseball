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
import random

# queried data containing play-by-play data from 2021-2025
df = pl.read_parquet(r"seasons_df.parquet")

#%%
# ========================================
# 2. Sampling function (for grabbing a proportion of grouping variables - in this case, I want to preserve the ratio of pitches in each part of the zone (1-14) in each possible platoon
# matchup (L-L, L-R, R-L, R-R) and each pitch group (FA, SL, CU, CH)
# ========================================
def group_sample(df: pl.DataFrame, group_cols: list, prop: float, seed: int = 18):
    """
    Sample proportionally within each group 
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

# removing any additional oddities, like knuckleballs, unknown pitch types, and events in 3-0 counts (not a requirement for a df containing only swings, but
# something I normally do regardless
working_df = (
    working_df.filter(~pl.col("pi_pitch_group").is_in(['KN', 'XX']))
    .filter(~((pl.col('balls') == 3) & (pl.col('strikes') == 0)))
    )

# using 20/80 rule, take .2 proportion from sample, preserving the conditions in the data
# taking pitches from each season in each count in addition to previously mentioned grouping variables
sampled_df = group_sample(working_df, ["stands", "throws", "game_year", "gameday_zone",
                                       "balls", "strikes", "pi_pitch_group"],
                          prop=0.2, seed=18)


#%%
# Define target and features
target = "is_whiff" # binary variable present in the data

# these feature some pre-determined variables I created. Season long release traits and the season long averages for that pitcher's primary pitch
# loc_adj_vaa is a pitch height + pitch velo scaled version of VAA. Contains that pitcher's fastball's "true life" that hitters must prepare for
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
pdf = pdf[~pdf["event_type"].isin(["null", "pitchout", "hit_by_pitch"])] # removing a few more oddities that could add noise

# Ensure categoricals are strings
pdf[categorical_features] = pdf[categorical_features].astype(str)

X = pdf[all_features]
y = pdf[target]

# splitting data into test/train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)

# Create Pools
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
test_pool = Pool(X_test, y_test, cat_features=categorical_features)

#%%
from optuna.integration import CatBoostPruningCallback # will be used to speed up tuning

# setting up hyperparameter tuning with Optuna
# this objective will be used by optuna to evaluate params
def objective(trial):
    params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": 500,  # going lower speeds up the training
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
# this step is where we find the best hyperparams for this objective. could take a while! adjust n_trials for a shorter tuning time as needed
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, n_jobs=1)

print("Best Logloss:", study.best_value)
print("Best Params:", study.best_params)

#%%
# now that we have our best params for the final model, we can set them
best_params = study.best_params
best_params.update({
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "iterations": 1000,
    "early_stopping_rounds": 25,
    "random_seed": 42,
    "verbose": 100,
})

# and train our finished model
final_model = CatBoostClassifier(**best_params)
final_model.fit(train_pool, eval_set=test_pool, use_best_model=True)

#%%
# this is for evaluating model fit on out-of-sample test set
y_pred = final_model.predict_proba(X_test)[:, 1]
print("Log loss:", log_loss(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
print("Correlation:", np.corrcoef(y_test, y_pred)[0, 1])

#%%
# save the trained model
final_model.save_model(f"{target}_catboost_model.cbm")

#%%
# this is exploring the inside of the model
import matplotlib.pyplot as plt
import seaborn as sns

# we can see which features have the biggest impact on model results
importances = final_model.get_feature_importance(prettified=True)
plt.figure(figsize=(8, 4))
sns.barplot(data=importances, x="Importances", y="Feature Id")
plt.title("CatBoost Feature Importance")
plt.tight_layout()
plt.show()

#%%
# and this is for applying the trained model to the original large dataset
from catboost import CatBoostClassifier

# create and load 
whiff_model = CatBoostClassifier()
whiff_model.load_model(f"is_whiff_catboost_model.cbm")

#%%
# Features used for prediction
# same features as before (don't actually need to re-define these in the same script, this is just for illustration)
numerical_features = ["avg_release_z", "avg_release_x", "avg_ext", 
            "pitch_velo", "rpm", "vbreak", "hbreak", "axis", "spin_efficiency",
            "z_angle_release", "x_angle_release", "vaa", "haa",
            "primary_velo", "primary_loc_adj_vaa",  "primary_z_release", "primary_x_release",
            "primary_rpm", "primary_axis"]

categorical_features = ["balls", "strikes", "throws", "stands"]
all_features = numerical_features + categorical_features

# Convert Polars to pandas for prediction
df_for_preds = df.select(all_features).to_pandas()

# set categorical features to strings
df_for_preds[categorical_features] = df_for_preds[categorical_features].astype(str)

#%%
# Predict
preds = whiff_model.predict_proba(df_for_preds)[:, 1]

#%%
# Add predictions to original df
df = df.with_columns([
    pl.Series(name="pred_whiff", values=preds)
])

#%%
#  test model output against actual results
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

# there's the model!
