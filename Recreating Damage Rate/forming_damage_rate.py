import polars as pl
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

#%%
# load in appropriate df - damage_df parquet should be in repo
# otherwise you need a df with EV/LA/Spray. You can learn how to calc spray
# from play-by-play data hit coordinates (listed as coord_x/coord_y normally)
# here: https://gist.github.com/bayesball/8ec179cd3b7b86bd849f047e07362680

# this data already has the spray angle, and spray_angle_adj mirrors it so that
# a negative spray_angle_adj is a pulled batted ball for both righties and lefties

df = pl.read_parquet(r"\damage_df.parquet")

#%% 
# recreating damage rate, which measures batted balls that clear a threshold of EV determined
# by their launch and spray angles. It's like hard hit percentage that factors in the angle the ball
# is hit at.

# EV by spray
ev_by_spray = ( 
    df.with_columns([
        pl.col("spray_angle_adj").round(0).alias("spray_angle_adj")])
    .group_by("spray_angle_adj")
    .agg([pl.sum("is_in_play").alias("bbe"),
          
          # I explored a few different %ile combinations here, the original statistic was always
          # 80th EV at each angle--and I settled on that too--but that's why some of these other
          # %ile lines exist.
          
          # pl.col("exit_velo").quantile(0.75, interpolation = "nearest").alias("EV75"),
          pl.col("exit_velo").quantile(0.8, interpolation = "nearest").alias("EV80"),
          # pl.col("exit_velo").quantile(0.85, interpolation = "nearest").alias("EV85"),
          # pl.col("exit_velo").quantile(0.9, interpolation = "nearest").alias("EV90"),
          pl.mean("woba_value").alias('wobacon')])
    .filter(pl.col("spray_angle_adj").is_not_nan())
    )

# EV by launch
ev_by_launch = ( 
    df.with_columns([
        pl.col("launch_angle").round(0).alias("launch_angle")])
    .group_by("launch_angle")
    .agg([pl.sum("is_in_play").alias("bbe"),
          
          # see above re: 75, 85, 90th %ile lines
          
          # pl.col("exit_velo").quantile(0.75, interpolation = "nearest").alias("EV75"),
          pl.col("exit_velo").quantile(0.8, interpolation = "nearest").alias("EV80"),
          # pl.col("exit_velo").quantile(0.85, interpolation = "nearest").alias("EV85"),
          # pl.col("exit_velo").quantile(0.9, interpolation = "nearest").alias("EV90"),
          pl.mean("woba_value").alias('wobacon')])
    .filter(pl.col("launch_angle").is_not_nan())
    )

#%%
# modeling for the individual components. These will be used together in a future step for the
# real EV threshold.
target = "EV80"

# launch half
ev_by_launch_pd = ev_by_launch.to_pandas()

# creating polynomial feature. Original damage rate was EV80 ~ poly(LA/SA, 2) in R
# this approximates that
ev_by_launch_pd["launch_angle2"] = ev_by_launch_pd["launch_angle"] ** 2

LA_model = smf.wls(f"{target} ~ launch_angle + launch_angle2", data=ev_by_launch_pd,
                   weights = ev_by_launch_pd["bbe"]).fit()

print(LA_model.summary())

# spray half
ev_by_spray_pd = ev_by_spray.to_pandas()
ev_by_spray_pd["spray_angle_adj2"] = ev_by_spray_pd["spray_angle_adj"] ** 2

SA_model = smf.wls(f"{target} ~ spray_angle_adj + spray_angle_adj2", data=ev_by_spray_pd,
                   weights = ev_by_spray_pd["bbe"]).fit()

print(SA_model.summary())

# now we have a model to predict the EV80 at each angle threshold of launch and spray
# we will create a cartesian grid containing combinations of each angle and take the higher
# of the 2 thresholds as our final target EV amount to create the final model
#%%
# this is for the grid
import itertools

# creating a standardized grid of LA+SA ranges
spray = np.linspace(-50, 50, 101)
launch = np.linspace(-50, 50, 101)

# Factor levels
# none here (yet - might explore handedness feature)

# Cartesian product of all combinations
grid = pd.DataFrame(
    list(itertools.product(spray, launch)),
    columns=["spray_angle_adj", "launch_angle"]
)

# creating polynomial terms for predictions
grid["spray_angle_adj2"] = grid["spray_angle_adj"] ** 2
grid["launch_angle2"] = grid["launch_angle"] ** 2

# inputting predictions 
grid['sa_pred'] = SA_model.predict(grid)
grid['la_pred'] = LA_model.predict(grid)

# this designates the higher of the 2 thresholds as *the* damage threshold for that combination
# a batted ball has to clear both EVs to be considered "damage"-worthy
grid['pred_thresh'] = grid[["la_pred", "sa_pred"]].max(axis = 1)

#%%
# now we finally create the actual damage model
from pygam import LinearGAM, te

# adding in a specially named and filled column "***_full" that I'll use for the damage model
# this is a workaround for the NaN/Inf issue that exists when I try to use the GAM
# on the original df
grid["spray_angle_adj_full"] = grid['spray_angle_adj']
grid["launch_angle_full"] = grid["launch_angle"]

# some quick tuning of the lambdas in the GAM
lam = np.logspace(-20, 20, 20)
lams = [lam] * 2

# features
X = grid[["spray_angle_adj_full", "launch_angle_full"]]
# target
y = grid["pred_thresh"]

# actual training. Splines are manually lowered for a smoother value map. This was done through
# trial and error more than any tuning mechanism
# Tensor product is the interaction between spray+launch
gam = LinearGAM(te(0, 1, n_splines = [8, 8]))

gam.fit(X, y)

# apploying our lambda grid
gam.gridsearch(X, y, lam=lams)
# check the model summary
gam.summary()

# inputting results
grid["gam_preds"] = gam.predict(grid[['spray_angle_adj_full', 'launch_angle_full']])

# grid should have pred_thresh and gam_preds, which should be close to ~equal

#%%
# creating non-null vectors to predict on within original df
# replacing NaN/Inf with 0 (these will be filtered out when we only look at batted ball events later)
# the model just gets stuck on them as is
df = df.with_columns([
    pl.col("spray_angle_adj")
        .replace([np.inf, -np.inf], None)
        .fill_null(0).alias('spray_angle_adj_full'),
    pl.col("launch_angle")
        .replace([np.inf, -np.inf], None)
        .fill_null(0).alias('launch_angle_full')
])

# inputting damage thresholds
df = df.with_columns([
    pl.Series("damage_thresh", gam.predict(df.select(["spray_angle_adj_full", "launch_angle_full"]).to_numpy())),
    ])

# optional column: adding a binary 1 or 0 indicating damage. This would be simple:
# pl.when((pl.col('exit_velo') >= pl.col('damage_thresh') & pl.col(launch_angle) > 0)).then(1).otherwise(0)

#%%
# aggregating on league level first to check that numbers make sense, then on player level
lg_df = ( df.filter(pl.col("is_in_play"))
         .group_by('game_year')
         .agg([pl.count().alias('bbe'),
               (pl.col('outcome') == 'hr').sum().alias('home_runs'),
               (pl.col('bb_type').is_in(["fly_ball", "popup", "line_drive"])).sum().alias('FBLD_count'),
               ((pl.col('exit_velo') >= pl.col('damage_thresh')) & (pl.col('launch_angle') > 0)).sum()
               .alias('damage_count')])
         .with_columns([(pl.col('damage_count') / pl.col('bbe')).alias('damage_rate'),
                        (pl.col('home_runs') / pl.col('FBLD_count')).alias('hr_fbld_pct')])
         )
# rates are lower than they are with the og damage but I'm okay with that


#%%
# player level aggregation by season on only bbe
player_df = ( df.filter(pl.col("is_in_play"))
         .group_by('batter_mlbid', 'game_year')
         .agg([pl.first('hitter_name').alias('hitter_name'),
               pl.count().alias('bbe'),
               pl.mean('woba_value').alias('wobacon'),
               pl.col('play_id').n_unique().alias('n_play_ids'),
               (pl.col('outcome') == 'hr').sum().alias('home_runs'),
               (pl.col('bb_type').is_in(["fly_ball", "popup", "line_drive"])).sum().alias('FBLD_count'),
               ((pl.col('exit_velo') >= pl.col('damage_thresh')) & (pl.col('launch_angle') > 0)).sum()
               .alias('damage_count')])
         .with_columns([(pl.col('damage_count') / pl.col('bbe')).alias('damage_rate'),
                        (pl.col('home_runs') / pl.col('FBLD_count')).alias('hr_fbld_pct')])
         .filter((pl.col('bbe') >= 50))
         .sort(pl.col('damage_rate'), descending = True)
         .select(pl.exclude(['FBLD_count', 'damage_count', 'n_play_ids']))
         )

# saving a csv for easy viewing of the results
player_df.write_csv(r'hitters_with_damage_rate.csv')

#%%
# loading in pre-queried full season data containing season end stats to check against
df_hitters = pl.read_parquet(r"comparison_df.parquet")

# joining with our created df that has damage included
hitter_df = player_df.join(df_hitters,
                           on = ['batter_mlbid', 'game_year', 'hitter_name'],
                           how = 'left')

#%%
# converting from polars to pandas for this script
# this line is where you enter the df name as the placeholder
df_placeholder = hitter_df

# dataframe for script converted to pandas
ds = df_placeholder.to_pandas()

# Ensure game_year is numeric
ds["game_year"] = pd.to_numeric(ds["game_year"], errors="coerce")

# much of this script can be plug-and-play for future variables, which is why it's annotated more
# generally
# =============================================
# Build lagged (year_n to year_n+1) join
# =============================================
player_type = "batter"  # "pitcher" or "batter"
unit = "bbe"            # sample size variable 
qty = 100               # minimum threshold to be used for season n and season n+1
                        # e.g. player has to have 100 bbe in consecutive seasons to be included

# Make a copy for year n and n+1
ds_n = ds.copy()
ds_n["game_year_n"] = ds_n["game_year"] - 1

# Join year_n (df_n) onto year_n+1 (df)
multi = (
    ds.merge(
        ds_n,
        left_on=[f"{player_type}_mlbid", "game_year"],
        right_on=[f"{player_type}_mlbid", "game_year_n"],
        how="inner",
        suffixes=("", "_next")
    )
)

# =============================================
# Filter to players with enough sample size in both years
# =============================================
multi = multi[
    (multi[unit] >= qty) &
    (multi[f"{unit}_next"] >= qty)
].copy()

# =============================================
# Compute average sample size weights - intentionally flexible but these are the 3 variables I use 
# most often for this.
# =============================================
multi[f"{unit}_avg"] = (multi[unit] + multi[f"{unit}_next"]) / 2

# (add more if needed)
if "pa" in multi.columns:
    multi["PA_avg"] = (multi["pa"] + multi.get("pa_next", 0)) / 2
    
if "bbe" in multi.columns:
    multi["bbe_avg"] = (multi["bbe"] + multi.get("bbe_next", 0)) / 2
    
if "plate_appearances" in multi.columns:
    multi["PA_avg"] = (multi["plate_appearances"] + multi.get("plate_appearances_next", 0)) / 2

#%%
def weighted_corr(x, y, w):
    """Weighted Pearson correlation."""
    # Remove any rows with NaN
    mask = (~x.isna()) & (~y.isna()) & (~w.isna())
    x, y, w = x[mask], y[mask], w[mask]

    # Weighted means
    mean_x = np.average(x, weights=w)
    mean_y = np.average(y, weights=w)
    cov_xy = np.average((x - mean_x) * (y - mean_y), weights=w)
    var_x = np.average((x - mean_x)**2, weights=w)
    var_y = np.average((y - mean_y)**2, weights=w)
    return cov_xy / np.sqrt(var_x * var_y)

# numeric columns only
num_cols_x = [
    c for c in multi.columns
    if c.endswith(("_next", "_avg")) is False and multi[c].dtype.kind in "fi"
]
num_cols_y = [
    c for c in multi.columns
    if c.endswith("_next") and multi[c].dtype.kind in "fi"
]


num_cols = multi.select_dtypes(include=[np.number]).columns
num_cols_x = [c for c in num_cols if not c.endswith("_next")]
num_cols_y = [c for c in num_cols if c.endswith("_next")]

# correlation matrix
corr_matrix = pd.DataFrame(
    index=num_cols_x,
    columns=[c.replace("_next", "") for c in num_cols_y],
    dtype=float
)

for xcol in num_cols_x:
    for ycol in num_cols_y:
        base_ycol = ycol.replace("_next", "")
        corr_matrix.loc[xcol, base_ycol] = weighted_corr(
            multi[xcol],
            multi[ycol],
            multi[f"{unit}_avg"]
        )

corr_matrix = corr_matrix.round(3)

#%%
# =============================================
# Plot correlation matrix
# =============================================
import seaborn as sns

plt.figure(figsize=(20, 20))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdYlBu_r",
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Weighted Year-to-Year Correlation"}
)
plt.title(f"{player_type.title()} Year-to-Year Stickiness (weighted by {unit}_avg ≥ {qty})")
plt.tight_layout()
plt.show()

#%%
# saving model
import pickle

with open(r"py_damage_model.pkl", "wb") as f:
    pickle.dump(gam, f)

with open(r"py_damage_model.pkl", "rb") as f:
    gam = pickle.load(f)