# F1 Top 10 Finish Prediction (2014–2024)

## 1. Project Goal
The goal of this project is to predict which drivers will finish in the Top 10 (i.e. score points) in each Formula 1 Grand Prix.

We train on historical data from the 2014–2023 seasons, and we generate predictions for every race in the 2024 season.

The task is framed as binary classification:
- `y_top10 = 1` if the driver finished in the Top 10 in that race,
- `y_top10 = 0` otherwise.

## 2. Data
The project uses public Formula 1 timing and results data (race results, qualifying, pit stops, lap times, etc.).  
We engineered features from multiple raw tables commonly available in F1 datasets (e.g. Kaggle F1 World Championship data).

Key input tables (not all included in this repo due to size / licensing):
- `results.csv`: final race classifications (position, points, DNF).
- `races.csv`: race metadata (raceId, year, round, Grand Prix name).
- `drivers.csv`: driver information (forename, surname) used for reporting.
- `constructors.csv`: team information.
- `qualifying.csv`: qualifying session results (Q1, Q2, Q3 lap times, grid position).
- `pit_stops.csv`: pit stop durations per driver.
- `lap_times.csv`: per-lap timing data for each driver.
- `sprint_results.csv`: sprint race performance (if the weekend had a sprint).
- `driver_standings.csv`, `constructor_standings.csv`: rolling season standings.

For convenience, we provide smaller prepared CSVs such as:
- `f1_enhanced_feature_preview.csv`: example of the final modeling table (joined features).
- `leaderboard_pre_race.csv`, `leaderboard_pre_week.csv`: per-race Top 10 predictions with probabilities.
- `model_compare_sorted.csv`: comparison across different ML models.
- `compare_prerace_vs_preweek.csv`: scenario comparison (see below).

We do **not** upload the full raw timing datasets here to keep the repo light.

## 3. Feature Engineering
We built features to reflect real racing context and driver/team form. Major groups:

**Recent form (rolling windows)**  
- Driver and team stats aggregated over the last 3 / 5 / 10 races:
  - average points,
  - typical finishing position (median),
  - pit stop consistency,
  - lap time consistency / best lap pace.
  
  Example features:
  - `drv_points_mean_3`, `drv_points_mean_5`, `drv_points_mean_10`
  - `drv_position_median_3`, `drv_position_median_5`, ...
  - `team_points_mean_5`, `team_points_mean_10`

**Last-race snapshot**
- The driver's previous race result and previous standing:
  - `position_prev`, `points_prev`
  - `team_position_prev`, `team_points_prev`

**Qualifying / Grid**
- Information from qualifying and sprint:
  - `grid` (starting P number),
  - `quali_position`,
  - `q1_ms`, `q2_ms`, `q3_ms` (session times in milliseconds),
  - sprint-related features (`sprint_grid`, `sprint_pts`, etc.).

These are extremely powerful because F1 is heavily track-position dependent.

**Pit stop performance**
- Rolling averages and variability of pit stop times:
  - `drv_pit_ms_mean_3`, `drv_pit_ms_std_3`, etc.

**Lap pace**
- Rolling pace-related signals:
  - `drv_bestlap_ms_mean_3` (best lap pace over recent races),
  - `drv_lap_med_ms_mean_5` (median race pace),
  - `drv_lap_var_mean_5` (lap time stability / consistency).

We create **two feature sets** for two real-world scenarios:

1. **Pre-race model (full info)**  
   Uses qualifying / sprint information (grid, quali position, etc.).  
   This is like making predictions after qualifying, before the race.

2. **Pre-weekend model (restricted info)**  
   Removes qualifying and sprint features.  
   This simulates predicting *before* the weekend starts (i.e. "Who are the likely Top 10 this GP?").

This allows us to evaluate how early in the race weekend we can make useful predictions.

## 4. Modeling and Evaluation
### 4.1 Train / Test split
- Training data: all driver–race entries from 2014–2023.
- Holdout test data: all driver–race entries from the 2024 season.

### 4.2 Cross-validation
We use `GroupKFold` where the group is `raceId`.  
That means data from the same Grand Prix (same raceId) never appears in both train and validation within the same fold.  
This prevents leakage across drivers in the same race.

### 4.3 Metrics
We evaluate using:
- **Macro-F1**: class-balanced F1 (treats Top10 vs non-Top10 fairly even if imbalanced).
- **Average Precision (AP)**: area under the precision–recall curve.

We also tune the decision threshold (not always 0.5) to maximize Macro-F1 on out-of-fold predictions.

### 4.4 Probability calibration
We optionally apply isotonic regression on out-of-fold predictions to calibrate the predicted probabilities.  
This makes the predicted score closer to “actual chance of finishing in the Top 10”, which is important for ranking and confidence estimation.

### 4.5 Per-race Top 10 enforcement
After we get probabilities for each driver in a given race, we force-pick the Top 10 drivers by probability for that race.  
This simulates the physical reality of F1: only 10 drivers score points.

We then measure how many of those 10 predictions actually finished in the Top 10.  
Result: on average we correctly identified about **8.4 out of 10** drivers per race (min ~7, median ~8.5, max 10).

We also export per-race leaderboards with:
- race metadata (Grand Prix name, round, year),
- driver name and team,
- predicted probability,
- whether that driver was truly in the Top 10.

See `leaderboard_pre_race.csv` and `leaderboard_pre_week.csv`.

## 5. Models Compared
We benchmarked several models:
- LightGBM
- XGBoost
- Random Forest
- SVM (RBF kernel)
- Logistic Regression (for interpretability)
- MLP (simple neural net baseline)

Results summary:
- **SVM (RBF), LightGBM, XGBoost, RandomForest** all perform competitively.
- A simple MLP underperforms here, likely due to limited sample size and no heavy tuning.

## 6. Key Results (2024 holdout)
### Pre-race (includes qualifying & sprint info)
- Macro-F1 ≈ 0.84  
- AP (Average Precision) ≈ 0.91  
- Per-race Top 10 picking: ~8–9 correct drivers on average out of 10

### Pre-weekend (no qualifying info)
- Macro-F1 ≈ 0.78  
- AP ≈ 0.84  
- Per-race Top 10 picking: ~8.4 / 10 average hit rate

Interpretation:
- Even **before** qualifying, the model can already guess most of the points finishers.
- After qualifying, accuracy jumps further.

This is useful for preview content, betting-style projection, broadcast graphics, or performance scouting.

## 7. Model Explainability
We include:
- LightGBM feature importances
- SHAP summary plots (global impact of each feature)
- SHAP waterfall plots for individual race predictions
- Precision–Recall curves and confusion matrices for selected models

High-impact signals include:
- starting grid / quali position,
- recent performance form (rolling points / median finish),
- lap pace consistency,
- pit stop stability,
- team form.

## 8. Next Steps / Limitations
- Add weather, circuit characteristics, and tire strategy features.
- Perform season-wise or walk-forward validation (leave-one-season-out) to measure robustness to long-term regulation changes.
- Improve probability calibration and uncertainty estimation.
- Consider model stacking / ensembling for final race-day forecasts.

## 9. Author's Note on AI Utilization
This project was completed by actively leveraging **Generative AI as a 'Coding Assistant' and 'Pair Programmer'**.

By accelerating routine code implementation (e.g., data loading, plotting, model boilerplate) using AI, I was able to focus my time and effort on higher-level analytical tasks that truly drive project value.

My core contribution and focus were on:
- **Problem Framing:** Defining the "Pre-race" vs. "Pre-weekend" scenarios.
- **Validation Design:** Architecting the leak-proof `GroupKFold` strategy and the 2024 holdout test.
- **Feature Engineering:** Hypothesizing and creating domain-specific features (like pit stop *stability* vs. speed).
- **Model Interpretation:** Analyzing SHAP values and translating complex metrics (like F1) into clear business insights (like "8.4 out of 10").

**In summary:**  
This project demonstrates an end-to-end pipeline to predict F1 Top 10 finishers using historical performance, qualifying data, pit stops, sprint results, and rolling form. The model generalizes well to a future season (2024) and produces realistic per-race Top 10 leaderboards with confidence scores.

Note: The main analysis notebook (`F1_Prediction_for_TOP10_2024.ipynb`) includes commentary in Korean.  
For English context, please refer to this README (project goal, data, features, validation, and results).

### Data Source

The raw timing and classification data (race results, lap times, pit stops, qualifying, etc.) comes from publicly available Formula 1 historical datasets (e.g. Kaggle "Formula 1 World Championship" data).

This repository includes a local copy of the CSV files only for reproducibility.  
All original credits for the underlying race timing and results data belong to their respective providers / recorders.
