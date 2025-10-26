# F1 Top 10 Finish Prediction (2014–2024)

## 1. Project Goal
The goal of this project is to predict which drivers will finish in the Top 10 (i.e. score points) in a Formula 1 Grand Prix.

We train on historical race data from the 2014–2023 seasons and generate predictions for every race in the 2024 season.

We frame the task as binary classification at the **driver–race** level:
- `y_top10 = 1` if the driver finished in the Top 10 (scored points) in that race,
- `y_top10 = 0` otherwise.

In plain terms:  
> “Before a race starts, which 10 drivers are most likely to score points?”

---

## 2. Data
The project uses public Formula 1 timing and results data (race results, qualifying, pit stops, lap times, sprint sessions, etc.).  
We build features by joining and aggregating multiple raw tables commonly found in historical F1 datasets (e.g. Kaggle ‘Formula 1 World Championship’ data).

Key source tables (not all shipped in this repo due to size/licensing):
- `results.csv`: final race classification (finish position, points, DNF status).
- `races.csv`: race metadata (`raceId`, year, round, Grand Prix name).
- `drivers.csv`: driver identity (forename, surname) for human-readable output.
- `constructors.csv`: constructor / team identity.
- `qualifying.csv`: qualifying and Q1/Q2/Q3 lap times (used to infer grid position).
- `pit_stops.csv`: pit stop durations per driver.
- `lap_times.csv`: per-lap timing for each driver in each race.
- `sprint_results.csv`: sprint session results (for sprint weekends).
- `driver_standings.csv`, `constructor_standings.csv`: rolling season standings.

To make the repo lightweight and reproducible, we include only derived / summarized artifacts:

- `f1_enhanced_feature_preview.csv`  
  A sample of the final feature table used for modeling.  
  Each row = (race, driver). Columns include rolling form, qualifying info, pit metrics, etc.

- `leaderboard_pre_week.csv`  
  Predicted Top 10 (with probabilities) **before** qualifying/sprint — early-weekend scenario.

- `leaderboard_pre_race.csv`  
  Predicted Top 10 (with probabilities) **after** qualifying/sprint — right before race start.

- `model_compare_sorted.csv`  
  Side-by-side performance metrics for multiple ML models (SVM RBF, LightGBM, XGBoost, RandomForest, MLP, etc.).

We intentionally do **not** upload the full raw timing datasets (lap-by-lap timing, all pit stops, etc.) to keep the repo small and respect original data licensing.  
These raw tables can be obtained from public F1 historical datasets (e.g. Kaggle).

---

## 3. Feature Engineering
Each (race, driver) row is enriched with features that reflect both long-term form and immediate weekend conditions.

### 3.1 Rolling form (3 / 5 / 10 most recent races)
For each driver and each team, we compute rolling statistics over their last N races:
- average points scored,
- typical finishing position (median, to capture consistency rather than one lucky podium),
- pit stop stability (mean / std of pit stop times),
- race pace stability (lap-time variability, best-lap pace).

Examples of engineered columns:
- `drv_points_mean_3`, `drv_points_mean_5`, `drv_points_mean_10`
- `drv_position_median_3`, `drv_position_median_5`, `drv_position_median_10`
- `team_points_mean_5`, `team_points_mean_10`
- `drv_pit_ms_mean_3`, `drv_pit_ms_std_3`, …
- `drv_bestlap_ms_mean_5`, `drv_lap_var_mean_5`, …

These rolling features are **shifted by one race** so we only use past information and avoid data leakage.

### 3.2 Last-race snapshot
We include “what just happened last race?” signals:
- `position_prev`, `points_prev` (driver’s previous finish / points),
- `team_position_prev`, `team_points_prev` (team’s recent form).

This captures momentum: did the driver/team just perform strongly?

### 3.3 Qualifying / grid context
We inject immediate weekend performance:
- `grid` (actual starting slot for Sunday),
- `quali_position`,
- `q1_ms`, `q2_ms`, `q3_ms` (qualifying session lap times in ms),
- sprint-related signals such as `sprint_grid`, `sprint_pts` for sprint weekends.

In Formula 1, starting position and quali pace are extremely predictive, so these features drive a big jump in accuracy.

### 3.4 Race craft and execution signals
- Pit stop metrics:  
  `drv_pit_ms_mean_3`, `drv_pit_ms_std_5`, etc.  
  → Is the team executing fast, consistent stops?

- Pace metrics from lap times:  
  `drv_bestlap_ms_mean_3` (how fast the driver can be at peak),  
  `drv_lap_med_ms_mean_5` (typical race pace, median lap),  
  `drv_lap_var_mean_5` (consistency / tire management / traffic handling).

Together, these features try to answer:  
> “Is this car/driver combination fast, consistent, and operationally clean?”

---

## 4. Two Prediction Scenarios
We build **two separate views of the world**, because in real life you don’t always have the same information available.

### 4.1 Pre-weekend model (“pre-week”)
- Timing: before qualifying / before sprint. Think Friday.
- We **do not** use grid, Q1/Q2/Q3 times, sprint results.
- We only use longer-term form and stability: rolling points, pit consistency, lap pace consistency, etc.
- Output file: `leaderboard_pre_week.csv`

Use case:  
media preview, early betting lines, internal performance scouting before track action really starts.

### 4.2 Pre-race model (“pre-race”)
- Timing: after qualifying (and sprint, if any), just before Sunday’s main race.
- We **do** use grid position, quali laps, sprint performance.
- Output file: `leaderboard_pre_race.csv`

Use case:  
race broadcast graphics (“Who’s likely to score points today?”), final race preview, last-minute odds.

**In short:**
- Pre-weekend = “Who *should* score points this GP, based on form alone?”
- Pre-race   = “Who *will* score points today, given their actual starting position?”

---

## 5. Modeling and Evaluation

### 5.1 Train / test split
- Training set: all (driver, race) samples from 2014–2023.
- Holdout test: all (driver, race) samples from 2024.  
  2024 is treated as a future, unseen season.

### 5.2 Cross-validation
We use `GroupKFold` with `raceId` as the group.  
That means data from the same Grand Prix never appears in both train and validation in the same fold.

Why this matters:  
Without grouping, one driver from a race could be in training and another driver *from the same race* could be in validation, which leaks race-level context. Grouping prevents that.

### 5.3 Metrics
We track:
- **Macro-F1**: F1-score averaged across both classes (Top10 vs non-Top10) to handle class imbalance fairly.
- **Average Precision (AP)**: area under the precision–recall curve, a ranking-oriented metric.

We also tune the classification threshold (the cut-off on predicted probability) to directly maximize Macro-F1 on out-of-fold predictions.  
> 0.5 is not automatically optimal in imbalanced problems.

### 5.4 Probability calibration
We calibrate predicted probabilities with isotonic regression on out-of-fold predictions.  
This makes “0.8 probability” behave more like “80% chance this driver scores points,” which is critical if you want to turn the model into a decision tool (rank top 10, assign confidence).

### 5.5 Per-race Top 10 enforcement
In the real world, exactly 10 drivers score points.  
So for each race we:
1. Sort all drivers by predicted Top10 probability.
2. Take the top 10 as the model’s “points finishers.”

We then compare that 10-driver list to what actually happened.

**Result (2024 holdout):**  
On average, the model correctly identified about **8.4 out of 10** point scorers per race.  
- min ≈ 7  
- median ≈ 8.5  
- max = 10 (perfect match)

We export these per-race leaderboards (with driver name, team, predicted probability, and whether they really finished in the Top 10) in:
- `leaderboard_pre_week.csv`
- `leaderboard_pre_race.csv`

These CSVs are essentially “race preview cards.”

---

## 6. Model Zoo and Comparison
We benchmark multiple algorithms:

- **SVM (RBF kernel)**  
- **LightGBM**
- **XGBoost**
- **Random Forest**
- **Logistic Regression** (baseline + interpretability)
- **MLP** (simple neural net baseline)

Key findings on the 2024 holdout season:
- The best traditional ML models (SVM-RBF, boosted trees like LightGBM / XGBoost, and Random Forest) all reach strong Macro-F1 (~0.83–0.84 in the pre-race scenario) and high Average Precision (>0.88 in many cases).
- The simple MLP underperforms. Likely causes:
  - dataset is not huge,
  - deep nets often need more tuning / regularization / feature scaling care,
  - tree models and SVM adapt well to tabular, engineered features.

`model_compare_sorted.csv` contains the numeric comparison:
- Out-of-fold (cross-val) scores,
- Holdout 2024 scores,
- Chosen probability threshold per model.

---

## 7. Explainability
We include:
- **Feature importances** (e.g. LightGBM split gain).
- **SHAP summary plots**:
  - Which features push the prediction up/down across the dataset.
  - Shows that grid position, recent rolling form, pit-stop stability, and lap pace consistency are high-impact.
- **SHAP waterfall plots** for individual predictions:
  - “Why did the model think THIS specific driver would score points in THIS specific race?”
- **Precision–Recall curves + Confusion Matrices**:
  - Visualize trade-offs between precision (how trustworthy a Top10 call is) and recall (how many actual Top10 finishers we caught).

This makes the model auditable and useful for non-technical stakeholders:
> “Why are you so confident about Driver X?”

---

## 8. Limitations and Next Steps
- **Track / circuit characteristics** are not fully modeled yet  
  (e.g. street circuits vs high-downforce tracks, altitude, tire degradation profile).
- **Weather and strategy** (safety cars, tire choice) are not included.
- **Regulation shifts** across seasons can change competitive order.  
  We plan “season-wise / walk-forward” validation (e.g. train up to 2022, test on 2023) to measure robustness.
- **Ensembling / stacking** across top models (SVM + boosted trees) could further improve stability.
- **Uncertainty estimation** can be improved to express “confidence bands,” not just point probabilities.

---

## 9. Author’s Note on AI Collaboration
This project was built with **Generative AI used as an assistant / pair programmer**, not as an automatic solution.

AI was used to:
- accelerate boilerplate code (data loading, plotting, model scaffolding),
- debug efficiently,
- speed up iteration on feature ideas.

My core contributions were:
1. **Problem framing:**  
   Defining two realistic prediction moments:
   - **Pre-weekend:** before qualifying / sprint (`leaderboard_pre_week.csv`)
   - **Pre-race:** after qualifying / sprint, just before lights out (`leaderboard_pre_race.csv`)

2. **Validation design:**  
   - Leak-proof `GroupKFold` by `raceId`
   - Clean 2014–2023 → 2024 season holdout split

3. **Feature engineering:**  
   - Rolling performance windows (3 / 5 / 10 races)  
   - Pit stop stability vs. pure speed  
   - Lap pace consistency metrics from raw lap timing  
   - Previous-race snapshots (momentum)

4. **Interpretation for stakeholders:**  
   Translating metrics like “Macro-F1 = 0.84” into an intuitive statement:
   > “We can correctly name ~8–9 of the 10 point scorers for a typical 2024 race before it starts.”

In summary, this repository is an end-to-end F1 race prediction pipeline:
- data engineering,
- feature generation,
- model training / calibration / validation,
- scenario-based deployment (pre-weekend vs pre-race),
- explainability,
- per-race Top 10 leaderboards with confidence scores.

---

### Data Source / Licensing Note
The raw timing and classification data (race results, lap times, pit stops, qualifying, etc.) come from publicly available historical Formula 1 datasets (e.g. Kaggle “Formula 1 World Championship”).  
This repository includes derived feature samples and prediction outputs for demonstration.  
All original credits for the underlying race timing and results data belong to their respective providers / recorders.

---

### How to Reproduce (High-Level)
1. Obtain publicly available F1 historical CSVs (results, races, qualifying, pit_stops, lap_times, sprint_results, etc.).
2. Build rolling-window features per driver/team (3/5/10 races), plus previous-race snapshot columns.
3. Train models on 2014–2023 driver–race rows, with `y_top10` as the label.
4. Calibrate probabilities, then generate predictions for 2024.
5. For each race, rank drivers by predicted Top10 probability and take the top 10 → export as leaderboard.

This mirrors the pipeline used to generate the CSV artifacts in this repo.

---

### Short Version
- We’re not just predicting “who wins.”  
- We’re predicting “who scores points,” which in F1 means the Top 10.  
- Even **before** qualifying, the model can usually name ~8 of those 10 drivers.  
- After qualifying, that goes to ~8–9 out of 10.
