# Olist late-delivery prediction

Predict, at the moment an order is placed, whether it will be delivered **later than the estimated delivery date** shown to the customer. Data: the public Olist Brazilian e-commerce dataset (Kaggle). Late orders are the minority class.

**Short version of the result.** With time-ordered cross-validation on the training period, the model ranks late orders better than chance (pooled out-of-fold PR-AUC 0.232 vs a base rate of 0.084). On the held-out, more recent test period it barely does (PR-AUC 0.047 vs a base rate of 0.035) and its probabilities are far too high. This is a hard, unstable problem on about 20 usable months of data, and the notebook documents it as such. It is not a production-ready model.

Everything is in one notebook, `Late_delivery_prediction_annotated.ipynb`, plus `features.sql` (the SQL it runs).

## Data and label

- Source: *Brazilian E-Commerce Public Dataset by Olist* on Kaggle (<https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>). The CSVs are **not** included in this repo; see "How to run". Check the dataset page for its terms of use before reusing it.
- Unit of analysis: one **delivered order** (96,470 rows in `order_features_v2`; 96,203 after dropping 267 purchases made before 2017-01).
- Label: `late_delivery_flag = 1` if the delivered date is later than `order_estimated_delivery_date`, compared at day level. Only orders with both dates are kept.
- The label is relative to the platform's *estimated* date, and the promised window changed over time (mean `eta_days` 25.1 in train vs 22.0 in test), so "late" is not the same thing as slow transit.

| | rows | late rate |
|---|---|---|
| Train | 70,332 | 7.67% |
| Test | 19,242 | 3.50% |

The monthly late rate ranges from about 1% to 19% (about 3% in calm months).

## Features (purchase-time information only)

Anything known only after the outcome (delivered date, carrier hand-off date, review score) is excluded. Built in SQL with DuckDB (`features.sql`):

- Order content: price, freight, freight ratio, weights, volume and product dimensions, item/seller/product counts, top category.
- Geography: seller and customer state, same-state flag, seller-to-customer distance (haversine on the median lat/lng of each zip prefix), distance per promised day.
- Timing: promised window `eta_days`, purchase hour and day of week.
- Point-in-time history, using only deliveries on days **strictly before** the order's purchase date: past orders and shrunken late rates for the seller, customer zip, state and category, and trailing 7/14-day platform volume and late rate. Shrinkage is `(late + K * parent_rate) / (n + K)` with `K_PRIOR = 20` (an assumption, not tuned), with the hierarchy zip -> state -> platform and seller/category -> platform.

After the SQL step the notebook adds `distance_unknown` and `new_seller` flags and a `log1p` transform on skewed columns, and removes identifiers, timestamps, the label and `month` (51 candidate feature columns).

## Method

- **Split.** Time-based. The cutoff is the 80th percentile of purchase time; the training set ends one `window` (95th percentile of delivery time) before the cutoff, so training labels were known at the cutoff. The orders in that gap are dropped.
- **Cross-validation.** `TimeSeriesSplit(n_splits=5)` (expanding window, `gap=0`). Scored by average precision.
- **Models.** Logistic Regression baseline (median imputation with missing indicators, scaling, one-hot encoding, `C = 0.03`) and XGBoost (`hist`, native categoricals; 20 + 30 random-search candidates scored on the same folds).
- **Metrics.** PR-AUC (average precision) and precision/recall at a threshold. Accuracy is not used.
- **Threshold.** Chosen on out-of-fold (OOF) predictions as the highest-recall threshold with at least 20% precision. The 20% target is an assumption; there is **no cost-based threshold** in this project.
- **Interpretation.** SHAP (TreeExplainer), a monthly check of one feature, an ablation, and a minimal error analysis on OOF predictions.

## Results

### Cross-validation, per fold (training period)

| fold | late rate | AP, Logistic Regression | AP, XGBoost |
|---|---|---|---|
| 1 | 0.030 | 0.055 | 0.052 |
| 2 | 0.063 | 0.104 | 0.107 |
| 3 | 0.088 | 0.196 | 0.249 |
| 4 | 0.138 | 0.328 | 0.279 |
| 5 | 0.101 | 0.286 | 0.323 |
| mean | | 0.194 | 0.202 |

The fold-to-fold standard deviation is about 0.10 and XGBoost wins 3 of 5 folds with margins that change sign, so the two models are treated as **a tie**. XGBoost is kept for explainability, not because it scored better. Because the same folds were used to tune XGBoost and to compare the models, 0.202 is an upper estimate.

### Out-of-fold predictions and threshold (training period)

Pooled OOF average precision is 0.232 against a base rate of 0.084 (lift 2.76x). Pooling mixes folds with different late rates, so it is not comparable with the per-fold means above.

| target precision | threshold | recall | share of orders flagged |
|---|---|---|---|
| 0.15 | 0.049 | 0.692 | 0.388 |
| 0.20 (chosen) | 0.080 | 0.506 | 0.213 |
| 0.30 | 0.155 | 0.261 | 0.073 |

### Test period (one evaluation, threshold 0.080)

| | value |
|---|---|
| late rate | 0.035 |
| PR-AUC | 0.047 (lift 1.34x) |
| share of orders flagged | 0.456 |
| precision | 0.042 |
| recall | 0.550 |
| mean predicted probability | 0.120 (actual 0.035) |

The model over-predicts throughout the test period, and recall of 55% mostly reflects flagging almost half of all orders. By month:

| month | orders | actual late rate | mean predicted |
|---|---|---|---|
| 2018-05 | 639 | 0.002 | 0.031 |
| 2018-06 | 6,096 | 0.012 | 0.078 |
| 2018-07 | 6,156 | 0.034 | 0.112 |
| 2018-08 | 6,351 | 0.062 | 0.177 |

Monthly ranking metrics rest on very few late orders (May has about one) and are not reported here.

### Why does it transfer poorly? (a hypothesis, not a proven cause)

- SHAP on 3,000 test rows vs 3,000 rows from the last 5,000 training rows: the mean SHAP value of `platform_late_rate_14d` moves from -0.550 to +0.039. The platform-level time-varying features together account for about 80% of the top-10 shift (my arithmetic on the displayed values).
- `platform_late_rate_14d` lags the real late rate (it is built from orders delivered in the previous 14 days); its correlation with the monthly late rate is 0.22 over 20 months.
- **Ablation** (training period only, same folds): removing the platform features changes mean AP by -0.004 to -0.012, which is within fold-to-fold noise (mean AP 0.202 with all features vs 0.189 with none of the five platform features). Cross-validation therefore neither confirms nor rules out this explanation, and no variant without these features was evaluated on the test set.

### Error analysis (OOF predictions, minimal)

Calm months get almost no alerts (recall 1.6-20.3%), the largest spike months (2017-11, 2018-02, 2018-03) are under-predicted, many alerts come right after spikes, and recall is lowest for the shortest promised windows (0.29 vs 0.50-0.59). These are descriptive patterns; several slices were examined and each calm month has only 73-187 late orders. Slices by state, category, distance and seller history were not done.

## Limitations

- About 20 usable months (2017-01 to 2018-08) with a few spike episodes: very few independent examples of how platform-wide conditions behave. Results depend strongly on the period.
- One test window (roughly May-Aug 2018), calmer than training.
- **Test-set reuse.** The test set was evaluated once, then looked at again while diagnosing the drift (SHAP, monthly check). Any change made after that point is post hoc. The ablation and error analysis use the training period only.
- The XGBoost search and the model comparison use the same folds; `K_PRIOR = 20` was not tuned by CV. Re-running the same model moves fold scores by up to about 0.01 (cause not investigated).
- Cross-validation uses `gap=0`, unlike the train/test split. Labels of orders delivered after a validation block starts may therefore be used in training for that fold, so CV scores may be slightly optimistic. This was not measured.
- History features count only orders that were eventually delivered (96,478 of 99,441 orders). The effect of this on the features was not measured.
- Probabilities are not calibrated and no calibration plot was made; on the test set the mean prediction is about 3.4 times the actual rate.
- The 20% target precision is an assumption; there is no cost-based threshold.
- About 1,278 multi-seller orders are assigned to the highest-priced item's seller; 476 orders (0.5%) have no distance; the geolocation is a zip-prefix median.
- Feature correlations with the label are weak (largest 0.115).

## Repository

| file | what it is |
|---|---|
| `Late_delivery_prediction_annotated.ipynb` | the full analysis, outputs included |
| `features.sql` | DuckDB SQL: load CSVs, `haversine_km` macro, `features_base`, `order_features_v2` |
| `requirements.txt` | Python packages |
| `.gitignore` | keeps the data and the DuckDB file out of the repo |

## How to run

1. Download the Olist CSVs from Kaggle into a `data/` folder next to the notebook.
2. `pip install -r requirements.txt`
3. Create the database and features: `duckdb olist.duckdb < features.sql` (DuckDB CLI), or run the SQL cells in the notebook. The `haversine_km` macro is defined in `features.sql`; the notebook itself only documents it, so run `features.sql` first.
4. Run the notebook top to bottom. Results vary slightly between runs (see limitations).

`features.sql` was copied from the notebook cells and has not been executed in the environment where it was written.
