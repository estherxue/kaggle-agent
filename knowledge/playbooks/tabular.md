# Tabular Data Competition Strategies

This playbook covers strategies specific to tabular data competitions.

## Algorithm Selection

### LightGBM (Recommended First Choice)

**When to use:**
- Medium to large datasets (10K - 10M rows)
- Mixed data types (numeric + categorical)
- Need for speed

**Key hyperparameters:**
- `num_leaves`: 31 (default), increase for complex patterns
- `max_depth`: -1 (unlimited) or 6-8 to prevent overfitting
- `learning_rate`: 0.05-0.1, lower for final training
- `n_estimators`: 1000+ with early stopping
- `min_child_samples`: 20 (higher = less overfitting)

**Advantages:**
- Fast training
- Native categorical feature support
- Good default performance
- Handles missing values

### XGBoost

**When to use:**
- Smaller datasets (< 100K rows)
- When LightGBM overfits
- Need for fine control

**Key hyperparameters:**
- `max_depth`: 3-6 (lower than LightGBM)
- `learning_rate`: 0.01-0.1
- `subsample`: 0.8-1.0
- `colsample_bytree`: 0.8-1.0

### CatBoost

**When to use:**
- High cardinality categorical features
- When target encoding is needed
- Less tuning required

**Key hyperparameters:**
- `depth`: 6-10
- `learning_rate`: 0.03-0.1
- `iterations`: 1000+ with early stopping
- `l2_leaf_reg`: 3-10

## Validation Strategies

### K-Fold Cross Validation

**Use when:**
- Dataset is IID (independent and identically distributed)
- No temporal component
- Standard case for most competitions

```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### Stratification

**Always stratify when:**
- Classification problems
- Imbalanced classes
- Small datasets

### Time-Based Splits

**Use when:**
- Time series data
- Data has temporal ordering
- Competition uses time-based split

```python
# Simple time split
train = df[df['date'] < '2023-01-01']
val = df[df['date'] >= '2023-01-01']
```

### Group K-Fold

**Use when:**
- Multiple rows per entity/group
- Prevent leakage between groups

```python
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
```

## Feature Engineering

### Numeric Features

**Standard transformations:**
- Log transform for skewed distributions
- Scaling (StandardScaler, MinMaxScaler)
- Binning for non-linear relationships

**Advanced:**
- Polynomial features (degree 2)
- Ratio/division features between columns
- Statistical aggregations

### Categorical Features

**Encoding strategies (in order of preference):**

1. **Label Encoding** - for tree models, high cardinality
2. **One-Hot Encoding** - for linear models, low cardinality (< 10)
3. **Target Encoding** - best CV, but risk of leakage
4. **Frequency/Count Encoding** - safe, captures popularity

### Date/Time Features

**Always extract:**
- Year, month, day, dayofweek, hour
- Is weekend, is holiday
- Days since reference date

### Text Features

**Basic:**
- Length statistics
- Character/word counts
- Presence of specific patterns

**Advanced:**
- TF-IDF with dimensionality reduction
- Pre-trained embeddings (BERT, etc.)

## Handling Missing Values

### Numeric

- Tree models: -999 (or any constant outside range)
- Mean/median imputation
- Model-based imputation

### Categorical

- "MISSING" or "-1" as new category
- Mode imputation
- Target encoding handles naturally

## Hyperparameter Tuning

### Strategy

1. **Random search** for wide exploration
2. **Bayesian optimization** for refinement (optuna)
3. **Never grid search** - too expensive

### What to tune first

Priority order:
1. Learning rate + iterations
2. num_leaves / max_depth
3. min_child_samples
4. subsample / colsample

### Early Stopping

Always use early stopping to prevent overfitting:

```python
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)
```

## Ensembling

### Weighted Average

Simple and effective:

```python
preds = 0.6 * lgb_preds + 0.3 * xgb_preds + 0.1 * cat_preds
```

### Stacking

**Two-level stacking:**
1. Train base models on training folds
2. Generate OOF predictions
3. Train meta-learner on OOF predictions

**Meta-learners:**
- Linear regression (simple, less overfitting)
- Ridge regression (adds regularization)
- LightGBM (if many base models)

## Competition-Specific Tips

### Imbalanced Classification

- Use appropriate metrics (AUC, F1, log loss)
- Stratified sampling
- Class weights or resampling
- Focus on calibration

### Regression

- Check target distribution (log transform if skewed)
- Outliers handling
- Metric-appropriate loss (RMSE, MAE, RMSLE)

### Multi-Class

- One-vs-Rest vs Softmax
- Calibration matters more
- Class imbalance handling

## Common Patterns

### Baseline Code Structure

```python
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

# Model params
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1,
}

# CV loop
skf = StratifiedKFold(n_splits=5)
oof_preds = np.zeros(len(train))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.train(
        params,
        lgb.Dataset(X_tr, y_tr),
        num_boost_round=1000,
        valid_sets=[lgb.Dataset(X_val, y_val)],
        callbacks=[lgb.early_stopping(50)]
    )

    oof_preds[val_idx] = model.predict(X_val)
```

## Resources

- Kaggle Notebooks: Search for "EDA", "baseline", "feature engineering"
- Papers: Read top solutions after competition ends
- Forums: Check for dataset insights and tips
