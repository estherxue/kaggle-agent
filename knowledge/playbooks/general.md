# General Methodology for Kaggle Competitions

This playbook contains general principles applicable to all types of Kaggle competitions.

## Core Principles

### 1. Start Simple, Iterate Fast

- Begin with a solid baseline before adding complexity
- Make one change at a time to measure impact
- Keep experiments under 10 minutes initially to iterate faster
- Use small data samples for rapid prototyping

### 2. Validation Strategy is Critical

- A bad validation strategy wastes all other efforts
- Match validation to the competition's data split logic
- Monitor CV-LB correlation throughout
- If CV and LB diverge, your validation is wrong

### 3. Understand the Evaluation Metric

- Optimize for the actual metric, not a proxy
- Understand how the metric behaves with edge cases
- Implement the metric yourself to verify understanding

### 4. EDA Before Modeling

- Never skip exploratory data analysis
- Understand feature distributions and relationships
- Identify data quality issues early
- Visualize the target variable

## Workflow

### Phase 1: Understanding (Day 1)

1. Read competition description thoroughly
2. Understand the evaluation metric
3. Download and inspect data files
4. Identify data types and sizes
5. Determine competition type (tabular, CV, NLP)

### Phase 2: EDA (Day 1-2)

1. Basic statistics (shape, dtypes, missing values)
2. Target variable analysis
3. Feature distributions
4. Correlation analysis
5. Identify potential leakage

### Phase 3: Baseline (Day 2)

1. Build a simple model (e.g., LightGBM with default params)
2. Establish validation strategy
3. Get initial CV score
4. Make first submission to establish LB baseline

### Phase 4: Iteration (Week 1+)

1. Feature engineering
2. Model tuning
3. Ensemble methods
4. Error analysis

### Phase 5: Final Push (Final Week)

1. Full training with best params
2. Ensembling/stacking
3. Submission strategy (daily limits)
4. Final validation checks

## Common Pitfalls

### Overfitting the Public LB

- Don't optimize blindly for public leaderboard
- Use time-based splits when available
- Trust your CV more than public LB
- Watch for shake-up potential

### Data Leakage

- Check for features that wouldn't be available at prediction time
- Be wary of ID-like features that encode information
- Temporal data requires careful handling
- Always sanity check features

### Underestimating Preprocessing

- Categorical encoding matters
- Missing value strategy affects results
- Feature scaling for some algorithms
- Text normalization for NLP

## Debugging Checklist

When things aren't working:

- [ ] Is the data loaded correctly?
- [ ] Are features being created as expected?
- [ ] Is the target variable correct?
- [ ] Does the validation split make sense?
- [ ] Is the evaluation metric implemented correctly?
- [ ] Are predictions in the expected format?
- [ ] Is the submission file valid?

## Tooling

### Essential Libraries

- **Data**: pandas, numpy
- **ML**: scikit-learn, LightGBM, XGBoost, CatBoost
- **NLP**: transformers, nltk, spacy
- **CV**: opencv, pillow, albumentations
- **Utils**: tqdm, matplotlib, seaborn

### Environment

- Use Kaggle notebooks for easy data access
- Track experiments (use ExperimentTracker)
- Version control your code
- Document decisions in markdown
