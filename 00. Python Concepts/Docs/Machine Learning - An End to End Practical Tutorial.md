# Machine Learning: An End-to-End Practical Tutorial

A complete, hands-on path from raw data to a deployed model. Every section has: the idea, the math intuition, working code, when to use it, what goes wrong, and an exercise.

**How to use this:** don't read it front to back like a novel. Read a section, type the code yourself (don't copy-paste), do the exercise, move on. Budget 8–12 weeks at a few hours a week.

---

## Table of Contents

**Part I — Foundations**
1. What ML actually is
2. Environment setup
3. The universal ML workflow
4. Your first end-to-end model (30 minutes)

**Part II — Data**
5. Exploratory data analysis
6. Cleaning and missing values
7. Feature engineering
8. Encoding, scaling, and pipelines

**Part III — Evaluation**
9. Train/test/validation and cross-validation
10. Regression metrics
11. Classification metrics
12. Bias, variance, and learning curves

**Part IV — Supervised algorithms**
13. Linear regression + regularization
14. Logistic regression
15. k-Nearest Neighbors
16. Naive Bayes
17. Support Vector Machines
18. Decision Trees
19. Random Forests
20. Gradient Boosting (XGBoost / LightGBM / CatBoost)
21. Algorithm cheat sheet

**Part V — Getting more out of models**
22. Hyperparameter tuning
23. Imbalanced data
24. Model interpretation (SHAP, PDP, permutation importance)
25. Ensembling and stacking

**Part VI — Unsupervised learning**
26. K-Means, hierarchical, DBSCAN
27. PCA, t-SNE, UMAP
28. Anomaly detection

**Part VII — Specialized domains**
29. Time series ML
30. Natural language processing
31. Neural networks with PyTorch
32. Recommender systems

**Part VIII — Shipping it**
33. Persistence, APIs, and serving
34. Monitoring, drift, and retraining
35. Experiment tracking

**Part IX — Practice**
36. Ten mini-projects
37. Exercise solutions
38. Study plan and resources

---

# PART I — FOUNDATIONS

## 1. What ML actually is

Traditional programming: you write rules, the computer applies them to data.
Machine learning: you give the computer data *and* answers, and it finds the rules.

```
Traditional:  rules + data  →  answers
ML:           data + answers → rules  (the "model")
```

That's the whole idea. Everything else is detail about *how* the rules get found and *how you know they're any good*.

### The three families

| Family | You have | You want | Examples |
|---|---|---|---|
| **Supervised** | Inputs X *and* labels y | Predict y for new X | Spam detection, price prediction, churn |
| **Unsupervised** | Only inputs X | Find structure | Customer segments, anomaly detection, compression |
| **Reinforcement** | An environment + rewards | A policy | Game playing, robotics, ad bidding |

Supervised learning is ~90% of applied ML. This tutorial spends most of its time there.

Supervised splits into two:
- **Regression** — predict a number (house price, temperature, demand)
- **Classification** — predict a category (spam/not, churn/stay, which of 10 digits)

### Vocabulary you must own

| Term | Meaning |
|---|---|
| **Feature** (X, predictor, independent variable) | An input column |
| **Target** (y, label, dependent variable) | What you're predicting |
| **Instance / sample / row** | One observation |
| **Model** | A function `f(X) → ŷ` with learnable parameters |
| **Parameters** | Learned from data (e.g. regression coefficients) |
| **Hyperparameters** | Set by you before training (e.g. tree depth, learning rate) |
| **Loss function** | Measures how wrong a prediction is; training minimizes it |
| **Training** | Adjusting parameters to reduce loss |
| **Inference** | Using the trained model on new data |
| **Overfitting** | Model memorizes training noise; great on train, bad on new data |
| **Underfitting** | Model is too simple to capture the real pattern |
| **Generalization** | Performance on data the model has never seen. The *only* thing that matters. |

### The one law of machine learning

> **Your model will be evaluated on data it has never seen. Anything you do that leaks information from the test set into training is cheating, and you will only find out in production.**

Write that on a sticky note. Ninety percent of ML failures in industry are variations of breaking this rule.

---

## 2. Environment setup

```bash
# Create an isolated environment (use whichever you prefer)
python -m venv mlenv
source mlenv/bin/activate        # Windows: mlenv\Scripts\activate

# Or with conda
conda create -n mlenv python=3.11 -y
conda activate mlenv
```

Core stack:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyterlab
pip install xgboost lightgbm catboost
pip install shap optuna imbalanced-learn
pip install statsmodels
# Deep learning (pick one)
pip install torch torchvision
# NLP
pip install transformers datasets sentence-transformers
# Serving
pip install fastapi uvicorn joblib
```

What each one is for:

| Library | Role |
|---|---|
| **numpy** | N-dimensional arrays, vectorized math. Everything sits on top of it. |
| **pandas** | Tabular data: load, clean, join, group, reshape. |
| **scikit-learn** | The workhorse. Classical algorithms, preprocessing, model selection, metrics — all with one consistent API. |
| **matplotlib / seaborn** | Plots. Seaborn is a nicer front-end for statistical charts. |
| **xgboost / lightgbm / catboost** | Gradient boosted trees. These win most tabular competitions. |
| **shap** | Explain any model's individual predictions. |
| **optuna** | Smart hyperparameter search. |
| **imbalanced-learn** | Resampling for skewed class distributions. |
| **statsmodels** | Statistical inference — p-values, confidence intervals, classical time series. |
| **torch** | Neural networks. |

Start Jupyter:

```bash
jupyter lab
```

### The scikit-learn API (learn this once, use it forever)

Every estimator in sklearn follows the same three-method contract:

```python
model = SomeAlgorithm(hyperparam=value)   # 1. construct
model.fit(X_train, y_train)               # 2. learn from data
predictions = model.predict(X_test)       # 3. apply to new data
```

Transformers (things that change data rather than predict) add:

```python
transformer.fit(X_train)                  # learn the transformation
X_train_t = transformer.transform(X_train)
X_test_t  = transformer.transform(X_test) # SAME transformation, no refitting
# shortcut for train:
X_train_t = transformer.fit_transform(X_train)
```

**Critical:** you `fit_transform` on training data and only `transform` on test data. Fitting a scaler on the test set leaks test statistics into your pipeline. This is the single most common beginner bug.

---

## 3. The universal ML workflow

Every real project follows this loop. Steps 1–3 take 70% of the time; everyone wishes it were step 5.

```
1. FRAME THE PROBLEM
   What decision does this prediction inform? What does "good" mean numerically?
   What's the baseline (current process / simple heuristic) you must beat?

2. GET AND UNDERSTAND THE DATA
   Load it. Look at it. Plot it. Find the lies in it.

3. PREPARE THE DATA
   Clean, handle missing values, engineer features, encode, scale.
   Split BEFORE you do any of this (or do it inside a Pipeline).

4. PICK A BASELINE MODEL
   Dumbest reasonable thing: predict the mean, or the majority class.
   Then a simple real model: linear/logistic regression.

5. TRY BETTER MODELS
   Random forest, gradient boosting. Compare with cross-validation.

6. TUNE
   Hyperparameter search on the best 1–2 candidates.

7. EVALUATE HONESTLY
   One final run on the held-out test set. Look at errors, not just the metric.

8. INTERPRET
   Which features drive predictions? Does it make domain sense? Any leakage?

9. DEPLOY
   Serialize, wrap in an API or batch job, log inputs and outputs.

10. MONITOR
    Data drift, performance decay, retrain schedule.
```

### Frame the problem properly

Before touching code, answer these:

- **What is one row?** (a customer? a customer-month? a transaction?) Getting this wrong invalidates everything downstream.
- **What exactly is y?** "Churn" — churned within 30 days? 90? Cancelled or just inactive?
- **What features will actually be available at prediction time?** If a feature is only known *after* the event you're predicting, it's leakage. (Classic: using `total_charges` to predict churn, when total charges are only final once someone churns.)
- **What's the cost of each error type?** A false negative on cancer screening ≠ a false positive.
- **What's the baseline?** If 95% of customers don't churn, a model with 95% accuracy is worthless.

---

## 4. Your first end-to-end model (30 minutes)

Let's do the whole loop on a real dataset before learning any theory. Predicting whether a passenger survived the Titanic — small, messy, has numeric + categorical + missing values, which makes it a perfect teaching set.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

# ---------- 1. LOAD ----------
df = sns.load_dataset("titanic")
print(df.shape)
print(df.head())
print(df.info())
```

```python
# ---------- 2. LOOK ----------
print(df["survived"].value_counts(normalize=True))   # baseline: 62% died
print(df.isna().sum().sort_values(ascending=False))

# Survival by sex and class
print(df.groupby(["sex", "pclass"])["survived"].mean())
```

```python
# ---------- 3. SELECT FEATURES ----------
# Drop leaky/duplicate columns: 'alive' IS the target; 'deck' is mostly missing;
# 'adult_male','who','class','embark_town' duplicate other columns.
features_num = ["age", "fare", "sibsp", "parch"]
features_cat = ["sex", "pclass", "embarked", "alone"]

X = df[features_num + features_cat]
y = df["survived"]

# ---------- 4. SPLIT FIRST. ALWAYS. ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

```python
# ---------- 5. PREPROCESSING PIPELINE ----------
numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, features_num),
    ("cat", categorical_pipe, features_cat),
])
```

```python
# ---------- 6. BASELINE + MODELS ----------
from sklearn.dummy import DummyClassifier

models = {
    "dummy":    DummyClassifier(strategy="most_frequent"),
    "logistic": LogisticRegression(max_iter=1000),
    "forest":   RandomForestClassifier(n_estimators=300, random_state=42),
}

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"{name:10s} CV ROC-AUC: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

```python
# ---------- 7. FINAL EVALUATION (once!) ----------
final = Pipeline([("prep", preprocess),
                  ("model", RandomForestClassifier(n_estimators=300,
                                                   min_samples_leaf=3,
                                                   random_state=42))])
final.fit(X_train, y_train)

y_pred  = final.predict(X_test)
y_proba = final.predict_proba(X_test)[:, 1]

print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("ROC-AUC :", round(roc_auc_score(y_test, y_proba), 3))
print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d",
            xticklabels=["pred died", "pred survived"],
            yticklabels=["died", "survived"], cmap="Blues")
plt.show()
```

```python
# ---------- 8. INTERPRET ----------
feat_names = final.named_steps["prep"].get_feature_names_out()
importances = final.named_steps["model"].feature_importances_
imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)
print(imp.head(10))
imp.head(10).plot.barh().invert_yaxis()
plt.show()
```

You just did the entire workflow. Everything that follows is depth on each step.

**Exercise 4.1** — Re-run the above but add a `family_size = sibsp + parch + 1` feature and an `is_child = age < 12` feature. Does CV AUC improve? By how much?

**Exercise 4.2** — Change `test_size` to 0.5 and re-run. Why does the test score become both worse *and* more stable? Which effect dominates and why?

---

# PART II — DATA

## 5. Exploratory data analysis (EDA)

EDA is not optional decoration. It's where you discover that 30% of your dates are in the future, that "gender" has seven spellings, and that your target is 99.7% one class.

### A standard first-pass script

```python
def explore(df, target=None):
    print("=" * 60)
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("=" * 60)

    info = pd.DataFrame({
        "dtype":     df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "n_unique":  df.nunique(),
        "sample":    [df[c].dropna().iloc[0] if df[c].notna().any() else None
                      for c in df.columns],
    })
    print(info.sort_values("pct_missing", ascending=False))

    print("\n--- Numeric summary ---")
    print(df.describe().T)

    dup = df.duplicated().sum()
    print(f"\nDuplicate rows: {dup:,}")

    if target:
        print(f"\n--- Target: {target} ---")
        if df[target].dtype.kind in "ifc" and df[target].nunique() > 20:
            print(df[target].describe())
        else:
            print(df[target].value_counts(normalize=True))

explore(df, target="survived")
```

### The visual checklist

```python
# 1. Distribution of every numeric feature — spot skew, outliers, weird spikes
df.select_dtypes("number").hist(figsize=(14, 10), bins=40)
plt.tight_layout(); plt.show()

# 2. Correlation heatmap — spot redundancy and target relationships
corr = df.select_dtypes("number").corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.show()

# 3. Target vs each feature
for col in ["age", "fare"]:
    sns.boxplot(data=df, x="survived", y=col)
    plt.show()

# 4. Categorical vs target
for col in ["sex", "pclass", "embarked"]:
    (df.groupby(col)["survived"].mean()
       .sort_values().plot.barh(title=f"Survival rate by {col}"))
    plt.show()

# 5. Missingness pattern — is it random or structured?
sns.heatmap(df.isna(), cbar=False, cmap="viridis")
plt.show()
```

### What you're hunting for

| Red flag | What it means | What to do |
|---|---|---|
| A feature correlates 0.99 with the target | Leakage | Remove it, understand why |
| A column is 95% missing | Little signal | Drop, or convert to `has_value` flag |
| A "numeric" column has 3 unique values | It's categorical | Recode |
| Class balance 99:1 | Accuracy is meaningless | See Part 23 |
| Duplicated rows | Data pipeline bug, or legitimately repeated events | Investigate before dropping |
| Wildly different train/test distributions | Non-random split, or drift | Fix the split |
| Impossible values (age=200, price=-5) | Sentinel codes or entry errors | Treat as missing |

**Exercise 5.1** — Load `sns.load_dataset("diamonds")`. Find at least three data-quality problems (hint: look at the minimum of `x`, `y`, `z`). Write a paragraph on what you'd do about each.

---

## 6. Cleaning and missing values

### Why data is missing matters

| Mechanism | Meaning | Example | Safe to impute? |
|---|---|---|---|
| **MCAR** (completely at random) | Missingness unrelated to anything | Sensor randomly drops readings | Yes |
| **MAR** (at random) | Depends on *observed* features | Income missing more often for younger users | Yes, using other features |
| **MNAR** (not at random) | Depends on the *missing value itself* | High earners refuse to state income | Dangerous — add a missingness indicator |

In practice, always consider adding an indicator column: the fact that something is missing is often itself predictive.

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

# Simple: constant / mean / median / most_frequent
imp = SimpleImputer(strategy="median", add_indicator=True)  # <- keeps the flag

# KNN: fill from k most similar rows
imp = KNNImputer(n_neighbors=5)

# Iterative (MICE): model each feature from the others, round-robin
imp = IterativeImputer(max_iter=10, random_state=0)
```

Rules of thumb:
- **Numeric** → median (robust to outliers). Mean only if roughly symmetric.
- **Categorical** → most frequent, or an explicit `"Missing"` category. Often the latter is better — it preserves information.
- **Time series** → forward-fill (`ffill`) if the value persists; interpolate if it changes smoothly. Never backward-fill in a forecasting context — that's leakage from the future.
- **>50% missing** → usually drop the column, but keep a binary "was present" flag.

### Outliers

```python
# IQR rule
Q1, Q3 = df["fare"].quantile([0.25, 0.75])
IQR = Q3 - Q1
mask = (df["fare"] < Q1 - 1.5*IQR) | (df["fare"] > Q3 + 1.5*IQR)
print(f"{mask.sum()} outliers")

# Z-score (assumes roughly normal)
z = (df["fare"] - df["fare"].mean()) / df["fare"].std()
mask = z.abs() > 3

# Winsorize instead of dropping — clip to percentiles
df["fare_clipped"] = df["fare"].clip(*df["fare"].quantile([0.01, 0.99]))

# Robust scaling handles outliers without removing rows
from sklearn.preprocessing import RobustScaler
```

**Do not reflexively delete outliers.** In fraud detection and anomaly detection, the outliers *are the signal*. Ask: is this a data-entry error, or a real rare event? Only delete the former.

### Duplicates, types, and text hygiene

```python
df = df.drop_duplicates()
df = df.drop_duplicates(subset=["user_id", "date"], keep="last")  # domain-aware

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["price"] = pd.to_numeric(df["price"].astype(str).str.replace(",", ""),
                            errors="coerce")

# Categorical hygiene
df["city"] = df["city"].str.strip().str.lower().str.title()
df["city"] = df["city"].replace({"Bangalore": "Bengaluru"})

# Memory: convert low-cardinality strings to category dtype
for c in df.select_dtypes("object"):
    if df[c].nunique() / len(df) < 0.05:
        df[c] = df[c].astype("category")
```

**Exercise 6.1** — Write a function `missing_report(df)` that returns a DataFrame with, for each column: % missing, and the mean of the target for rows where that column is missing vs. not missing. This tells you whether missingness itself is predictive.

---

## 7. Feature engineering

> "Applied machine learning is basically feature engineering." — Andrew Ng

A gradient boosting model with good features beats a neural network with bad ones on tabular data, every time. This is where domain knowledge converts into performance.

### Numeric transformations

```python
# Log transform for right-skewed data (income, prices, counts)
df["log_fare"] = np.log1p(df["fare"])     # log1p handles zeros

# Binning — turns a continuous feature into ordinal buckets
df["age_bin"] = pd.cut(df["age"], bins=[0, 12, 18, 35, 60, 100],
                       labels=["child", "teen", "young", "adult", "senior"])
df["fare_decile"] = pd.qcut(df["fare"], 10, labels=False, duplicates="drop")

# Ratios and differences often beat raw values
df["fare_per_person"] = df["fare"] / (df["sibsp"] + df["parch"] + 1)

# Polynomial / interaction features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# Power transforms to make data more Gaussian
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method="yeo-johnson")   # handles negatives too
```

### Date/time features

Never feed a raw timestamp to a model. Decompose it.

```python
d = df["date"]
df["year"]        = d.dt.year
df["month"]       = d.dt.month
df["day"]         = d.dt.day
df["dayofweek"]   = d.dt.dayofweek
df["quarter"]     = d.dt.quarter
df["is_weekend"]  = d.dt.dayofweek.isin([5, 6]).astype(int)
df["is_month_end"]= d.dt.is_month_end.astype(int)
df["days_since"]  = (d.max() - d).dt.days

# Cyclical encoding — so December is adjacent to January
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
```

Cyclical encoding matters: with a plain integer, the model sees month 12 and month 1 as maximally distant, when they're actually adjacent.

### Aggregation features (the big win for transactional data)

If one row is a transaction but you're predicting something about a *customer*, aggregate:

```python
agg = (transactions.groupby("customer_id")
       .agg(n_orders      = ("order_id", "count"),
            total_spend   = ("amount", "sum"),
            avg_spend     = ("amount", "mean"),
            std_spend     = ("amount", "std"),
            max_spend     = ("amount", "max"),
            n_categories  = ("category", "nunique"),
            first_order   = ("date", "min"),
            last_order    = ("date", "max"))
       .reset_index())

agg["tenure_days"]  = (agg["last_order"] - agg["first_order"]).dt.days
agg["recency_days"] = (pd.Timestamp.today() - agg["last_order"]).dt.days
agg["order_freq"]   = agg["n_orders"] / agg["tenure_days"].clip(lower=1)
```

This RFM-style pattern (Recency, Frequency, Monetary) is the backbone of most customer-level models.

### Target encoding (powerful, dangerous)

Replace a high-cardinality category with the mean of the target for that category. Great for things like ZIP code or product ID. **Must be fit inside cross-validation folds or it leaks catastrophically.**

```python
from sklearn.preprocessing import TargetEncoder   # sklearn >= 1.3
te = TargetEncoder(smooth="auto", cv=5)           # internally cross-fitted
X_enc = te.fit_transform(X[["zipcode"]], y)
```

Or use `category_encoders` for more variants (leave-one-out, CatBoost encoder, WOE).

### The leakage checklist

Before you train, ask of every feature:

1. **Would I know this value at the moment I need to make the prediction?** If a feature is populated after the outcome, it's leakage.
2. **Does this feature contain the target in disguise?** (`total_revenue` when predicting `made_a_purchase`)
3. **Did I compute this using the full dataset?** (scaling, imputation, target encoding fit before the split)
4. **For time series: does any feature use future information?** Rolling means must be backward-looking only.

A model that's suspiciously good is almost never a breakthrough. It's leakage.

**Exercise 7.1** — Take the diamonds dataset. Engineer at least five features (volume, density ratios, cut/color interaction, log price...). Compare a LinearRegression on raw features vs. engineered features using 5-fold CV R².

**Exercise 7.2** — You're predicting whether a loan will default. You have: `application_date`, `income`, `loan_amount`, `credit_score`, `n_late_payments`, `collections_agency_assigned`, `final_status`. Which columns are leaky and why?

---

## 8. Encoding, scaling, and pipelines

### Encoding categorical variables

| Method | Use when | Watch out for |
|---|---|---|
| **One-hot** | Low cardinality (<15), no order | Explodes dimensionality |
| **Ordinal** | Natural order (small < medium < large) | Implies false distances if no real order |
| **Target/mean encoding** | High cardinality (ZIP, product ID) | Leakage — must be cross-fitted |
| **Frequency encoding** | High cardinality, frequency is meaningful | Collides distinct categories with equal counts |
| **Hashing** | Very high cardinality, streaming | Hash collisions, uninterpretable |
| **Embeddings** | Neural nets, very high cardinality | Needs lots of data |
| **Native handling** | LightGBM/CatBoost accept categories directly | Only in those libraries |

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                    min_frequency=0.01)     # lump rare levels together

ordinal = OrdinalEncoder(categories=[["Fair","Good","Very Good","Premium","Ideal"]],
                         handle_unknown="use_encoded_value", unknown_value=-1)
```

`handle_unknown="ignore"` is essential: production will show you categories that weren't in training.

### Scaling

| Scaler | Formula | Use when |
|---|---|---|
| **StandardScaler** | (x − μ) / σ | Default; roughly normal data |
| **MinMaxScaler** | (x − min) / (max − min) | Bounded output needed (e.g. neural nets) |
| **RobustScaler** | (x − median) / IQR | Heavy outliers |
| **Normalizer** | Scale each *row* to unit norm | Text/TF-IDF vectors |
| **QuantileTransformer** | Map to uniform/normal by rank | Very non-normal data |

**Which models need scaling?**

| Needs scaling | Doesn't care |
|---|---|
| Linear/logistic regression *with regularization* | Decision trees |
| SVM, KNN, K-Means, PCA | Random forest |
| Neural networks | Gradient boosting (XGBoost/LGBM/CatBoost) |
| Anything distance- or gradient-based | Naive Bayes |

Rule: if the algorithm computes distances or penalizes coefficient magnitude, scale.

### Pipelines — the most important habit in this document

A `Pipeline` chains preprocessing and modeling into one object. It guarantees that every transformation is fitted only on training folds. Once you use pipelines, entire classes of leakage bugs become impossible.

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

numeric_features = make_column_selector(dtype_include=np.number)
categorical_features = make_column_selector(dtype_include=object)

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc",  StandardScaler())]), numeric_features),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh",  OneHotEncoder(handle_unknown="ignore"))]),
                categorical_features),
    ],
    remainder="drop",
    verbose_feature_names_out=True,
)

pipe = Pipeline([
    ("prep",  preprocess),
    ("model", LogisticRegression(max_iter=1000)),
])

pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
```

The whole pipeline is now a single estimator: you can cross-validate it, grid-search across preprocessing *and* model hyperparameters simultaneously, and pickle it as one deployable artifact.

```python
# Tuning across the whole pipeline
from sklearn.model_selection import GridSearchCV

param_grid = {
    "prep__num__imp__strategy": ["mean", "median"],
    "model__C": [0.01, 0.1, 1, 10],
}
gs = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
gs.fit(X_train, y_train)
print(gs.best_params_, gs.best_score_)
```

### Custom transformers

```python
from sklearn.base import BaseEstimator, TransformerMixin

class FamilyFeatures(BaseEstimator, TransformerMixin):
    """Adds family_size and is_alone from sibsp/parch."""
    def fit(self, X, y=None):
        return self                      # nothing to learn

    def transform(self, X):
        X = X.copy()
        X["family_size"] = X["sibsp"] + X["parch"] + 1
        X["is_alone"] = (X["family_size"] == 1).astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(list(input_features) + ["family_size", "is_alone"])

# Quick one-off version:
from sklearn.preprocessing import FunctionTransformer
log_tf = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
```

**Exercise 8.1** — Build a single pipeline for the diamonds dataset that ordinal-encodes `cut`, `color`, `clarity` in their correct orders, log-transforms `price`... wait, `price` is the target. Log-transform `carat`, scales numerics, and fits a Ridge regression. Grid search over `alpha`.

**Exercise 8.2** — Write a custom transformer `RareCategoryGrouper(min_freq=0.01)` that replaces categories appearing in less than `min_freq` of rows with `"__other__"`, learned from training data only.

---

# PART III — EVALUATION

## 9. Splitting and cross-validation

### The three-way split

```
Full data
├── Training set   (~60%)  → fit model parameters
├── Validation set (~20%)  → choose hyperparameters / compare models
└── Test set       (~20%)  → ONE final honest estimate. Touch once.
```

Every time you look at the test set and change something, you leak a little information into your model choice. Do that fifty times and your test score is as optimistic as a training score. Lock it away.

```python
from sklearn.model_selection import train_test_split

# Stratify for classification to preserve class ratios
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
# 0.25 of the remaining 80% = 20% of the original
```

### Cross-validation

Rather than one validation split, rotate through k folds. Every point gets used for both training and validation. Gives a mean *and a standard deviation* — the spread tells you how stable the estimate is.

```python
from sklearn.model_selection import (KFold, StratifiedKFold, GroupKFold,
                                     TimeSeriesSplit, cross_val_score,
                                     cross_validate, RepeatedStratifiedKFold)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = cross_validate(pipe, X_train, y_train, cv=cv,
                         scoring=["roc_auc", "f1", "precision", "recall"],
                         return_train_score=True, n_jobs=-1)

for k, v in results.items():
    if k.startswith(("test_", "train_")):
        print(f"{k:22s} {v.mean():.3f} ± {v.std():.3f}")
```

Comparing `train_` vs `test_` scores immediately tells you about overfitting: a large gap means the model is memorizing.

### Choosing the right splitter

| Splitter | When |
|---|---|
| `KFold` | Regression, independent rows |
| `StratifiedKFold` | Classification — keeps class balance in every fold |
| `GroupKFold` | Rows are grouped (multiple rows per patient/user); keeps a group entirely in one fold |
| `TimeSeriesSplit` | Temporal data — trains on past, validates on future only |
| `RepeatedStratifiedKFold` | Small datasets — repeats CV with different shuffles for a tighter estimate |
| `LeaveOneOut` | Very small datasets (<100 rows); expensive and high variance |

```python
# Time series: expanding window, never trains on the future
tscv = TimeSeriesSplit(n_splits=5, test_size=90, gap=1)
for train_idx, test_idx in tscv.split(X):
    print(f"train: {train_idx.min()}-{train_idx.max()}  "
          f"test: {test_idx.min()}-{test_idx.max()}")
```

**Grouped data is the silent killer.** If you have five records per patient and split randomly, the same patient appears in train and test, and your model looks brilliant because it memorized patients. Use `GroupKFold(groups=patient_id)`.

**Exercise 9.1** — Simulate the group-leakage problem: create a dataset with 100 users × 10 rows each, where the target depends only on a per-user random effect. Show that KFold gives high CV accuracy while GroupKFold gives chance-level accuracy. Explain which one is telling the truth.

---

## 10. Regression metrics

| Metric | Formula | Reads as | Notes |
|---|---|---|---|
| **MAE** | mean(\|y − ŷ\|) | Average error in target units | Robust to outliers, easy to explain |
| **MSE** | mean((y − ŷ)²) | — | Penalizes large errors heavily |
| **RMSE** | √MSE | Error in target units | Most common; outlier-sensitive |
| **R²** | 1 − SS_res/SS_tot | Fraction of variance explained | Can be negative (worse than the mean) |
| **Adjusted R²** | R² penalized by #features | Same, fairer for model comparison | Use when comparing different feature counts |
| **MAPE** | mean(\|y − ŷ\|/\|y\|) | % error | Explodes near zero; asymmetric |
| **SMAPE** | symmetric version | % error | Better behaved than MAPE |
| **RMSLE** | RMSE of log1p values | Relative error | Use when target spans orders of magnitude |
| **Pinball loss** | quantile loss | For quantile regression | Prediction intervals |

```python
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, mean_absolute_percentage_error)

def regression_report(y_true, y_pred):
    return pd.Series({
        "MAE":   mean_absolute_error(y_true, y_pred),
        "RMSE":  mean_squared_error(y_true, y_pred) ** 0.5,
        "R2":    r2_score(y_true, y_pred),
        "MAPE":  mean_absolute_percentage_error(y_true, y_pred),
    })

print(regression_report(y_test, y_pred))
```

**Which to optimize?** MAE if all errors are equally bad per unit. RMSE if large errors are disproportionately costly. RMSLE if you care about relative error (predicting 10 when truth is 100 is as bad as predicting 1,000 when truth is 10,000).

### Always plot residuals

```python
resid = y_test - y_pred
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].scatter(y_pred, resid, alpha=0.4); ax[0].axhline(0, c="r")
ax[0].set(xlabel="Predicted", ylabel="Residual", title="Residuals vs fitted")
ax[1].hist(resid, bins=40); ax[1].set_title("Residual distribution")
ax[2].scatter(y_test, y_pred, alpha=0.4)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax[2].plot(lims, lims, "r--"); ax[2].set(xlabel="Actual", ylabel="Predicted")
plt.tight_layout(); plt.show()
```

A funnel shape in residuals-vs-fitted means heteroscedasticity (error grows with magnitude) — try log-transforming the target. Curvature means you're missing a nonlinear term.

---

## 11. Classification metrics

### The confusion matrix is the source of everything

```
                  Predicted
                  Neg     Pos
        Neg  |   TN   |   FP  |   ← FP = false alarm  (Type I error)
Actual       |--------|-------|
        Pos  |   FN   |   TP  |   ← FN = miss         (Type II error)
```

| Metric | Formula | Question it answers |
|---|---|---|
| **Accuracy** | (TP+TN)/all | What fraction did I get right? |
| **Precision** | TP/(TP+FP) | When I say positive, how often am I right? |
| **Recall / Sensitivity / TPR** | TP/(TP+FN) | Of all actual positives, how many did I catch? |
| **Specificity / TNR** | TN/(TN+FP) | Of all actual negatives, how many did I correctly pass? |
| **F1** | 2·P·R/(P+R) | Harmonic mean of precision and recall |
| **Fβ** | weighted harmonic mean | β>1 favors recall, β<1 favors precision |
| **ROC-AUC** | area under TPR vs FPR | Ranking quality across all thresholds |
| **PR-AUC / Average Precision** | area under precision-recall | Ranking quality on the positive class |
| **Log loss** | −mean(y·log p + (1−y)·log(1−p)) | Quality of probability estimates |
| **Brier score** | mean((p − y)²) | Calibration + accuracy of probabilities |
| **MCC** | correlation of predictions and truth | Balanced single number, good for imbalance |
| **Cohen's kappa** | agreement above chance | Multi-class agreement |

### Precision vs recall: pick based on cost

- **Spam filter** → precision matters. A real email in the spam folder is worse than one spam in the inbox.
- **Cancer screening** → recall matters. A missed cancer is catastrophic; a false alarm means one more test.
- **Fraud detection** → depends on review capacity. If analysts can review 100 cases/day, you want precision@100.

### ROC-AUC vs PR-AUC

ROC-AUC is insensitive to class imbalance because FPR uses the (huge) negative class as a denominator. With 1% positives, a model can have ROC-AUC 0.95 and still be useless in practice. **When positives are rare and are the class you care about, report PR-AUC (average precision).**

```python
from sklearn.metrics import (roc_curve, precision_recall_curve, auc,
                             roc_auc_score, average_precision_score,
                             RocCurveDisplay, PrecisionRecallDisplay,
                             ConfusionMatrixDisplay, classification_report)

y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, digits=3))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("PR-AUC :", average_precision_score(y_test, y_proba))

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax[0])
PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax[1])
plt.show()
```

### The threshold is a business decision, not a default

`predict()` uses 0.5. That is almost never optimal.

```python
prec, rec, thr = precision_recall_curve(y_test, y_proba)
f1 = 2 * prec * rec / (prec + rec + 1e-12)
best_idx = np.nanargmax(f1[:-1])
print(f"Best F1 {f1[best_idx]:.3f} at threshold {thr[best_idx]:.3f}")

# Or optimize expected value directly
def expected_profit(y_true, y_proba, threshold,
                    value_tp=100, cost_fp=-10, cost_fn=-50, value_tn=0):
    pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return tp*value_tp + fp*cost_fp + fn*cost_fn + tn*value_tn

ths = np.linspace(0.01, 0.99, 99)
profits = [expected_profit(y_test, y_proba, t) for t in ths]
plt.plot(ths, profits); plt.xlabel("Threshold"); plt.ylabel("Expected profit")
print("Optimal threshold:", ths[int(np.argmax(profits))])
```

This — converting a model score into a decision rule with an explicit cost matrix — is the step that separates a notebook from a product.

### Calibration

A model can rank well (high AUC) but output badly-scaled probabilities. If you need "70% means it happens 70% of the time," check calibration.

```python
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV

CalibrationDisplay.from_predictions(y_test, y_proba, n_bins=10)
plt.show()

calibrated = CalibratedClassifierCV(model, method="isotonic", cv=5)
# method="sigmoid" (Platt scaling) for small datasets; "isotonic" needs more data
```

Random forests are typically under-confident at the extremes; boosted trees and SVMs need calibration too. Logistic regression is usually well-calibrated by construction.

### Multi-class

```python
# averaging strategies
f1_score(y_test, y_pred, average="macro")     # unweighted mean over classes
f1_score(y_test, y_pred, average="weighted")  # weighted by support
f1_score(y_test, y_pred, average="micro")     # global TP/FP/FN (= accuracy for single-label)
roc_auc_score(y_test, y_proba_matrix, multi_class="ovr", average="macro")
```

Use **macro** when all classes matter equally (even the rare ones). Use **weighted** when you care proportionally to frequency.

**Exercise 11.1** — Build a classifier on a dataset with 5% positives. Report accuracy, ROC-AUC, and PR-AUC for (a) your model and (b) a model that predicts random probabilities. Show how each metric behaves and explain which is honest.

**Exercise 11.2** — Write `optimal_threshold(y_true, y_proba, cost_fp, cost_fn)` returning the threshold minimizing total cost. Plot the cost curve.

---

## 12. Bias, variance, and diagnosis

**Total error = bias² + variance + irreducible noise**

- **Bias** — error from wrong assumptions. The model is too simple. Symptom: bad on train *and* test. → *underfitting*.
- **Variance** — error from sensitivity to the specific training sample. Symptom: great on train, bad on test. → *overfitting*.

| Symptom | Diagnosis | Fixes |
|---|---|---|
| Train error high, test ≈ train | Underfitting (high bias) | More complex model, more/better features, less regularization, train longer |
| Train error low, test error high | Overfitting (high variance) | More data, simpler model, more regularization, feature selection, bagging, early stopping, dropout |
| Both low | You're done | Ship it |
| Test error > train but both acceptable | Normal | Fine |

### Learning curves — do you need more data?

```python
from sklearn.model_selection import learning_curve, validation_curve

sizes, train_scores, val_scores = learning_curve(
    pipe, X_train, y_train, cv=5, scoring="roc_auc",
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)

plt.plot(sizes, train_scores.mean(1), "o-", label="train")
plt.plot(sizes, val_scores.mean(1),   "o-", label="validation")
plt.fill_between(sizes, val_scores.mean(1)-val_scores.std(1),
                 val_scores.mean(1)+val_scores.std(1), alpha=0.2)
plt.xlabel("Training samples"); plt.ylabel("ROC-AUC"); plt.legend(); plt.show()
```

Reading it:
- Curves converged at a **low** score → high bias. More data won't help. Get a better model or better features.
- A **large persistent gap** → high variance. More data *will* help.
- Validation still rising at the right edge → collect more data.

### Validation curves — how complex should the model be?

```python
depths = [1, 2, 3, 5, 8, 12, 20, None]
train_s, val_s = validation_curve(
    RandomForestClassifier(random_state=0), X_train, y_train,
    param_name="max_depth", param_range=[d if d else 50 for d in depths],
    cv=5, scoring="roc_auc", n_jobs=-1)
```

Plot both; the sweet spot is where validation peaks, just before it turns down.

---

# PART IV — SUPERVISED ALGORITHMS

For each algorithm below: the intuition, the math that matters, code, hyperparameters that actually move the needle, strengths, weaknesses.

## 13. Linear regression and regularization

### Intuition

Fit a straight line (or hyperplane) through the data: **ŷ = w₀ + w₁x₁ + w₂x₂ + … + wₙxₙ**

Find the weights **w** that minimize squared error: **L = Σ(yᵢ − ŷᵢ)²**

There's a closed-form solution (the normal equation, **w = (XᵀX)⁻¹Xᵀy**), but in practice sklearn uses a numerically stable least-squares solver, and for large data you'd use gradient descent.

### Regularization: the antidote to overfitting

Add a penalty on the size of the weights:

| Variant | Penalty added to loss | Effect |
|---|---|---|
| **Ridge (L2)** | α·Σwⱼ² | Shrinks all coefficients toward zero, never exactly zero. Handles multicollinearity. |
| **Lasso (L1)** | α·Σ\|wⱼ\| | Drives some coefficients to *exactly* zero → automatic feature selection. |
| **ElasticNet** | α·(ρ·Σ\|wⱼ\| + (1−ρ)/2·Σwⱼ²) | Blend of both. Good with many correlated features. |

Larger α = more regularization = simpler model = more bias, less variance.

```python
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
                                  RidgeCV, LassoCV, ElasticNetCV,
                                  HuberRegressor, QuantileRegressor)
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

models = {
    "OLS":        LinearRegression(),
    "Ridge":      RidgeCV(alphas=np.logspace(-3, 3, 30)),
    "Lasso":      LassoCV(alphas=np.logspace(-3, 1, 30), max_iter=5000),
    "ElasticNet": ElasticNetCV(l1_ratio=[.1,.5,.7,.9,.95,1], max_iter=5000),
}

for name, m in models.items():
    pipe = make_pipeline(StandardScaler(), m)
    pipe.fit(X_train, y_train)
    print(f"{name:11s} R2={pipe.score(X_test, y_test):.4f}")
```

**Scaling is mandatory with regularization** — the penalty treats all coefficients equally, so a feature measured in rupees and one in millions get penalized incomprehensibly differently otherwise.

### Reading the coefficients

```python
pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_train, y_train)
coefs = pd.Series(pipe[-1].coef_, index=X.columns).sort_values()
coefs.plot.barh(title="Ridge coefficients (standardized features)")
plt.show()
```

Because features are standardized, coefficients are directly comparable: "a one-standard-deviation increase in median income raises predicted price by 0.85 units, holding others constant." That interpretability is linear regression's main selling point.

### Robust variants

```python
HuberRegressor(epsilon=1.35)     # less sensitive to outliers than squared loss
QuantileRegressor(quantile=0.9)  # predict the 90th percentile, not the mean
```

Quantile regression is how you build prediction *intervals*: fit models at 0.05, 0.5, 0.95.

### Assumptions (matter for inference, less for prediction)

Linearity, independent errors, constant error variance (homoscedasticity), normally distributed errors, no perfect multicollinearity. If you only care about predictive accuracy, violations mostly cost you efficiency. If you're quoting p-values and confidence intervals, use `statsmodels` and check the assumptions properly.

```python
import statsmodels.api as sm
Xc = sm.add_constant(X_train)
print(sm.OLS(y_train, Xc).fit().summary())   # p-values, CIs, R², F-stat, Durbin-Watson
```

**Strengths:** fast, interpretable, well-understood, strong baseline, extrapolates sensibly.
**Weaknesses:** only linear relationships (unless you engineer them), sensitive to outliers, struggles with high-dimensional correlated features without regularization.

**Exercise 13.1** — On California housing, plot Lasso coefficient paths as α varies from 1e-4 to 10 (`lasso_path`). At what α does each feature drop out? What does the drop-out order tell you?

---

## 14. Logistic regression

Despite the name, this is a **classification** algorithm. It models the probability of the positive class:

**p = σ(w·x + b)** where **σ(z) = 1/(1 + e⁻ᶻ)** (the sigmoid)

The sigmoid squashes any real number into (0,1). Training minimizes **log loss** (cross-entropy):

**L = −Σ [ yᵢ·log(pᵢ) + (1−yᵢ)·log(1−pᵢ) ]**

The coefficients are **log-odds**: `exp(wⱼ)` is the odds ratio for a one-unit increase in feature j.

```python
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

clf = LogisticRegression(
    penalty="l2",        # 'l1', 'l2', 'elasticnet', None
    C=1.0,               # INVERSE regularization — smaller C = stronger penalty
    solver="lbfgs",      # 'liblinear' (small data, l1), 'saga' (large, all penalties)
    class_weight=None,   # 'balanced' for imbalanced targets
    max_iter=1000,
)
pipe = make_pipeline(StandardScaler(), clf).fit(X_train, y_train)

# Interpret as odds ratios
odds = pd.Series(np.exp(pipe[-1].coef_[0]), index=X.columns).sort_values()
print(odds)   # >1 increases odds of positive class, <1 decreases
```

Note the trap: in Ridge/Lasso the knob is `alpha` (bigger = more regularization), in LogisticRegression and SVM it's `C = 1/alpha` (bigger = *less* regularization).

**Strengths:** fast, calibrated probabilities out of the box, interpretable, works well when the decision boundary is roughly linear, hard to overfit with regularization.
**Weaknesses:** can't capture interactions or nonlinearity unless you engineer them.

**Exercise 14.1** — On the Titanic pipeline, extract the odds ratios. Which feature has the largest effect? Verify it against a simple groupby of the raw data.

---

## 15. k-Nearest Neighbors

No training at all — just store the data. To predict, find the k closest training points and take a majority vote (classification) or average (regression).

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

knn = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",   # 'uniform' or 'distance' (closer points count more)
    metric="minkowski", p=2,   # p=2 Euclidean, p=1 Manhattan
    n_jobs=-1,
)
pipe = make_pipeline(StandardScaler(), knn)   # scaling is NOT optional here
```

Choosing k: small k = low bias, high variance (jagged boundary, sensitive to noise). Large k = smoother, higher bias. Tune it; a common heuristic starting point is √n.

**The curse of dimensionality:** in high dimensions, all points become roughly equidistant and "nearest neighbor" loses meaning. KNN degrades badly beyond ~20 informative features. Reduce dimensions (PCA) first, or use something else.

**Strengths:** zero assumptions about the data shape, naturally handles multi-class, trivially simple, good for recommender-style similarity.
**Weaknesses:** slow at prediction time (O(n) per query), memory-hungry, needs scaling, dies in high dimensions, no interpretability.

---

## 16. Naive Bayes

Applies Bayes' theorem with a strong ("naive") assumption: all features are conditionally independent given the class.

**P(class | features) ∝ P(class) · Π P(featureⱼ | class)**

The assumption is essentially always false, and the algorithm works surprisingly well anyway — especially for text.

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB

GaussianNB()      # continuous features, assumes each is normally distributed per class
MultinomialNB()   # counts (word frequencies) — the classic text classifier
BernoulliNB()     # binary features (word present/absent)
ComplementNB()    # MultinomialNB variant that handles imbalanced text better
```

Text example:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

cats = ["sci.space", "rec.sport.hockey", "talk.politics.mideast"]
train = fetch_20newsgroups(subset="train", categories=cats,
                           remove=("headers", "footers", "quotes"))
test  = fetch_20newsgroups(subset="test", categories=cats,
                           remove=("headers", "footers", "quotes"))

pipe = make_pipeline(TfidfVectorizer(stop_words="english", min_df=2),
                     MultinomialNB(alpha=0.1))
pipe.fit(train.data, train.target)
print(pipe.score(test.data, test.target))
```

**Strengths:** extremely fast, works with tiny datasets, handles very high dimensions (text), naturally multi-class, an excellent baseline.
**Weaknesses:** independence assumption breaks calibration (probabilities are over-confident), can't learn feature interactions.

---

## 17. Support Vector Machines

Find the hyperplane that separates classes with the **largest margin** — the widest possible street between them. Only the points on the edge of the street (the *support vectors*) determine the boundary.

For non-linearly-separable data, the **kernel trick** implicitly maps features into a higher-dimensional space where a linear separator exists, without ever computing the mapping.

```python
from sklearn.svm import SVC, SVR, LinearSVC

svm = SVC(
    C=1.0,            # regularization: low C = wide margin, more misclassification allowed
    kernel="rbf",     # 'linear', 'poly', 'rbf', 'sigmoid'
    gamma="scale",    # RBF width: high gamma = tight, wiggly boundary (overfits)
    probability=True, # needed for predict_proba — slow (internal CV), avoid if not needed
    class_weight="balanced",
)
pipe = make_pipeline(StandardScaler(), svm)   # scaling essential
```

The C/gamma interaction is the whole game with RBF SVMs:

| | Low gamma | High gamma |
|---|---|---|
| **Low C** | Very smooth, likely underfit | Smooth-ish |
| **High C** | Fits training data with a smooth boundary | Very wiggly, overfits |

Search them jointly on a log grid: `C ∈ [0.1, 1, 10, 100]`, `gamma ∈ [1e-4, 1e-3, 1e-2, 1e-1]`.

**Strengths:** effective in high dimensions, memory-efficient (stores only support vectors), flexible via kernels, strong on small-to-medium clean datasets.
**Weaknesses:** scales roughly O(n²)–O(n³) — impractical beyond ~50k rows; no native probabilities; needs careful tuning and scaling; a black box.

For large linear problems use `LinearSVC` or `SGDClassifier(loss="hinge")`, which scale linearly.

---

## 18. Decision Trees

Repeatedly split the data on the feature/threshold that best separates the target, forming a tree of if/else rules.

Split quality is measured by impurity reduction:
- **Gini impurity** = 1 − Σpᵢ² (default, slightly faster)
- **Entropy** = −Σpᵢ·log₂pᵢ (information gain)
- **MSE / MAE** for regression trees

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text

tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,               # THE main overfitting control
    min_samples_split=20,      # don't split nodes smaller than this
    min_samples_leaf=10,       # every leaf must have at least this many samples
    max_features=None,         # features considered per split
    ccp_alpha=0.0,             # cost-complexity pruning strength
    class_weight="balanced",
    random_state=42,
)
tree.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=X.columns, class_names=["no","yes"],
          filled=True, max_depth=3, fontsize=9)
plt.show()

print(export_text(tree, feature_names=list(X.columns), max_depth=3))
```

### Cost-complexity pruning (the principled way to size a tree)

```python
path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas[:-1]
scores = [cross_val_score(DecisionTreeClassifier(ccp_alpha=a, random_state=0),
                          X_train, y_train, cv=5).mean() for a in alphas]
best_alpha = alphas[int(np.argmax(scores))]
```

**Strengths:** fully interpretable (you can print the rules), no scaling needed, handles mixed data types, captures interactions and nonlinearity automatically, fast.
**Weaknesses:** wildly unstable — change a few rows and the tree changes completely; overfits badly if unconstrained; predicts piecewise constants so it cannot extrapolate beyond the training range.

That instability is precisely what ensembles exploit.

---

## 19. Random Forests

Train many decision trees, each on a bootstrap sample of the rows (**bagging**) and considering only a random subset of features at each split (**feature bagging**). Average their predictions.

Why it works: individual trees are high-variance but roughly unbiased. Averaging many *decorrelated* high-variance estimators reduces variance without adding bias. The two sources of randomness (row sampling, feature sampling) are what decorrelate them.

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier

rf = RandomForestClassifier(
    n_estimators=500,        # more is always better for accuracy, just slower. 300–1000.
    max_depth=None,          # usually leave unlimited; control via min_samples_leaf
    min_samples_leaf=1,      # raise to 3–10 for noisy data
    max_features="sqrt",     # 'sqrt' for classification, 1.0 or 0.3 for regression
    bootstrap=True,
    oob_score=True,          # free validation estimate from out-of-bag samples
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)
print("OOB score:", rf.oob_score_)
```

**Out-of-bag scoring** is a free lunch: each tree is trained on ~63% of rows, so the other ~37% act as a validation set for that tree. You get a validation estimate without a separate split.

**ExtraTrees** (Extremely Randomized Trees) go further — split thresholds are chosen at random rather than optimally. Faster, more variance reduction, slightly more bias. Worth trying alongside RF.

### Feature importance — two kinds, one of them misleading

```python
# 1. Impurity-based (built in) — FAST but BIASED toward high-cardinality
#    and continuous features. Computed on training data.
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

# 2. Permutation importance — shuffle a column, measure the score drop.
#    Slower, computed on held-out data, model-agnostic, much more trustworthy.
from sklearn.inspection import permutation_importance
r = permutation_importance(rf, X_test, y_test, n_repeats=20,
                           random_state=0, n_jobs=-1, scoring="roc_auc")
perm = pd.Series(r.importances_mean, index=X.columns).sort_values()
perm.plot.barh(xerr=r.importances_std); plt.show()
```

Always prefer permutation importance for reporting. Note that with correlated features both methods split credit arbitrarily — if two features are duplicates, each may show near-zero importance because the other covers for it.

**Strengths:** excellent out-of-the-box performance, very hard to overfit by adding trees, no scaling, handles missing-ish data and mixed types, parallelizes perfectly, gives OOB validation and importances free.
**Weaknesses:** large memory footprint, slower inference than a single tree, less accurate than tuned boosting on most tabular problems, can't extrapolate.

---

## 20. Gradient Boosting

The most important family for tabular data. Where random forests build trees *independently and average*, boosting builds them **sequentially, each one correcting the errors of the ensemble so far.**

The algorithm, in words:
1. Start with a constant prediction (e.g. the mean).
2. Compute the residuals (or more generally, the gradient of the loss).
3. Fit a small tree to predict those residuals.
4. Add that tree's predictions to the ensemble, scaled by a learning rate η.
5. Repeat.

Each tree is weak (depth 3–8), but hundreds of them, each nudging in the right direction, compose into a very strong model. The learning rate controls how big each nudge is: smaller η needs more trees but generalizes better.

### The three libraries

| | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| **Tree growth** | Level-wise (depth-wise) | Leaf-wise (best-first) | Symmetric (oblivious) trees |
| **Speed** | Fast | Fastest, especially wide data | Moderate |
| **Categorical features** | Needs encoding (or `enable_categorical`) | Native (`categorical_feature`) | Native, best-in-class (ordered target statistics) |
| **Default quality** | Needs tuning | Needs tuning | Excellent defaults |
| **Overfitting risk** | Moderate | Higher (leaf-wise) — control `num_leaves` | Lowest |
| **Best for** | General purpose, well-documented | Large datasets, many features | Many categorical features, minimal tuning |

```python
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# ---------- XGBoost ----------
model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=1,        # min sum of instance weight in a child
    subsample=0.8,             # row sampling per tree
    colsample_bytree=0.8,      # feature sampling per tree
    reg_alpha=0.0,             # L1
    reg_lambda=1.0,            # L2
    gamma=0.0,                 # min loss reduction to split
    scale_pos_weight=1,        # for imbalance: neg_count/pos_count
    eval_metric="auc",
    early_stopping_rounds=100,
    tree_method="hist",        # fast histogram algorithm
    n_jobs=-1, random_state=42,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
print("Best iteration:", model.best_iteration)

# ---------- LightGBM ----------
model = lgb.LGBMClassifier(
    n_estimators=2000, learning_rate=0.05,
    num_leaves=31,             # THE key LGBM knob; keep < 2^max_depth
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.0, reg_lambda=0.0,
    n_jobs=-1, random_state=42,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
          eval_metric="auc",
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])

# ---------- CatBoost ----------
model = CatBoostClassifier(
    iterations=2000, learning_rate=0.05, depth=6,
    l2_leaf_reg=3.0, loss_function="Logloss", eval_metric="AUC",
    cat_features=["sex", "embarked"],   # pass raw strings, no encoding needed
    early_stopping_rounds=100, verbose=200, random_seed=42,
)
model.fit(X_train, y_train, eval_set=(X_val, y_val))
```

### Early stopping is not optional

Boosting will overfit if you let it run forever. Always hold out a validation set and stop when validation loss stops improving. Then either use `best_iteration` directly, or refit on train+val for that many rounds.

### A tuning order that works

1. Fix `learning_rate = 0.1` and find a rough `n_estimators` with early stopping.
2. Tune tree structure: `max_depth` / `num_leaves`, `min_child_weight` / `min_child_samples`.
3. Tune sampling: `subsample`, `colsample_bytree` (0.6–1.0).
4. Tune regularization: `reg_alpha`, `reg_lambda`, `gamma`.
5. Lower `learning_rate` to 0.01–0.03 and raise `n_estimators` proportionally. Re-run early stopping.

`sklearn`'s own `HistGradientBoostingClassifier` is a solid LightGBM-alike with no extra dependency and native NaN handling:

```python
from sklearn.ensemble import HistGradientBoostingClassifier
HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                               max_leaf_nodes=31, early_stopping=True)
```

**Strengths:** state-of-the-art on tabular data, handles mixed types, built-in missing-value handling, gives importances, flexible loss functions.
**Weaknesses:** many hyperparameters, sequential (less parallel than RF), sensitive to noisy labels, will overfit without early stopping.

---

## 21. Algorithm cheat sheet

| Situation | Start with |
|---|---|
| Tabular data, any size | **LightGBM / XGBoost / CatBoost** |
| Need interpretability for regulators | Logistic/linear regression, or a shallow tree, or GBM + SHAP |
| < 1,000 rows | Regularized linear model, Naive Bayes, or SVM |
| Text classification | TF-IDF + LinearSVC/LogReg baseline → fine-tuned transformer |
| Images | CNN, or transfer learning from a pretrained backbone |
| Time series forecasting | Baselines → ETS/ARIMA → GBM on lag features → deep models |
| Very high dimensions, sparse | Linear models with L1/L2, Naive Bayes |
| Need probability calibration | Logistic regression, or calibrate anything else |
| Need speed at inference | Linear model, or a small GBM |
| Nobody knows what's in the data | RandomForest as a diagnostic, then look at importances |

**Suggested default workflow for a new tabular problem:**

```python
# 1. Dummy baseline
# 2. LogisticRegression / Ridge  (fast, interpretable, catches leakage — if this
#    scores 0.99 you have a bug, not a model)
# 3. RandomForest  (robust, no tuning)
# 4. LightGBM with early stopping  (usually your winner)
# 5. Tune the winner with Optuna
# 6. Maybe blend 3 and 4
```


---

# PART V — GETTING MORE OUT OF MODELS

## 22. Hyperparameter tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV
from scipy.stats import uniform, randint, loguniform

# --- Grid search: exhaustive. Only for small spaces. ---
grid = {"model__max_depth": [3, 5, 7], "model__learning_rate": [0.01, 0.05, 0.1]}
gs = GridSearchCV(pipe, grid, cv=5, scoring="roc_auc", n_jobs=-1, refit=True)

# --- Random search: usually better per unit of compute ---
dist = {
    "model__n_estimators":     randint(200, 1500),
    "model__max_depth":        randint(3, 10),
    "model__learning_rate":    loguniform(1e-3, 3e-1),
    "model__subsample":        uniform(0.6, 0.4),   # loc=0.6, scale=0.4 → [0.6, 1.0]
    "model__colsample_bytree": uniform(0.6, 0.4),
    "model__reg_lambda":       loguniform(1e-3, 1e2),
}
rs = RandomizedSearchCV(pipe, dist, n_iter=60, cv=5, scoring="roc_auc",
                        n_jobs=-1, random_state=42, verbose=1)
rs.fit(X_train, y_train)
print(rs.best_params_, rs.best_score_)

results = pd.DataFrame(rs.cv_results_).sort_values("rank_test_score")
print(results[["params", "mean_test_score", "std_test_score"]].head())
```

Random search beats grid search when only a few hyperparameters matter (which is usually). Grid search wastes all its budget re-testing the same values of the important parameter.

### Bayesian optimization with Optuna

Optuna models the relationship between hyperparameters and score, and samples where improvement is likely. Typically finds better configs in 3–5× fewer trials.

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples":trial.suggest_int("min_child_samples", 5, 100),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
    }
    model = lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
    score = cross_val_score(model, X_train, y_train, cv=5,
                            scoring="roc_auc", n_jobs=1).mean()
    return score

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=42),
                            pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(study.best_params, study.best_value)
optuna.visualization.plot_param_importances(study).show()
optuna.visualization.plot_optimization_history(study).show()
```

`plot_param_importances` is genuinely useful — it tells you which hyperparameters actually mattered, so you can narrow the search next time.

### Nested cross-validation (when you need an unbiased estimate)

If you tune with CV and then report that CV score, it's optimistically biased — you selected on it. Nested CV gives an honest number:

```python
from sklearn.model_selection import cross_val_score, KFold
inner = KFold(5, shuffle=True, random_state=1)
outer = KFold(5, shuffle=True, random_state=2)
search = RandomizedSearchCV(pipe, dist, n_iter=30, cv=inner, scoring="roc_auc")
nested_scores = cross_val_score(search, X_train, y_train, cv=outer, scoring="roc_auc")
print(f"Unbiased estimate: {nested_scores.mean():.3f} ± {nested_scores.std():.3f}")
```

Expensive (n_inner × n_outer fits), but the right thing to do for small datasets and published results.

### Tuning discipline

- Tune on **validation/CV**, never on test.
- Reduce the search space with reasoning before brute force.
- If the best config sits at the edge of your range, extend the range.
- A 0.002 AUC improvement from tuning is noise. Check the CV standard deviation before believing it.
- Feature engineering usually buys more than tuning. Spend your time there first.

---

## 23. Imbalanced data

When 1% of rows are positive, a model that always predicts negative gets 99% accuracy and zero value.

### The hierarchy of fixes (try in this order)

**1. Change the metric.** Often the "imbalance problem" is really a "wrong metric" problem. Use PR-AUC, F1, recall@k, or expected business value.

**2. Move the threshold.** Your model probably ranks fine; 0.5 is just the wrong cutoff. This is free and should always be tried first.

**3. Class weights.** Tell the loss function that minority errors cost more. No data duplication, no distortion of the underlying distribution.

```python
LogisticRegression(class_weight="balanced")
RandomForestClassifier(class_weight="balanced_subsample")
xgb.XGBClassifier(scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum())
lgb.LGBMClassifier(class_weight="balanced")
```

**4. Resampling.** Only if the above are insufficient.

```python
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline   # NOTE: imblearn's Pipeline

pipe = ImbPipeline([
    ("prep",   preprocess),
    ("sample", SMOTE(random_state=42, k_neighbors=5)),
    ("model",  LGBMClassifier()),
])
```

**Critical:** resampling must happen **inside** the CV fold, applied to training data only. Never oversample before splitting — synthetic copies of a training point landing in the test set makes your score meaningless. `imblearn.pipeline.Pipeline` handles this correctly; sklearn's does not (it would apply the sampler to test data too).

| Technique | What it does | Risk |
|---|---|---|
| **Random oversampling** | Duplicates minority rows | Overfits to duplicated points |
| **SMOTE** | Creates synthetic minority points by interpolating between neighbors | Can create nonsense points in overlapping regions; bad with categorical features |
| **ADASYN** | SMOTE that focuses on hard-to-learn regions | Amplifies noise |
| **Random undersampling** | Drops majority rows | Throws away information |
| **TomekLinks / ENN** | Cleans boundary-overlapping majority points | Mild effect |

Honest assessment: on modern gradient boosting, class weights + threshold tuning usually match or beat SMOTE, with less complexity. Try the simple thing first.

**5. Reframe as anomaly detection** if the minority class is under ~0.1% — see section 28.

```python
# Always evaluate imbalanced problems with the right metrics
from sklearn.metrics import average_precision_score, recall_score, precision_score
print("PR-AUC:", average_precision_score(y_test, y_proba))

# Precision at fixed budget: "we can only review 500 cases"
k = 500
top_k = np.argsort(y_proba)[-k:]
print(f"Precision@{k}: {y_test.iloc[top_k].mean():.3f}")
```

**Exercise 23.1** — On a 1% positive dataset, compare four approaches (baseline, class_weight, SMOTE, threshold-tuned) on PR-AUC and precision@100. Which wins? Was SMOTE worth it?

---

## 24. Model interpretation

You need this for three reasons: debugging (is the model using a leaky feature?), trust (will stakeholders adopt it?), and compliance (can you explain a denied loan?).

### Global: what matters overall

```python
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# Permutation importance — the reliable default
r = permutation_importance(model, X_test, y_test, n_repeats=30,
                           random_state=0, scoring="roc_auc", n_jobs=-1)
pd.Series(r.importances_mean, index=X_test.columns).sort_values().plot.barh()

# Partial dependence — how does the prediction change with feature value?
PartialDependenceDisplay.from_estimator(
    model, X_test, features=["age", "fare", ("age", "fare")], kind="both")
# kind="both" overlays ICE curves (one line per individual) on the average PDP
plt.show()
```

PDP shows the average effect; **ICE** curves show individual effects. If ICE curves fan out in different directions, there's an interaction the PDP average is hiding.

Caveat: PDPs assume feature independence. If age and income are correlated, the PDP evaluates unrealistic combinations (age 20, income ₹5 crore). ALE plots (`alibi`, `PyALE`) fix this.

### Local: why this prediction?

**SHAP** (SHapley Additive exPlanations) assigns each feature a contribution to a specific prediction, based on cooperative game theory. It's the current standard, with strong theoretical guarantees (contributions sum exactly to the prediction minus the base value).

```python
import shap

explainer = shap.TreeExplainer(model)        # fast, exact for tree models
shap_values = explainer(X_test)

shap.plots.bar(shap_values)                  # global importance
shap.plots.beeswarm(shap_values)             # global + direction + value
shap.plots.waterfall(shap_values[0])         # one prediction explained
shap.plots.scatter(shap_values[:, "age"], color=shap_values)  # dependence + interaction
shap.plots.force(shap_values[0])             # compact single-prediction view

# For non-tree models:
explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
explainer = shap.LinearExplainer(linear_model, X_train)
explainer = shap.DeepExplainer(torch_model, background)
```

The **beeswarm plot** is the single most informative ML chart. Each dot is one prediction; x-position is that feature's SHAP contribution; color is the feature's value. You see importance, direction, and nonlinearity at once.

### Surrogate models

Fit an interpretable model to mimic your black box:

```python
surrogate = DecisionTreeClassifier(max_depth=4).fit(X_train, model.predict(X_train))
print("Fidelity:", surrogate.score(X_train, model.predict(X_train)))
print(export_text(surrogate, feature_names=list(X_train.columns)))
```

Only trustworthy if fidelity is high (>0.9).

### Using interpretation to find bugs

- A feature with implausibly dominant importance → probable leakage.
- A PDP that jumps at exactly 0 → your missing-value sentinel is being read as a real value.
- SHAP shows the model relying on `customer_id` → your IDs encode time or segment. Remove it.

**Exercise 24.1** — Train a GBM on any dataset, produce a SHAP beeswarm, and write three sentences a non-technical manager would understand about what drives predictions.

---

## 25. Ensembling and stacking

Combining diverse models nearly always beats any single one — the errors partially cancel. Diversity matters more than individual quality.

```python
from sklearn.ensemble import VotingClassifier, StackingClassifier

# --- Voting: average predictions ---
vote = VotingClassifier(
    estimators=[("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
                ("rf", RandomForestClassifier(n_estimators=500, random_state=0)),
                ("gb", lgb.LGBMClassifier(random_state=0, verbose=-1))],
    voting="soft",              # average probabilities (better than 'hard' voting)
    weights=[1, 2, 3],
)

# --- Stacking: a meta-model learns how to combine base predictions ---
stack = StackingClassifier(
    estimators=[("rf", RandomForestClassifier(n_estimators=500, random_state=0)),
                ("gb", lgb.LGBMClassifier(random_state=0, verbose=-1)),
                ("svm", make_pipeline(StandardScaler(), SVC(probability=True)))],
    final_estimator=LogisticRegression(),
    cv=5,                       # base predictions generated out-of-fold — no leakage
    passthrough=False,          # True also feeds original features to the meta-model
    n_jobs=-1,
)
```

Simple blending often works as well as stacking and is easier to reason about:

```python
final_proba = 0.5*gbm_proba + 0.3*rf_proba + 0.2*lr_proba
# Find weights by optimizing CV score, or just use rank averaging:
from scipy.stats import rankdata
final = 0.5*rankdata(gbm_proba) + 0.3*rankdata(rf_proba) + 0.2*rankdata(lr_proba)
```

Rank averaging is robust when models produce probabilities on different scales.

Costs to weigh: 3× the inference latency, 3× the things that can break in production, harder to explain. Ensemble when the accuracy gain justifies the operational complexity — often it doesn't.

---

# PART VI — UNSUPERVISED LEARNING

## 26. Clustering

### K-Means

Partition data into k clusters, each represented by its centroid. Iteratively: assign each point to the nearest centroid, then recompute centroids. Minimizes within-cluster sum of squares (inertia).

```python
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, silhouette_samples

X_scaled = StandardScaler().fit_transform(X)

# Choosing k: elbow method + silhouette
inertias, sils = [], []
ks = range(2, 12)
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_scaled, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(ks, inertias, "o-"); ax[0].set(xlabel="k", ylabel="Inertia", title="Elbow")
ax[1].plot(ks, sils, "o-");     ax[1].set(xlabel="k", ylabel="Silhouette")
plt.show()
```

**Silhouette score** ranges from −1 to 1: how much closer a point is to its own cluster than to the next nearest. Above 0.5 is decent structure; near 0 means overlapping clusters; negative means points are in the wrong cluster.

K-Means assumptions (and their failure modes): clusters are spherical, similar in size, similar in density. Elongated or nested clusters break it completely. It also always returns exactly k clusters, even when there's no structure at all.

### DBSCAN — density-based

Groups points in dense regions; points in sparse regions become noise (label −1). No need to specify the number of clusters, and it finds arbitrary shapes.

```python
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import NearestNeighbors

# Pick eps from the k-distance elbow
nn = NearestNeighbors(n_neighbors=5).fit(X_scaled)
dist, _ = nn.kneighbors(X_scaled)
plt.plot(np.sort(dist[:, -1])); plt.ylabel("5th-NN distance"); plt.show()

db = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
print("Clusters:", len(set(db.labels_)) - (1 if -1 in db.labels_ else 0))
print("Noise points:", (db.labels_ == -1).sum())

# HDBSCAN: hierarchical DBSCAN, handles varying density, less eps-sensitive
hdb = HDBSCAN(min_cluster_size=15).fit(X_scaled)
```

### Hierarchical (agglomerative)

Start with every point its own cluster; repeatedly merge the closest pair. Produces a dendrogram you can cut at any level.

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(X_scaled, method="ward")   # 'ward','complete','average','single'
plt.figure(figsize=(14, 5))
dendrogram(Z, truncate_mode="lastp", p=30)
plt.show()

agg = AgglomerativeClustering(n_clusters=4, linkage="ward").fit(X_scaled)
```

### Comparison

| Algorithm | Need k? | Cluster shapes | Handles noise | Scales to |
|---|---|---|---|---|
| K-Means | Yes | Spherical, similar size | No | Millions (MiniBatch) |
| Hierarchical | No (cut later) | Any (depends on linkage) | No | ~10k (O(n²) memory) |
| DBSCAN | No | Arbitrary | Yes | ~100k |
| HDBSCAN | No | Arbitrary, varying density | Yes | ~100k |
| Gaussian Mixture | Yes | Elliptical, soft assignment | No | Large |

### Making clusters actionable

A cluster label is useless until you can describe it:

```python
df["cluster"] = km.labels_
profile = df.groupby("cluster").agg(["mean", "median", "count"])
# Compare each cluster's mean to the overall mean to find defining traits
z = (df.groupby("cluster").mean() - df.mean()) / df.std()
sns.heatmap(z.T, cmap="RdBu_r", center=0, annot=True, fmt=".1f")
plt.show()
```

The heatmap of standardized deviations instantly tells you "cluster 2 = high spend, low frequency, recent" — which you can then name and act on.

**Exercise 26.1** — Build RFM features from any transactional dataset, cluster with K-Means, and write a one-line business description of each segment. Then repeat with DBSCAN and compare which gives more actionable groups.

---

## 27. Dimensionality reduction

### PCA — linear, fast, reversible

Finds orthogonal directions of maximum variance. The first principal component captures the most variance, the second the most of what remains, and so on.

```python
from sklearn.decomposition import PCA, TruncatedSVD, NMF

pca = PCA(n_components=0.95)          # keep enough components for 95% of variance
X_pca = pca.fit_transform(X_scaled)   # SCALE FIRST — PCA is variance-based
print(f"{X.shape[1]} → {X_pca.shape[1]} components")

plt.plot(np.cumsum(pca.explained_variance_ratio_), "o-")
plt.axhline(0.95, c="r", ls="--")
plt.xlabel("Components"); plt.ylabel("Cumulative explained variance"); plt.show()

# What does each component mean?
loadings = pd.DataFrame(pca.components_[:3].T, index=X.columns,
                        columns=["PC1", "PC2", "PC3"])
print(loadings.round(2))
```

Uses: compression, noise reduction, decorrelating features for linear models, visualization, speeding up KNN/SVM.

`TruncatedSVD` is PCA for sparse matrices (use it on TF-IDF — this is LSA). `NMF` gives non-negative, parts-based, more interpretable components for text and images.

### t-SNE and UMAP — nonlinear, for visualization

```python
from sklearn.manifold import TSNE
import umap

emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(X_scaled)
emb = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2).fit_transform(X_scaled)

plt.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab10", s=5, alpha=0.6)
plt.colorbar(); plt.show()
```

**Warnings that matter:** in t-SNE, cluster *sizes* and *distances between clusters* are not meaningful — only local neighborhood structure is. Different perplexities produce genuinely different pictures. Never use t-SNE output as features for a downstream model (it has no `transform` for new data). UMAP is faster, preserves more global structure, and does have a `transform`, so it can be used in a pipeline — carefully.

Common practice: PCA down to ~50 dimensions first (denoise + speed), then t-SNE/UMAP to 2 for the plot.

---

## 28. Anomaly detection

For when anomalies are too rare or too varied to model as a supervised class.

```python
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

# Isolation Forest — randomly partition; anomalies get isolated in fewer splits.
# Best default: fast, handles high dimensions, few assumptions.
iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
pred = iso.fit_predict(X_scaled)          # -1 = anomaly, 1 = normal
scores = iso.score_samples(X_scaled)      # lower = more anomalous

# Local Outlier Factor — compares local density to neighbors'. Good for local anomalies.
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=False)

# One-Class SVM — learns a boundary around normal data. Slow, sensitive to gamma.
ocs = OneClassSVM(nu=0.01, kernel="rbf", gamma="scale")

# Elliptic Envelope — assumes Gaussian; fits a robust covariance ellipse.
ee = EllipticEnvelope(contamination=0.01)
```

Also very effective: **reconstruction error from an autoencoder** (or even PCA). Train to reconstruct normal data; anomalies reconstruct poorly.

```python
pca = PCA(n_components=10).fit(X_normal_scaled)
recon = pca.inverse_transform(pca.transform(X_scaled))
error = ((X_scaled - recon) ** 2).mean(axis=1)
threshold = np.percentile(error, 99)
anomalies = error > threshold
```

Evaluating anomaly detection without labels is genuinely hard. If you have *any* labels, use them to at least set the threshold. If not, sample the top anomalies and have a domain expert review them — that's a real evaluation, and it's how these systems get validated in practice.


---

# PART VII — SPECIALIZED DOMAINS

## 29. Time series

Time series breaks nearly every assumption of standard ML: rows are not independent, the future must never leak into the past, and the data-generating process itself changes.

### Rules

1. **Split chronologically.** Never shuffle. Train on the past, test on the future.
2. **All features must be backward-looking.** Rolling windows use only prior values.
3. **Validate with a rolling or expanding window**, matching how you'll actually forecast.
4. **Always compare against naive baselines.** For many series, "tomorrow = today" is shockingly hard to beat.

```python
# NAIVE BASELINES — compute these first, always
naive       = y.shift(1)                       # last value
seasonal    = y.shift(7)                       # same day last week
drift       = y.shift(1) + (y.iloc[-1] - y.iloc[0]) / (len(y) - 1)
mean_window = y.rolling(30).mean().shift(1)
```

### Feature engineering for time series

```python
def make_ts_features(df, col="value", lags=(1,2,3,7,14,28), windows=(7,14,30)):
    out = df.copy()
    # Lags
    for L in lags:
        out[f"lag_{L}"] = out[col].shift(L)
    # Rolling stats — .shift(1) FIRST so the current value is excluded
    s = out[col].shift(1)
    for w in windows:
        out[f"roll_mean_{w}"] = s.rolling(w).mean()
        out[f"roll_std_{w}"]  = s.rolling(w).std()
        out[f"roll_min_{w}"]  = s.rolling(w).min()
        out[f"roll_max_{w}"]  = s.rolling(w).max()
    # Expanding and differences
    out["expanding_mean"] = s.expanding().mean()
    out["diff_1"]  = out[col].diff(1).shift(1)
    out["pct_1"]   = out[col].pct_change(1).shift(1)
    # Exponentially weighted
    out["ewm_7"]   = s.ewm(span=7).mean()
    # Calendar
    idx = out.index
    out["dow"], out["month"], out["weekofyear"] = idx.dayofweek, idx.month, idx.isocalendar().week
    out["dow_sin"] = np.sin(2*np.pi*out["dow"]/7)
    out["dow_cos"] = np.cos(2*np.pi*out["dow"]/7)
    return out.dropna()
```

That `.shift(1)` before rolling is the single most important line. Without it, `roll_mean_7` at time t includes the value at t — you'd be using the answer to predict itself, and your backtest would look magnificent and your live model would fail.

### Decomposition and stationarity

```python
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

STL(y, period=7).fit().plot(); plt.show()      # trend / seasonal / residual

adf = adfuller(y.dropna())
print(f"ADF stat {adf[0]:.3f}, p-value {adf[1]:.4f}")
# p < 0.05 → stationary. If not, difference the series and re-test.

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(y.dropna(), lags=40, ax=ax[0])        # which lags matter → MA order
plot_pacf(y.dropna(), lags=40, ax=ax[1])       # direct effects  → AR order
plt.show()
```

### Classical models

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Holt-Winters: level + trend + seasonality. Fast, strong baseline.
hw = ExponentialSmoothing(y_train, trend="add", seasonal="add",
                          seasonal_periods=7).fit()
fc = hw.forecast(30)

# SARIMA(p,d,q)(P,D,Q,s)
sar = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,7)).fit(disp=False)
fc = sar.get_forecast(30)
print(fc.predicted_mean, fc.conf_int())
```

For automatic order selection use `pmdarima.auto_arima`. For business series with holidays and multiple seasonalities, `prophet` is a reasonable, low-effort choice.

### ML approach: turn it into a supervised problem

```python
feat = make_ts_features(df)
X_ts, y_ts = feat.drop(columns=["value"]), feat["value"]

tscv = TimeSeriesSplit(n_splits=5, test_size=30)
scores = []
for tr, te in tscv.split(X_ts):
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, verbose=-1)
    m.fit(X_ts.iloc[tr], y_ts.iloc[tr])
    p = m.predict(X_ts.iloc[te])
    scores.append(mean_absolute_error(y_ts.iloc[te], p))
print("MAE per fold:", np.round(scores, 3), "| mean:", np.mean(scores).round(3))
```

Note the limitation: gradient boosting cannot extrapolate a trend. If your series trends upward indefinitely, either detrend first (model the differences, or divide by a moving average) or use a model that handles trend explicitly.

### Multi-step forecasting

- **Recursive** — predict t+1, feed it back in, predict t+2. Simple, but errors compound.
- **Direct** — train a separate model for each horizon. No compounding, more models.
- **Multi-output** — one model predicting the whole horizon vector at once.

**Exercise 29.1** — Take any daily series. Build naive, seasonal-naive, Holt-Winters, and LightGBM forecasts. Evaluate with rolling-origin CV. If the naive baseline wins, explain why that's a legitimate result and not a failure.

---

## 30. Natural language processing

### Classical: bag of words → TF-IDF

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),     # unigrams + bigrams
    min_df=3,               # ignore terms in fewer than 3 documents
    max_df=0.9,             # ignore terms in >90% of documents
    max_features=50_000,
    sublinear_tf=True,      # use 1+log(tf) — usually helps
)
X_text = tfidf.fit_transform(train_texts)
```

TF-IDF weights a term by how often it appears in a document, divided by how common it is across the corpus. Common words get down-weighted; distinctive words get boosted.

A TF-IDF + LinearSVC baseline is competitive on many classification tasks and takes seconds to train. **Always build it before reaching for a transformer.**

```python
from sklearn.svm import LinearSVC
pipe = make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=2), LinearSVC(C=1.0))
pipe.fit(train_texts, train_labels)
```

### Word and sentence embeddings

Dense vectors where semantic similarity becomes geometric proximity.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")   # small, fast, good quality
emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                   normalize_embeddings=True)     # (n_texts, 384)

# Now it's just a feature matrix — use any classifier
clf = LogisticRegression(max_iter=1000).fit(emb_train, y_train)

# Or do semantic search
from sklearn.metrics.pairwise import cosine_similarity
sims = cosine_similarity(model.encode(["query text"]), emb)[0]
top = np.argsort(sims)[::-1][:5]
```

This "embed then classify" recipe is the best effort-to-quality ratio in modern NLP.

### Fine-tuning a transformer

```python
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from datasets import Dataset
import evaluate

ckpt = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)

ds = Dataset.from_dict({"text": texts, "label": labels}).train_test_split(0.2)
ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=256), batched=True)

metric = evaluate.load("f1")
def compute_metrics(p):
    return metric.compute(predictions=p.predictions.argmax(-1), references=p.label_ids)

args = TrainingArguments(
    output_dir="out", num_train_epochs=3,
    per_device_train_batch_size=16, learning_rate=2e-5,
    eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="f1",
    warmup_ratio=0.1, weight_decay=0.01, fp16=True,
)
Trainer(model=model, args=args, train_dataset=ds["train"],
        eval_dataset=ds["test"], compute_metrics=compute_metrics).train()
```

Typical fine-tuning hyperparameters: learning rate 2e-5 to 5e-5, 2–4 epochs, batch size 16–32. More epochs usually overfits.

### Text preprocessing notes

For classical models: lowercase, remove punctuation, maybe lemmatize (`spacy`), remove stop words. For transformers: **do almost nothing** — the tokenizer and pretrained model expect natural text, including casing and punctuation. Aggressive cleaning hurts.

**Exercise 30.1** — On 20 newsgroups, compare (a) TF-IDF + LinearSVC, (b) sentence embeddings + logistic regression, (c) fine-tuned DistilBERT. Record accuracy *and* training time. Which has the best accuracy per minute of compute?

---

## 31. Neural networks with PyTorch

Neural networks are compositions of linear transformations and nonlinearities, trained by gradient descent through backpropagation. Use them for images, audio, text, and sequences. For tabular data, gradient boosting usually wins — don't reach for a net just because it's fashionable.

### Core building blocks

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64), out_dim=1, p_drop=0.3):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

### The training loop (memorize this shape)

```python
def train_model(model, train_dl, val_dl, epochs=50, lr=1e-3, patience=7):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()          # logits in, no sigmoid in the model
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    best_loss, best_state, wait = float("inf"), None, 0
    for epoch in range(epochs):
        # --- train ---
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb).squeeze(), yb).item() * len(xb)

        train_loss /= len(train_dl.dataset); val_loss /= len(val_dl.dataset)
        scheduler.step(val_loss)
        print(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

        # --- early stopping ---
        if val_loss < best_loss:
            best_loss, best_state, wait = val_loss, model.state_dict(), 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    return model
```

### CNN for images

```python
class SimpleCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))
```

Convolutions exploit two facts about images: nearby pixels are related (locality) and a feature detector useful in one place is useful elsewhere (translation invariance). Pooling downsamples; batch norm stabilizes training; dropout regularizes.

### Transfer learning (what you'll actually do)

Almost never train an image model from scratch. Take a network pretrained on ImageNet and adapt it.

```python
import torchvision.models as models
from torchvision import transforms

model = models.resnet18(weights="IMAGENET1K_V1")
for p in model.parameters():
    p.requires_grad = False                  # freeze the backbone
model.fc = nn.Linear(model.fc.in_features, n_classes)   # new head, trainable

# Later: unfreeze the last block and fine-tune with a 10x smaller LR
for p in model.layer4.parameters():
    p.requires_grad = True

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

With a few hundred labeled images, transfer learning gets you 90%+ on many tasks. Training from scratch would need tens of thousands.

### Regularization toolkit

| Technique | What it does |
|---|---|
| **Dropout** | Randomly zeroes activations during training; prevents co-adaptation |
| **Weight decay (L2)** | Penalizes large weights |
| **Batch norm** | Normalizes layer inputs; stabilizes and mildly regularizes |
| **Early stopping** | Stop when validation stops improving |
| **Data augmentation** | Synthetically expand training data (the strongest one for images) |
| **Label smoothing** | Soften targets from 1.0 to 0.9; reduces over-confidence |
| **Gradient clipping** | Prevents exploding gradients in deep/recurrent nets |

### When training misbehaves

| Symptom | Likely cause |
|---|---|
| Loss = NaN | Learning rate too high, or log(0). Lower LR, clip gradients, check for division by zero |
| Loss flat from epoch 0 | LR too low, dead ReLUs, unnormalized inputs, or a bug in the loss |
| Train loss ↓, val loss ↑ | Overfitting. More augmentation/dropout/data, smaller model |
| Both losses stuck high | Underfitting. Bigger model, higher LR, train longer, better features |
| Wildly noisy loss curve | Batch size too small, or LR too high |

Sanity check that catches most bugs: **overfit a single batch.** If your model can't drive the loss to near zero on 32 examples, something is broken in the code, not the data.

---

## 32. Recommender systems

```python
# --- Collaborative filtering via matrix factorization ---
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

# user-item interaction matrix
R = csr_matrix((ratings["rating"],
                (ratings["user_idx"], ratings["item_idx"])))

svd = TruncatedSVD(n_components=50, random_state=42)
user_factors = svd.fit_transform(R)          # (n_users, 50)
item_factors = svd.components_.T             # (n_items, 50)
predicted = user_factors @ item_factors.T

# --- Item-item similarity (content-based / "more like this") ---
from sklearn.metrics.pairwise import cosine_similarity
item_sim = cosine_similarity(item_features)

def recommend(item_id, k=10):
    return np.argsort(item_sim[item_id])[::-1][1:k+1]
```

Key issues to handle: the **cold-start problem** (new users/items have no history — fall back to content features or popularity), **popularity bias** (naive models recommend only bestsellers — add diversity or novelty terms), and **evaluation** (offline metrics like precision@k, NDCG, MAP correlate imperfectly with online engagement; A/B test).

Libraries worth knowing: `implicit` (ALS/BPR for implicit feedback, fast), `surprise` (classic explicit-rating algorithms), `lightfm` (hybrid content + collaborative).

---

# PART VIII — SHIPPING IT

## 33. Persistence and serving

```python
import joblib

# Save the ENTIRE pipeline, not just the model
joblib.dump(pipe, "model_v1.joblib", compress=3)
loaded = joblib.load("model_v1.joblib")

# Always save metadata alongside it
import json, sklearn, datetime
meta = {
    "version": "1.0.0",
    "trained_at": datetime.datetime.now().isoformat(),
    "sklearn_version": sklearn.__version__,
    "features": list(X_train.columns),
    "metrics": {"test_roc_auc": 0.87, "test_pr_auc": 0.61},
    "training_rows": len(X_train),
    "threshold": 0.34,
}
json.dump(meta, open("model_v1_meta.json", "w"), indent=2)
```

Pickle/joblib artifacts are tied to library versions. Pin them in `requirements.txt`, or export to ONNX for a portable, version-independent artifact.

### A FastAPI service

```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib, pandas as pd, logging

app = FastAPI(title="Churn Model", version="1.0.0")
model = joblib.load("model_v1.joblib")
META = json.load(open("model_v1_meta.json"))

class Request(BaseModel):
    age: float = Field(..., ge=0, le=120)
    fare: float = Field(..., ge=0)
    sex: str
    pclass: int = Field(..., ge=1, le=3)

class Response(BaseModel):
    probability: float
    prediction: int
    model_version: str

@app.get("/health")
def health():
    return {"status": "ok", "version": META["version"]}

@app.post("/predict", response_model=Response)
def predict(req: Request):
    try:
        X = pd.DataFrame([req.model_dump()])
        proba = float(model.predict_proba(X)[0, 1])
        logging.info({"input": req.model_dump(), "proba": proba})
        return Response(probability=proba,
                        prediction=int(proba >= META["threshold"]),
                        model_version=META["version"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
def predict_batch(reqs: list[Request]):
    X = pd.DataFrame([r.model_dump() for r in reqs])
    return {"probabilities": model.predict_proba(X)[:, 1].tolist()}
```

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":29,"fare":80,"sex":"female","pclass":1}'
```

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Batch vs. real-time

Most production ML is **batch**: a nightly job scores every customer and writes to a table the application reads. It's simpler, cheaper, easier to monitor, and sufficient for churn, credit, propensity, and forecasting. Only build a real-time endpoint if the prediction depends on data that arrives at request time.

### Production checklist

- [ ] Input validation with explicit ranges and allowed categories
- [ ] Graceful handling of unseen categories and missing fields
- [ ] Log every input and output (you cannot debug or monitor without this)
- [ ] Health check endpoint
- [ ] Model version in every response
- [ ] Latency monitoring (p50, p95, p99)
- [ ] A fallback when the model errors (return the base rate, don't 500)
- [ ] Shadow mode before full rollout: run the model, log predictions, don't act on them
- [ ] A rollback plan

---

## 34. Monitoring, drift, and retraining

A deployed model decays. Reality shifts, upstream pipelines change, users adapt.

| Drift type | Definition | Detection |
|---|---|---|
| **Data drift** | P(X) changes — inputs look different | KS test, PSI, Jensen-Shannon distance per feature |
| **Concept drift** | P(y\|X) changes — the relationship itself changes | Performance drop on labeled data |
| **Label drift** | P(y) changes — base rate shifts | Track the positive rate over time |
| **Upstream bugs** | A column silently becomes all-null | Schema and null-rate checks |

```python
from scipy.stats import ks_2samp

def population_stability_index(expected, actual, bins=10):
    """PSI < 0.1 stable | 0.1-0.25 moderate shift | > 0.25 significant shift"""
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    e = np.histogram(expected, breakpoints)[0] / len(expected)
    a = np.histogram(actual,   breakpoints)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

for col in numeric_features:
    psi = population_stability_index(X_train[col], X_live[col])
    ks = ks_2samp(X_train[col], X_live[col])
    if psi > 0.25 or ks.pvalue < 0.01:
        print(f"⚠ DRIFT in {col}: PSI={psi:.3f}, KS p={ks.pvalue:.2e}")
```

Also monitor the **prediction distribution** itself — it's a leading indicator. If the average predicted probability drifts from 0.12 to 0.31, something changed even before labels arrive.

Retraining triggers: scheduled (monthly), performance-based (metric drops below a threshold), drift-based (PSI exceeds 0.25), or data-volume-based (X new labeled rows). Always validate the retrained model against the incumbent before swapping — a fresh model is not automatically a better one.

`evidently` is a good library for automated drift reports.

---

## 35. Experiment tracking

After twenty experiments you will not remember which combination produced the good number. Track from day one.

```python
import mlflow

mlflow.set_experiment("churn-prediction")
with mlflow.start_run(run_name="lgbm-tuned-v3"):
    mlflow.log_params(params)
    mlflow.log_metrics({"cv_auc": cv_score, "test_auc": test_score,
                        "test_pr_auc": pr_auc})
    mlflow.log_artifact("feature_importance.png")
    mlflow.sklearn.log_model(pipe, "model")
    mlflow.set_tags({"features": "v3", "data_snapshot": "2026-08-01"})
```

Minimum viable alternative if MLflow is overkill: a CSV where every row is one experiment, with columns for timestamp, data version, feature set, model, hyperparameters, CV score, test score, and notes. Plus a git commit hash. This is genuinely enough for solo projects.

Non-negotiable reproducibility habits:
- Set `random_state` / seeds everywhere.
- Pin library versions.
- Version your data (or at least record a snapshot date and row count).
- Commit the code that produced each model.

---

# PART IX — PRACTICE

## 36. Ten mini-projects

Work these in order. Each introduces something new. For each: write it as a clean notebook, then refactor into scripts.

---

### Project 1 — Titanic survival (classification fundamentals)
**Data:** `seaborn.load_dataset("titanic")` or Kaggle.
**Goal:** Beat 0.82 CV accuracy.
**Skills:** EDA, missing values, categorical encoding, pipelines, CV.
**Stretch:** Extract titles (Mr/Mrs/Master/Dr) from names — it's the single best engineered feature. Handle cabin deck. Compare five algorithms with a proper CV table.

---

### Project 2 — California house prices (regression)
**Data:** `sklearn.datasets.fetch_california_housing()`.
**Goal:** RMSE below 0.50.
**Skills:** Regression metrics, residual analysis, regularization, feature engineering with ratios.
**Stretch:** Add geographic features (distance to nearest city center, cluster ID from K-Means on lat/long). Compare linear, RF, and GBM. Plot residuals on a map to find where the model fails.

---

### Project 3 — Telco customer churn (imbalance + business value)
**Data:** Telco Customer Churn (Kaggle / IBM sample).
**Goal:** Maximize expected profit given retention offer cost ₹500 and saved-customer value ₹5,000.
**Skills:** Imbalanced classification, threshold optimization, cost-sensitive learning, SHAP.
**Stretch:** Build a decile lift chart. Answer: "if we can only call 1,000 customers, which ones?" Quantify the money saved.

---

### Project 4 — Credit card fraud (extreme imbalance)
**Data:** Kaggle Credit Card Fraud (0.17% positives).
**Goal:** Maximize PR-AUC and precision@100.
**Skills:** Extreme imbalance, PR curves, anomaly detection, sampling techniques.
**Stretch:** Compare supervised (LGBM + class weights) vs. unsupervised (Isolation Forest, autoencoder reconstruction error). Which detects *novel* fraud patterns better? Simulate this by training on the first half of time and testing on the second.

---

### Project 5 — Customer segmentation (unsupervised, end-to-end)
**Data:** Online Retail II (UCI) or any transaction log.
**Goal:** 4–6 actionable segments with names and recommended actions.
**Skills:** RFM feature engineering, clustering, silhouette analysis, cluster profiling, PCA visualization.
**Stretch:** Compare K-Means, hierarchical, and DBSCAN. Add a CLV estimate per segment. Write a one-page memo for a marketing team.

---

### Project 6 — Text classification (NLP)
**Data:** 20 Newsgroups, IMDB reviews, or scraped news headlines.
**Goal:** Beat a TF-IDF baseline by a meaningful margin.
**Skills:** Text preprocessing, TF-IDF, embeddings, transformer fine-tuning.
**Stretch:** Build the accuracy-vs-compute frontier across four approaches. Examine misclassified examples — what does the model consistently get wrong?

---

### Project 7 — Image classification (deep learning)
**Data:** Fashion-MNIST, then CIFAR-10, then a custom folder of your own photos.
**Goal:** >92% on Fashion-MNIST; >85% on CIFAR-10 via transfer learning.
**Skills:** PyTorch, CNNs, data augmentation, transfer learning, learning-rate scheduling.
**Stretch:** Build a confusion matrix and find the classes that get confused. Add Grad-CAM to visualize what the network looks at.

---

### Project 8 — Demand forecasting (time series)
**Data:** Store Item Demand (Kaggle), electricity load, or web traffic.
**Goal:** Beat seasonal-naive on rolling-origin MAE.
**Skills:** Lag features, rolling windows, TimeSeriesSplit, avoiding lookahead bias, multi-step forecasting.
**Stretch:** Produce prediction *intervals* using quantile regression. Compare recursive vs. direct multi-step. Test what happens at a structural break.

---

### Project 9 — Movie recommender
**Data:** MovieLens 100k/1M.
**Goal:** Top-10 recommendations with precision@10 above popularity baseline.
**Skills:** Sparse matrices, matrix factorization, similarity, ranking metrics, cold start.
**Stretch:** Build a hybrid combining collaborative filtering with content features (genre, year). Handle a brand-new user with zero ratings.

---

### Project 10 — Ship one of the above
Take your best model from projects 1–9 and put it in production:
- Refactor the notebook into `data.py`, `features.py`, `train.py`, `predict.py`.
- Add a `config.yaml` and a CLI.
- Write tests: does the pipeline handle a missing column? An unseen category? An empty dataframe?
- Wrap in FastAPI, containerize with Docker.
- Add drift monitoring and a scheduled retrain script.
- Write a README explaining the model, its metrics, its known failure modes, and how to retrain it.

This last project teaches more employable skill than the other nine combined. Most people never do it.

---

## 37. Exercise solutions and hints

**4.1** — `family_size` typically adds 0.005–0.015 AUC; `is_child` a little more, because "women and children first" was literal. The gain is real but modest — engineered features on small clean datasets rarely transform performance.

**4.2** — With `test_size=0.5` you train on half the data, so the model is genuinely worse (higher bias, less to learn from). But the test set is twice as large, so the *estimate* of performance has lower variance. On Titanic (~890 rows) the training-size effect dominates because the dataset is small. On a million rows it wouldn't matter.

**5.1** — Diamonds has `x`, `y`, `z` dimensions of exactly 0 (physically impossible — treat as missing), a `y` value of 58.9 and a `z` of 31.8 (clear data-entry errors given `x` maxes at ~10.7), and `price` is strongly right-skewed (log-transform it). Also `carat` has suspicious clustering just below round numbers — a real market phenomenon, not an error.

**6.1** —
```python
def missing_report(df, target):
    rows = []
    for c in df.columns:
        if c == target or df[c].isna().sum() == 0:
            continue
        m = df[c].isna()
        rows.append({"column": c, "pct_missing": m.mean()*100,
                     "target_when_missing": df.loc[m, target].mean(),
                     "target_when_present": df.loc[~m, target].mean()})
    out = pd.DataFrame(rows)
    out["gap"] = (out["target_when_missing"] - out["target_when_present"]).abs()
    return out.sort_values("gap", ascending=False)
```
A large gap means missingness is informative — always add the indicator column.

**7.2** — `collections_agency_assigned` is leakage (only happens after default) and `final_status` *is* the target. `n_late_payments` is fine only if computed strictly before the application date; if it's a lifetime count including post-application behavior, it's leaky too. This ambiguity is why you must define the feature cut-off time explicitly for every feature.

**9.1** — KFold will show high accuracy because the model memorizes the user-level random effect and sees the same users in both splits. GroupKFold will show ~50% — the truth, since the target contains no generalizable signal. Sketch:
```python
users = np.repeat(np.arange(100), 10)
effect = np.repeat(np.random.randn(100), 10)
X = pd.DataFrame({"user": users, "noise": np.random.randn(1000)})
y = (effect + np.random.randn(1000)*0.1 > 0).astype(int)
X_enc = pd.get_dummies(X["user"], prefix="u").join(X["noise"])
print(cross_val_score(RandomForestClassifier(), X_enc, y, cv=KFold(5, shuffle=True)).mean())
print(cross_val_score(RandomForestClassifier(), X_enc, y, cv=GroupKFold(5), groups=users).mean())
```

**11.1** — Accuracy will be ~95% for both your model and the random one (both mostly predict the majority). ROC-AUC will separate them (0.85 vs 0.50). PR-AUC will show the starkest and most honest difference (0.40 vs 0.05, where 0.05 is the base rate). PR-AUC's baseline is the positive class prevalence, which is why it's the informative metric under imbalance.

**13.1** — Use `sklearn.linear_model.lasso_path`. The last features standing as α grows are the most predictive — on California housing, `MedInc` survives longest. Drop-out order is a crude but useful feature-ranking, though unstable when features are correlated.

**23.1** — Typical result: class_weight and threshold-tuning give most of the gain; SMOTE adds little on GBM and sometimes hurts PR-AUC because synthetic points blur the decision boundary. This is the expected, and slightly disappointing, answer — SMOTE is more popular than it deserves.

**29.1** — If the naive baseline wins, your series is close to a random walk. Its best predictor of tomorrow genuinely is today. That's a valid, publishable finding, and recognizing it saves months of wasted modeling effort.

---

## 38. Study plan and resources

### A 12-week plan

| Weeks | Focus | Deliverable |
|---|---|---|
| 1 | Parts I–II. Numpy, pandas, EDA, pipelines | Project 1 (Titanic) |
| 2 | Part III. Metrics, CV, bias-variance | Project 2 (regression) |
| 3 | Sections 13–17. Linear, logistic, KNN, NB, SVM | Compare all five on both projects |
| 4 | Sections 18–20. Trees, forests, boosting | Redo projects 1–2 with GBM |
| 5 | Sections 22–24. Tuning, imbalance, SHAP | Project 3 (churn) |
| 6 | Consolidate | Project 4 (fraud) |
| 7 | Part VI. Clustering, PCA, anomalies | Project 5 (segmentation) |
| 8 | Section 30. NLP | Project 6 (text) |
| 9–10 | Section 31. PyTorch, CNNs, transfer learning | Project 7 (images) |
| 11 | Section 29. Time series | Project 8 (forecasting) |
| 12 | Part VIII. Deployment, monitoring | Project 10 (ship it) |

### Books, in the order they become useful

1. **Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow** — Aurélien Géron. The best single practical book. Start here.
2. **An Introduction to Statistical Learning** (ISLR) — Free PDF. The theory, gently. Python edition now available.
3. **The Elements of Statistical Learning** (ESL) — Free PDF. ISLR's rigorous older sibling. Reference, not a read-through.
4. **Feature Engineering for Machine Learning** — Zheng & Casari.
5. **Designing Machine Learning Systems** — Chip Huyen. The production side, which most courses skip.
6. **Deep Learning** — Goodfellow et al. Free online. Theory-heavy.
7. **Interpretable Machine Learning** — Christoph Molnar. Free online. The definitive SHAP/PDP/LIME reference.

### Courses
- Andrew Ng's *Machine Learning Specialization* (Coursera) — best conceptual foundation.
- fast.ai *Practical Deep Learning for Coders* — top-down, code-first, excellent.
- Kaggle Learn micro-courses — free, 2–4 hours each, immediately practical.

### Practice
- Kaggle Playground competitions (monthly, clean data, good for technique)
- UCI ML Repository, OpenML, HuggingFace Datasets
- **Best of all:** find data about something you actually care about. Motivation beats curriculum.

---

## Ten things that took practitioners years to learn

1. **Feature engineering beats model selection.** Time spent understanding the domain returns more than time spent tuning.
2. **A simple model you understand beats a complex one you don't.** You will have to debug it at 2am.
3. **If your results look amazing, you have a bug.** Leakage first, genius second.
4. **The baseline is sacred.** Always compute it. Frequently it wins.
5. **Cross-validation setup matters more than the algorithm.** Wrong CV = confidently wrong conclusions.
6. **The metric you optimize is the behavior you get.** Choose it to match the actual decision.
7. **Most of the work is data.** Modeling is the fun 10%.
8. **A deployed 80% model beats a notebook 95% model.** Shipping is a skill.
9. **Models decay.** Plan for retraining before you plan the launch.
10. **Know when ML is the wrong tool.** If a three-line rule solves 90% of it, write the rule.

---

*Work through it, break things, read the tracebacks. That's the actual curriculum.*
