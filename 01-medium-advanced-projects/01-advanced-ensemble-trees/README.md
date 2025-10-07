# 📦 Module 1: Advanced Ensemble Trees

Build robust, high-performing models using bagging-based tree ensembles with production-grade practices.

## 🚀 Objectives
- Master Random Forests and Extra Trees for tabular dominance
- Optimize OOB scoring, class weighting, and balanced subsampling
- Tune for performance with Optuna and scikit-learn search APIs
- Achieve reproducible training, versioning, and test-time stability

## 🔧 Tech Stack
- Python 3.8+
- scikit-learn 1.3+
- Optuna 3.4+
- pandas, numpy, matplotlib, seaborn

## 📂 Structure
```
01-advanced-ensemble-trees/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_random_forest_baseline.ipynb
│   ├── 03_extra_trees_optimization.ipynb
│   ├── 04_feature_importance_and_stability.ipynb
│   └── 05_model_export_and_inference.ipynb
├── datasets/
│   └── README.md  # download links or generation scripts
├── utils/
│   ├── preprocessing.py
│   ├── evaluation.py
│   └── io.py
└── README.md
```

## 🧪 Experiments
- Compare RF vs ExtraTrees on multiple datasets (imbalanced + balanced)
- Feature importance robustness with permutation vs impurity
- OOB vs CV metrics agreement analysis
- Impact of n_estimators, max_features, and bootstrap on variance

## 🏗️ Reproducible Training Template
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import optuna, numpy as np

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1200, step=100),
        'max_depth': trial.suggest_int('max_depth', 4, 32),
        'max_features': trial.suggest_float('max_features', 0.3, 1.0),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
    }
    clf = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, proba)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=40)
print('Best params:', study.best_params)
```

## 📈 Reporting & Visualization
- Confusion matrices, ROC/PR curves, calibration plots
- Stability plots for feature importances across bootstraps
- Partial Dependence and ICE for top features

## 🧰 Deliverables
- Tuned RF/ET models with serialized artifacts (.joblib/.pkl)
- Reproducible notebooks with results
- Deployment-ready inference script

## ✅ Next: Module 2 — Gradient Boosting Mastery
Move to XGBoost, LightGBM, and CatBoost for best-in-class accuracy with explainability hooks.
