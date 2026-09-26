# Non-DL proxy semantic validation

Compares the original non-DL estimator (sklearn / XGBoost) against the
PyTorch proxy used in the FL pipeline. All splits use `test_size=0.2`,
`random_state=42`, with `StandardScaler` features, matching the
`*_fl_structured.py` data loaders.

## Results

| Script | Dataset | Original | Proxy (central) | Proxy (FedAvg, 2 IID) | Δ (proxy_central − original) | Within 5%? |
|---|---|---|---|---|---|---|
| `benchmarks/sklearn/mlp_digits.py` | Digits | sklearn.MLPClassifier = 0.9667 | 0.9833 | 0.9778 | +0.0167 | yes |
| `benchmarks/sklearn/svm_iris.py` | Iris | sklearn.SVC(rbf) = 0.9667 | 0.9667 | 0.9333 | +0.0000 | yes |
| `benchmarks/xgboost/xgb_breast_cancer.py` | BreastCancer | xgb.XGBClassifier = 0.9561 | 0.9649 | 0.9737 | +0.0088 | yes |

## Verdict

All three proxies stay within ±5% test accuracy of the original estimator. The FL proxy is therefore a faithful surrogate rather than a degenerate substitute, and the FedAvg accuracy is a meaningful indicator of federated learnability for the task.

## Procedure

- Original estimators trained centrally with the same train/test split and feature scaling described in each `*.py` benchmark script.
- PyTorch proxies imported via `build_model` from the corresponding `*_fl_structured.py` file. Centralized run uses Adam, lr=1e-3, matching the FL local optimizer settings.
- FedAvg runs use 2 IID clients (random partition of the train set), 10 communication rounds, and the same total number of local epochs as the centralized proxy run (split evenly per round).
- Test accuracy is reported on the held-out 20% test split.
