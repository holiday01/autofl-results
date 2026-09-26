# AutoFL Extended Benchmark Notes — sklearn & XGBoost

**Date:** 2026-04-25
**Scripts added:** 3 (sklearn × 2, xgboost × 1)
**Strategies evaluated:** ast, zero_shot, few_shot, structured
**Total evaluations:** 12

---

## 1. New Scripts and What Each Tests

### `benchmarks/sklearn/svm_iris.py`
- **Task:** 3-class Iris classification
- **Framework:** scikit-learn `SVC` with RBF kernel
- **Features tested:** Non-differentiable model (sklearn estimator API); no PyTorch
  gradient graph; 4-dimensional tabular features from `sklearn.datasets.load_iris`.
- **FL challenge:** sklearn estimators have no `.to(device)` method, no `.parameters()`,
  and no gradient tensor. The FL contract requires a `torch.nn.Module`.

### `benchmarks/sklearn/mlp_digits.py`
- **Task:** 10-class handwritten digit recognition
- **Framework:** `sklearn.neural_network.MLPClassifier` (hidden_layer_sizes=(128, 64))
- **Features tested:** A sklearn MLP that does have internal weights but exposes them
  via `coefs_` / `intercepts_` arrays, not as PyTorch parameters; early-stopping and
  `validation_fraction` built in. `sklearn.datasets.load_digits` (1797 × 64).
- **FL challenge:** Same as SVM — estimator interface incompatible with FL runtime.
  However the conceptual translation to PyTorch is straightforward: one hidden-layer
  MLP with ReLU, CrossEntropyLoss.

### `benchmarks/xgboost/xgb_breast_cancer.py`
- **Task:** Binary breast-cancer malignancy classification
- **Framework:** `xgboost.XGBClassifier` (200 trees, depth-4, logloss)
- **Features tested:** Gradient-boosted tree ensemble; non-differentiable wrt feature
  space; uses `sklearn.datasets.load_breast_cancer` (569 × 30 features, binary target).
- **FL challenge:** No neural-network parameters at all; FedAvg cannot average
  decision-tree weights. Conversion to FL requires replacing the ensemble with a
  differentiable PyTorch MLP (`BCEWithLogitsLoss`) that learns the same binary task.

---

## 2. How sklearn/XGBoost FL Differs Conceptually from Deep-Learning FL

### Deep-learning FL (PyTorch/TF/MONAI/Lightning)
- Model parameters are tensors in continuous space; FedAvg = element-wise average.
- Forward pass produces a differentiable loss; gradients flow naturally.
- The FL contract (`build_model → nn.Module`, `train_step → tensor loss`) maps
  directly to how the model already works.

### sklearn / classical-ML FL
- sklearn estimators and XGBoost models have **no differentiable parameter space**.
  FedAvg-style averaging is not directly defined.
- **Conversion strategy chosen** (as produced by the structured converter):
  - Replace the non-differentiable model with an equivalent PyTorch `nn.Module`
    that learns the same classification task using gradient descent.
  - For SVM → 2-layer MLP with BatchNorm+ReLU mimicking RBF kernel decision boundary.
  - For sklearn MLP → PyTorch `nn.Sequential` with the same hidden-layer topology.
  - For XGBoost → deeper MLP (128→64→32→1) with Dropout and `BCEWithLogitsLoss`
    to replicate the binary logloss objective.
- Data loading is unified via `TensorDataset` wrapping the in-memory sklearn
  dataset arrays (no external files needed; synthetic fallback included).
- Alternative approaches (not implemented here but noted for paper):
  - **FedXGBoost:** federate histogram statistics rather than parameters.
  - **Serialised-weight federation:** broadcast `coefs_` arrays and average them
    directly (only valid for linear/MLP sklearn models, not trees).
  - **Split learning:** keep sklearn model local; only share encoded representations.

---

## 3. Benchmark Results (3 scripts × 4 methods)

| script | framework | method | e2e_runnable | preflight_pass | component_coverage | error_stage | elapsed_sec |
|---|---|---|---|---|---|---|---|
| mlp_digits | sklearn | ast | False | False | 1.0 | preflight/data_load | 0.40 |
| mlp_digits | sklearn | zero_shot | False | False | 1.0 | preflight/forward_pass | 0.03 |
| mlp_digits | sklearn | few_shot | False | False | 1.0 | preflight/backward_pass | 0.49 |
| mlp_digits | sklearn | structured | **True** | **True** | 1.0 | — | 12.14 |
| svm_iris | sklearn | ast | False | False | 1.0 | preflight/data_load | 0.05 |
| svm_iris | sklearn | zero_shot | False | False | 1.0 | preflight/forward_pass | 0.02 |
| svm_iris | sklearn | few_shot | False | False | 1.0 | preflight/forward_pass | 0.02 |
| svm_iris | sklearn | structured | **True** | **True** | 1.0 | — | 7.86 |
| xgb_breast_cancer | xgboost | ast | False | False | 1.0 | preflight/data_load | 0.06 |
| xgb_breast_cancer | xgboost | zero_shot | False | False | 1.0 | preflight/forward_pass | 0.03 |
| xgb_breast_cancer | xgboost | few_shot | False | False | 1.0 | preflight/forward_pass | 0.04 |
| xgb_breast_cancer | xgboost | structured | **True** | **True** | 1.0 | — | 26.38 |

### Failure analysis

| method | failure mode |
|---|---|
| **ast** | AST converter cannot detect a `Dataset` class (sklearn scripts define none) → `NotImplementedError` from the generated `build_dataloader` stub. |
| **zero_shot** | LLM wraps sklearn/XGBoost estimator in `build_model` without converting it to `nn.Module`; preflight fails with `'SVC'/'XGBClassifier' object has no attribute 'to'`. For mlp_digits zero_shot the LLM produces PyTorch code but calls `optimizer.zero_grad()` inside `train_step` (violates FL contract) → `AttributeError: 'NoneType' object has no attribute 'zero_grad'`. |
| **few_shot** | For svm_iris and xgb, same sklearn-object error as zero_shot (few_shot example is PyTorch-only, no precedent for sklearn→torch conversion). For mlp_digits few_shot the LLM produces a valid MLP but the loss tensor is computed under `torch.no_grad()` (copied from sklearn's non-gradient logic) → `RuntimeError: element 0 of tensors does not require grad`. |
| **structured** | System prompt explicitly instructs: "If the original uses sklearn or non-PyTorch framework, convert to equivalent `torch.nn.Module`." This produces correct PyTorch models with proper gradient flow. All 3 pass e2e. |

### Summary table

| framework | ast e2e | zero_shot e2e | few_shot e2e | structured e2e | component_coverage (all) |
|---|---|---|---|---|---|
| sklearn (2 scripts) | 0/2 | 0/2 | 0/2 | **2/2** | 1.00 |
| xgboost (1 script) | 0/1 | 0/1 | 0/1 | **1/1** | 1.00 |

---

## 4. Whether This Extends the Paper's Contribution

**Yes, meaningfully.** The existing 10-script benchmark covers gradient-based DL
frameworks exclusively (PyTorch, TensorFlow/Keras, MONAI, Lightning). Adding sklearn
and XGBoost demonstrates two new dimensions:

1. **Framework heterogeneity:** AutoFL's structured converter handles non-differentiable
   ML frameworks by transparently substituting equivalent PyTorch modules. The zero_shot
   and few_shot strategies fail here, quantifying the *value* of the structured prompt's
   explicit framework-translation rules.

2. **Tabular data FL:** All existing benchmark scripts use image data. The new scripts
   test 1D tabular features (Iris, Digits, breast_cancer), exercising the `build_dataloader`
   path without image transforms or torchvision.

3. **Performance gap is starkest here:** structured = 100% e2e; all other methods = 0%.
   This is a stronger differential than for PyTorch scripts (where zero_shot/few_shot
   often also succeed). The result strengthens the paper's argument that the structured
   prompt is essential for non-DL frameworks.

**Suggested paper framing:**
- Report sklearn + XGBoost as a "cross-framework robustness" extension section.
- Table 2 (or an extended Table 1) can show the 0% / 100% structured-only pattern
  as evidence that the FL contract specification in the system prompt drives success.
- Acknowledge limitation: the converted FL module is *semantically approximate*
  (MLP ≠ SVM / gradient boosting) — federated training optimises a related but
  not identical objective. This is an inherent limitation of any gradient-based FL
  approach applied to non-differentiable models.

---

*Generated by Sub-Researcher B for the AutoFL benchmark extension (sklearn + XGBoost).*
