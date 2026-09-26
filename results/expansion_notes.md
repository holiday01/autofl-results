# AutoFL benchmark expansion: fetch log, freeze, and template-converter baseline

Prepared 2026-09-22 from `benchmarks/EXPANSION_CANDIDATES.md`. No LLM was called. All files under `benchmarks/expansion/` are frozen by `benchmarks/expansion/MANIFEST.sha256`.

**Freeze**: `benchmarks/expansion/MANIFEST.sha256` lists 33 files (sha256sum format, verified with `sha256sum -c`); **sha256 of the manifest file itself = `73b0531b3d32cb219e0018252f5bf40f1f5256ba1b772f92e10af0eed124d71f`** (cite this value when referring to the frozen set).

Environment used for the fetch and the template baseline: Python 3.11.9 (pyenv), torch 2.10.0+cu128 (CPU forced via `CUDA_VISIBLE_DEVICES=""`, as in `preflight/validator.py`), monai 1.5.2, xgboost 3.1.3, scikit-learn 1.9.0, nbformat 5.10.4. Not installed: tensorflow, keras, lightning, lightgbm, accelerate, evaluate, jupytext, medicai, kagglehub. The template converter translates Keras/Lightning sources into PyTorch modules and tree-shakes imports, so evaluation of its output does not need those packages; the source scripts themselves were never executed. The host was heavily loaded by unrelated jobs during the baseline run (load average ~80 on 24 cores), so `elapsed_sec` values are not comparable to `results/template_baseline.csv`.

## 1. What was fetched (pinned commit SHAs)

Every upstream file was fetched with `gh api repos/<repo>/contents/<path>?ref=<sha> -H 'Accept: application/vnd.github.raw'` on 2026-09-22, where `<sha>` is the default-branch HEAD of the repository at fetch time (recorded as `commit_sha` in `results/expansion_manifest.csv`). The file-level last commit and the first commit at the current path come from `repos/<repo>/commits?path=<path>` (GitHub does not follow renames).

Repository HEADs used as `ref`:

- `pytorch/examples` @ `acc295dc7b90714f1bf47f06004fc19a7fe235c4`
- `keras-team/keras-io` @ `bc58bcd7ecc24921b38b3c0ae85289a4e3bcdec4`
- `Project-MONAI/tutorials` @ `2ce51396477367791eb53b49cd62617b630d58dd`
- `Lightning-AI/pytorch-lightning` @ `84df182f50ab34301aabb3c0eb4031815bfb413d`
- `microsoft/LightGBM` @ `a0afb85496b1f483710d2a861330321eab794fff`
- `huggingface/transformers` @ `2c4914fb939fe9de0d8e7a798af4684d552f18b4`
- `pytorch/vision` @ `7b0e250acf82aac5a2389f54c6855da17bfeace9`
- `scikit-learn/scikit-learn` @ `2df5847f9c9722fbd333dfc1147f99b9d8de4c74`
- `dmlc/xgboost` @ `09bd738f3ca1e12d8ac0aef95035be2bfefda717`

### 1.1 New development scripts (7)

| ID | framework | stem (local path) | upstream path | file last commit | first commit at path | lines | materialization | sha256 (local, first 12) |
|---|---|---|---|---|---|---|---|---|
| C02 | pytorch | `benchmarks/expansion/dev/pytorch/vae_main.py` | `pytorch/examples:vae/main.py` | `fcce71c1f0` (2025-06-15) | 2017-01-17 | 139 | verbatim | `366116e8252a` |
| C10 | pytorch | `benchmarks/expansion/dev/pytorch/mnist_rnn_main.py` | `pytorch/examples:mnist_rnn/main.py` | `65722fe3ce` (2025-04-30) | 2022-10-27 | 142 | verbatim | `ebd462cfda1f` |
| K10 | tensorflow | `benchmarks/expansion/dev/tensorflow/bidirectional_lstm_imdb.py` | `keras-team/keras-io:examples/nlp/bidirectional_lstm_imdb.py` | `ba29b2832b` (2026-01-21) | 2020-05-06 | 64 | verbatim | `32fb59288c1e` |
| K01 | tensorflow | `benchmarks/expansion/dev/tensorflow/oxford_pets_image_segmentation.py` | `keras-team/keras-io:examples/vision/oxford_pets_image_segmentation.py` | `cfff377988` (2026-01-21) | 2020-05-06 | 275 | verbatim | `80b58e01b0f0` |
| M01 | monai | `benchmarks/expansion/dev/monai/unet_training_array_2d.py` | `Project-MONAI/tutorials:2d_segmentation/torch/unet_training_array.py` | `4b2771acbd` (2023-01-21) | 2020-09-21 | 169 | verbatim | `47d91450b986` |
| L01 | lightning | `benchmarks/expansion/dev/lightning/autoencoder.py` | `Lightning-AI/pytorch-lightning:examples/pytorch/basics/autoencoder.py` | `1b26ac4cf5` (2025-01-07) | 2023-03-07 | 196 | verbatim | `fbd691666054` |
| N11 | lightgbm | `benchmarks/expansion/dev/lightgbm/lgbm_sklearn_example.py` | `microsoft/LightGBM:examples/python-guide/sklearn_example.py` | `a7d00a9611` (2026-04-01) | 2016-12-02 | 80 | verbatim | `51ba4b525dbc` |

### 1.2 Prospective holdout scripts (21 files = 20 plan rows)

| ID | framework | stem (local path) | upstream path | file last commit | first commit at path | lines | materialization | sha256 (local, first 12) |
|---|---|---|---|---|---|---|---|---|
| C01 | pytorch | `benchmarks/expansion/holdout/pytorch/word_language_model_main.py` | `pytorch/examples:word_language_model/main.py` | `28d16ffaa5` (2025-08-23) | 2016-10-17 | 493 | inlined | `e42bcc0be0ad` |
| C03 | pytorch | `benchmarks/expansion/holdout/pytorch/siamese_network_main.py` | `pytorch/examples:siamese_network/main.py` | `fcce71c1f0` (2025-06-15) | 2022-05-05 | 302 | verbatim | `3d02a5162bfa` |
| C04 | pytorch | `benchmarks/expansion/holdout/pytorch/super_resolution_main.py` | `pytorch/examples:super_resolution/main.py` | `6f616144df` (2025-06-24) | 2017-01-13 | 232 | inlined | `deddc828c00a` |
| C05 | pytorch | `benchmarks/expansion/holdout/pytorch/time_sequence_prediction_train.py` | `pytorch/examples:time_sequence_prediction/train.py` | `2c57b0011a` (2020-10-03) | 2017-04-05 | 90 | verbatim | `3a5d6603cecf` |
| C06 | pytorch | `benchmarks/expansion/holdout/pytorch/gcn_main.py` | `pytorch/examples:gcn/main.py` | `d47f0f34f1` (2025-07-10) | 2023-06-12 | 256 | verbatim | `152157088c7a` |
| K03 | tensorflow | `benchmarks/expansion/holdout/tensorflow/ct_3d_image_classification.py` | `keras-team/keras-io:examples/vision/3D_image_classification.py` | `cfff377988` (2026-01-21) | 2020-09-25 | 444 | verbatim | `6c6bddd865ef` |
| K13 | tensorflow | `benchmarks/expansion/holdout/tensorflow/structured_data_classification_from_scratch.py` | `keras-team/keras-io:examples/structured_data/structured_data_classification_from_scratch.py` | `fb0ed4df86` (2025-01-25) | 2020-06-10 | 428 | verbatim | `365ba2d070c6` |
| K16 | tensorflow | `benchmarks/expansion/holdout/tensorflow/timeseries_classification_from_scratch.py` | `keras-team/keras-io:examples/timeseries/timeseries_classification_from_scratch.py` | `857530a3f5` (2024-02-21) | 2020-08-28 | 227 | verbatim | `b96b755488bc` |
| K23 | tensorflow | `benchmarks/expansion/holdout/tensorflow/conditional_gan.py` | `keras-team/keras-io:examples/generative/conditional_gan.py` | `2f12b22af6` (2026-01-21) | 2021-07-16 | 341 | verbatim | `352edfd740cf` |
| K31 | tensorflow | `benchmarks/expansion/holdout/tensorflow/eeg_bci_ssvepformer.py` | `keras-team/keras-io:examples/timeseries/eeg_bci_ssvepformer.py` | `54c7d7366e` (2025-02-06) | 2025-02-06 | 655 | verbatim | `1898fc19a347` |
| K30 | tensorflow | `benchmarks/expansion/holdout/tensorflow/brain_tumor_segmentation.py` | `keras-team/keras-io:examples/vision/brain_tumor_segmentation.py` | `74dcf99e14` (2026-02-05) | 2026-02-02 | 960 | verbatim | `a3611cd3ace1` |
| M03 | monai | `benchmarks/expansion/holdout/monai/densenet_training_dict.py` | `Project-MONAI/tutorials:3d_classification/torch/densenet_training_dict.py` | `4b2771acbd` (2023-01-21) | 2020-09-21 | 162 | verbatim | `593d90a8d75e` |
| M04 | monai | `benchmarks/expansion/holdout/monai/unet_training_array_3d.py` | `Project-MONAI/tutorials:3d_segmentation/torch/unet_training_array.py` | `fc260202fd` (2023-01-30) | 2020-09-21 | 174 | verbatim | `8345326edae2` |
| M13 | monai | `benchmarks/expansion/holdout/monai/densenet_regression_3d.py` | `Project-MONAI/tutorials:3d_regression/densenet_training_array.ipynb` | `3c891ec76d` (2024-09-07) | 2024-06-27 | 204 | notebook | `e6da0ebeb7f4` |
| L02 | lightning | `benchmarks/expansion/holdout/lightning/transformer.py` | `Lightning-AI/pytorch-lightning:examples/pytorch/basics/transformer.py` | `1b26ac4cf5` (2025-01-07) | 2023-04-06 | 62 | verbatim | `417e7ceb2e06` |
| L03 | lightning | `benchmarks/expansion/holdout/lightning/computer_vision_fine_tuning.py` | `Lightning-AI/pytorch-lightning:examples/pytorch/domain_templates/computer_vision_fine_tuning.py` | `1b26ac4cf5` (2025-01-07) | 2023-03-07 | 284 | verbatim | `a8070a4d188e` |
| L05 | lightning | `benchmarks/expansion/holdout/lightning/semantic_segmentation.py` | `Lightning-AI/pytorch-lightning:examples/pytorch/domain_templates/semantic_segmentation.py` | `1b26ac4cf5` (2025-01-07) | 2023-03-07 | 406 | verbatim | `36dbd656bd21` |
| H02 | huggingface | `benchmarks/expansion/holdout/huggingface/run_glue_no_trainer.py` | `huggingface/transformers:examples/pytorch/text-classification/run_glue_no_trainer.py` | `f441076206` (2026-09-22) | 2021-04-21 | 699 | verbatim | `f08c5b707921` |
| T03 | torchvision | `benchmarks/expansion/holdout/torchvision/similarity_train.py` | `pytorch/vision:references/similarity/train.py` | `c35d3855cc` (2023-12-14) | 2019-07-17 | 400 | inlined | `273ef7c3c6fa` |
| N03 | sklearn | `benchmarks/expansion/holdout/sklearn/plot_sgd_early_stopping.py` | `scikit-learn/scikit-learn:examples/linear_model/plot_sgd_early_stopping.py` | `c832691b42` (2026-09-14) | 2018-07-05 | 157 | verbatim | `f906090d6cae` |
| N07 | xgboost | `benchmarks/expansion/holdout/xgboost/xgb_sklearn_examples.py` | `dmlc/xgboost:demo/guide-python/sklearn_examples.py` | `f47b02fc17` (2025-09-03) | 2015-04-02 | 93 | verbatim | `8a0bfcd834bf` |

The holdout stratum contains 21 files because plan row 20 lists two scripts (N03 sklearn + N07 xgboost). The candidates file says to drop H02 (Hugging Face) or T03 (torchvision) to land on exactly 20 if strict within-framework generalization is preferred; both were fetched, and H02 is additionally dependency-gated (`accelerate`, `evaluate` are not installed), so H02 is the natural drop candidate. Deciding that is left to the pre-registration, not to this fetch.

First-commit dates were re-derived from the API today; two differ slightly from the candidates file (C01 `word_language_model/main.py`: 2016-10-17 here vs 2016-10-05 there; C04 `super_resolution/main.py`: 2017-01-13 vs 2017-01-04). The manifest carries today's API values. All contamination conclusions are unchanged.

## 2. Substitutions, multi-file handling, notebook conversion

**Substitutions: none.** No candidate was replaced by an alternate; every script in the plan's dev/holdout tables was fetched successfully.

**Multi-file entry scripts** (sibling modules inlined at the import site; the entry file's own lines are unchanged; each inlined block is delimited by `# ---- BEGIN/END inlined sibling module ... ----` comments carrying the sibling's upstream path, commit and sha256):

- C01 `benchmarks/expansion/holdout/pytorch/word_language_model_main.py`: `data.py` (`1b1a9740df`), `model.py` (`c0b889d5f4`). siblings data.py + model.py inlined at their import sites; `import data` replaced by a SimpleNamespace shim so `data.Corpus(args.data)` stays verbatim; entry code otherwise unchanged.
- C04 `benchmarks/expansion/holdout/pytorch/super_resolution_main.py`: `model.py` (`645c7c386e`), `data.py` (`e0d33a69be`), `dataset.py` (`1c16b6c4b4`). siblings model.py, dataset.py, data.py inlined at their import sites (data.py's own `from dataset import DatasetFromFolder` commented out because dataset.py precedes it); entry code otherwise unchanged.
- T03 `benchmarks/expansion/holdout/torchvision/similarity_train.py`: `loss.py` (`d367a01a18`), `model.py` (`d367a01a18`), `sampler.py` (`7dc5e5bd60`). siblings loss.py, model.py, sampler.py inlined at their import sites; entry code otherwise unchanged.
- C05 `time_sequence_prediction/train.py` is single-file; its helper `generate_sine_wave.py` only writes `traindata.pt` and is not imported, so it was fetched for the record but not inlined.
- H02 `run_glue_no_trainer.py` is single-file but imports `accelerate` and `evaluate` (not installed): kept as fetched, dependency-gated (§6.5 of the candidates file).

**Notebook conversion**: M13 `3d_regression/densenet_training_array.ipynb` -> `benchmarks/expansion/holdout/monai/densenet_regression_3d.py`. `jupytext` is not installed, so the conversion used nbformat 5.10.4: nbformat conversion; 8 code cells; 1 magics commented out (the `!pip install` line), markdown cells dropped, percent-style `# %% [code cell i]` markers inserted, nothing else changed. Note that this notebook auto-downloads `IXI-T1.tar` via `monai.apps.download_and_extract` (the candidates file listed IXI as manual).

All 28 fetched files and all 5 mutants parse with `ast.parse`. 24 of the 28 fetched files are byte-identical to the upstream raw file (manifest column `byte_identical_to_upstream`); the 4 exceptions are the three inlined scripts and the converted notebook, each of which carries a provenance header with the upstream raw sha256.

## 3. Adversarial mutants (5)

Built by exact string edits of dev-set parents (`scratchpad/build_mutants.py`, reproducible); the unified zero-context diff is embedded in each file's header. Changed-line counts are added + removed lines of that diff; the header itself is not part of the diff.

| ID | stem (path) | parent | changed lines | invariant stressed |
|---|---|---|---|---|
| A1 | `benchmarks/expansion/adversarial/pytorch/mnist_detach_logging.py` | `benchmarks/pytorch/mnist_main.py` | 6 (+5/-1) | I3 (the returned loss must stay attached to the graph; the legitimate .detach() calls in logging must not propagate into train_step's return value) |
| A2 | `benchmarks/expansion/adversarial/pytorch/mnist_two_optimizers.py` | `benchmarks/pytorch/mnist_main.py` | 14 (+9/-5) | I1 (the harness owns the single optimizer; the converter must not construct or step either optimizer inside train_step) |
| A3 | `benchmarks/expansion/adversarial/lightning/backbone_manual_optimization.py` | `benchmarks/lightning/backbone_image_classifier.py` | 10 (+9/-1) | I2 (+I1): the backward pass is hidden behind `self.manual_backward`, the optimizer is stepped by the module, and the LR scheduler is stepped per batch |
| A4 | `benchmarks/expansion/adversarial/tensorflow/mnist_convnet_tfdata_missing_dir.py` | `benchmarks/tensorflow/mnist_convnet.py` | 10 (+9/-1) | I4 (the synthetic fallback must reproduce the element structure of a tf.data pipeline built from a missing directory: image float32 [B,28,28,1] (uint8 PNG scale |
| A5 | `benchmarks/expansion/adversarial/pytorch/imagenet_amp_scaler_stepsched.py` | `benchmarks/pytorch/imagenet_main.py` | 15 (+9/-6) | I1 + I2 + I3 combined: scaled loss (`scaler.scale(loss).backward()`), scaler-owned optimizer step with unscale + gradient clipping, and a per-batch `scheduler.s |

Deviations from the plan's construction sketch, both to stay within the 15-line budget: A2 expresses the features/head split by parameter selection (conv1+conv2 -> SGD, fc1+fc2 -> Adam) instead of restructuring `Net`; A4 routes only the training data through the missing-directory tf.data pipeline and keeps the in-memory test arrays for validation. A5 gates autocast/GradScaler on CUDA so the mutant source still runs on CPU; the epoch-level `scheduler.step()` was removed because the scheduler now steps per batch.

## 4. Template-converter baseline (local, no LLM)

Command: `python eval/run_expansion.py --methods template --eval-timeout 3600 --out results/expansion_template_baseline.csv` (data_root `.`, strict False, CPU forced). Generated clients: `eval/_expansion/<stratum>/none/<stem>_template_s0.py`, evaluator logs next to them (`.log`).

**The template rules (`converter/template_converter.py`) were NOT modified for these scripts and must not be: the expansion exists to measure how the frozen rule set generalizes, so a failure below is a result, not a bug to fix.** Likewise no prompt, exemplar, or validator stage was touched.

Column key: `model` = `TEMPLATE_FALLBACK` from the generated header (none = model taken from source; generic = placeholder MLP; surrogate = tabular stand-in for a non-neural estimator); `train_step` = `TEMPLATE_TRAIN_STEP` (extracted from the source loop vs generic); `fallbacks` = number of `TEMPLATE_FALLBACKS` entries; `stage` = evaluator stage reached (`PASS` = 1-round FL simulation completed; otherwise the stage that failed: syntax / interface / preflight/<sub-stage> / e2e / timeout).

### dev_new: e2e 6/7 (with the source model 4; via generic placeholder model 1; via tabular surrogate 1), preflight 6/7, interface 7/7

A PASS with `model = generic` means the converter could not translate the source model and substituted a placeholder MLP on the synthetic shape; it exercises the harness contract, not the source architecture, and should be reported separately from a PASS with `model = none` (source model kept).

| script | framework | model | train_step | fallbacks | stage | preflight | e2e | dataset seen | elapsed s | first error line |
|---|---|---|---|---|---|---|---|---|---|---|
| vae_main | pytorch | none | extracted | 0 | PASS | True | True | MNIST (60000) | 283.51 |  |
| mnist_rnn_main | pytorch | none | extracted | 0 | PASS | True | True | MNIST (60000) | 55.71 |  |
| bidirectional_lstm_imdb | tensorflow | none | generic | 3 | PASS | True | True | TensorDataset (256) | 3.25 |  |
| oxford_pets_image_segmentation | tensorflow | generic | generic | 3 | PASS | True | True | TensorDataset (218) | 4.16 |  |
| unet_training_array_2d | monai | none | extracted | 0 | PASS | True | True | TensorDataset (256) | 10.2 |  |
| autoencoder | lightning | none | extracted | 0 | preflight/forward_pass | False | False | MNIST (60000) | 1.99 | NameError: name 'self' is not defined |
| lgbm_sklearn_example | lightgbm | surrogate | generic | 4 | PASS | True | True | TensorDataset (256) | 1.91 |  |

### holdout: e2e 10/21 (with the source model 3; via generic placeholder model 5; via tabular surrogate 2), preflight 10/21, interface 20/21

A PASS with `model = generic` means the converter could not translate the source model and substituted a placeholder MLP on the synthetic shape; it exercises the harness contract, not the source architecture, and should be reported separately from a PASS with `model = none` (source model kept).

| script | framework | model | train_step | fallbacks | stage | preflight | e2e | dataset seen | elapsed s | first error line |
|---|---|---|---|---|---|---|---|---|---|---|
| word_language_model_main | pytorch | none | extracted | 2 | preflight/import_exit | False | False |  | 4.1 | SystemExit(2) raised while evaluating the module (module-level parse_args()/sys.exit in the generated client;  |
| siamese_network_main | pytorch | none | extracted | 0 | PASS | True | True | MNIST (60000) | 2717.35 |  |
| super_resolution_main | pytorch | none | extracted | 1 | preflight/forward_pass | False | False | TensorDataset (256) | 3.82 | TypeError: Net.__init__() missing 1 required positional argument: 'upscale_factor' |
| time_sequence_prediction_train | pytorch | none | generic | 2 | preflight/forward_pass | False | False | TensorDataset (256) | 3.56 | RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float |
| gcn_main | pytorch | none | generic | 3 | preflight/forward_pass | False | False | TensorDataset (256) | 3.73 | FileNotFoundError: ./cora/cora.content not found. |
| ct_3d_image_classification | tensorflow | generic | generic | 3 | PASS | True | True | TensorDataset (16) | 9.37 |  |
| structured_data_classification_from_scratch | tensorflow | generic | generic | 4 | preflight/forward_pass | False | False | TensorDataset (256) | 3.54 | RuntimeError: Expected floating point type for target with class probabilities, got Long |
| timeseries_classification_from_scratch | tensorflow | generic | generic | 4 | PASS | True | True | TensorDataset (256) | 3.54 |  |
| conditional_gan | tensorflow | generic | generic | 2 | preflight/forward_pass | False | False | MNIST (60000) | 3.61 | RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x784 and 8624x128) |
| eeg_bci_ssvepformer | tensorflow | generic | generic | 4 | PASS | True | True | TensorDataset (256) | 3.55 |  |
| brain_tumor_segmentation | tensorflow | generic | generic | 3 | preflight/backward_pass | False | False | TensorDataset (8) | 8.7 | StopIteration:  |
| densenet_training_dict | monai | none | extracted | 0 | PASS | True | True | _TemplateDictDataset (18) | 32.48 |  |
| unet_training_array_3d | monai | none | extracted | 0 | PASS | True | True | ImageDataset (20) | 20.33 |  |
| densenet_regression_3d | monai | generic | extracted | 2 | PASS | True | True | TensorDataset (256) | 11.07 |  |
| transformer | lightning | none | extracted | 2 | preflight/forward_pass | False | False | TensorDataset (256) | 3.59 | TypeError: LanguageModel.__init__() missing 1 required positional argument: 'vocab_size' |
| computer_vision_fine_tuning | lightning | none | extracted | 1 | preflight/forward_pass | False | False | TensorDataset (111) | 7.33 | NameError: name 'get_torchvision_model' is not defined |
| semantic_segmentation | lightning | none | extracted | 2 | preflight/import_exit | False | False |  | 4.8 | SystemExit(2) raised while evaluating the module (module-level parse_args()/sys.exit in the generated client;  |
| run_glue_no_trainer | huggingface | generic | generic | 4 | PASS | True | True | TensorDataset (256) | 3.74 |  |
| similarity_train | torchvision | none | extracted | 0 | timeout | False | False |  | 3600.07 | evaluation exceeded --eval-timeout=3600s |
| plot_sgd_early_stopping | sklearn | surrogate | generic | 3 | PASS | True | True | TensorDataset (256) | 24.93 |  |
| xgb_sklearn_examples | xgboost | surrogate | generic | 2 | PASS | True | True | TensorDataset (360) | 4.23 |  |

### adversarial: e2e 5/5 (with the source model 5; via generic placeholder model 0; via tabular surrogate 0), preflight 5/5, interface 5/5

A PASS with `model = generic` means the converter could not translate the source model and substituted a placeholder MLP on the synthetic shape; it exercises the harness contract, not the source architecture, and should be reported separately from a PASS with `model = none` (source model kept).

| script | framework | model | train_step | fallbacks | stage | preflight | e2e | dataset seen | elapsed s | first error line |
|---|---|---|---|---|---|---|---|---|---|---|
| mnist_detach_logging | pytorch | none | extracted | 0 | PASS | True | True | MNIST (60000) | 106.21 |  |
| mnist_two_optimizers | pytorch | none | extracted | 0 | PASS | True | True | MNIST (60000) | 107.91 |  |
| backbone_manual_optimization | lightning | none | extracted | 0 | PASS | True | True | MNIST (60000) | 12.7 |  |
| mnist_convnet_tfdata_missing_dir | tensorflow | none | generic | 1 | PASS | True | True | MNIST (60000) | 22.59 |  |
| imagenet_amp_scaler_stepsched | pytorch | none | extracted | 0 | PASS | True | True | TensorDataset (111) | 8.91 |  |

### Per-framework e2e pass counts (all three strata)

| framework | e2e pass / n |
|---|---|
| huggingface | 1/1 |
| lightgbm | 1/1 |
| lightning | 1/5 |
| monai | 4/4 |
| pytorch | 6/10 |
| sklearn | 1/1 |
| tensorflow | 6/9 |
| torchvision | 0/1 |
| xgboost | 1/1 |

Failure-stage histogram (all strata): PASS: 21, preflight/forward_pass: 8, preflight/import_exit: 2, preflight/backward_pass: 1, timeout: 1.

### Adversarial mutants: what the generated train_step actually contains

An e2e PASS on a mutant only shows that the harness contract was met; the table below checks, by AST, whether the construct each mutant plants survived into the generated `train_step` (it must not) and which dataset the evaluation actually ran on.

| mutant | train_step returns | planted constructs found in train_step body | dataset used | reading |
|---|---|---|---|---|
| mnist_detach_logging | `loss` | none | MNIST (60000) | I3 respected: attached `loss` returned, logging detaches dropped |
| mnist_two_optimizers | `loss` | none | MNIST (60000) | I1 respected: both `.step()`/`.zero_grad()` scrubbed (header warnings list them); the two-optimizer split is lost silently (single harness optimizer) |
| backbone_manual_optimization | `loss` | none | MNIST (60000) | I2 respected: `manual_backward`, `opt.step()`, `lr_schedulers().step()` all scrubbed |
| mnist_convnet_tfdata_missing_dir | `loss` | none | MNIST (60000) | wrong-route pass: `train_step` is the generic Keras-fit stand-in and `build_dataloader` probes torchvision `MNIST(data_path)` FIRST (alias of the retained `keras.datasets.mnist.load_data()`), then `ImageFolder('./data/mnist_png/train')`, then synthetic. With `./MNIST` present at data_root the run trained on the in-memory arrays the mutant deleted, i.e. the plausible-but-wrong route named in the mutant header. I4 is only exercised when data_root has no MNIST |
| imagenet_amp_scaler_stepsched | `loss` | none | TensorDataset (111) | I1+I2+I3 respected: scaler/autocast/clip/scheduler all absent from train_step, unscaled attached `loss` returned (dummy FakeData path) |

Expected-fail rows (out-of-contract by construction, per the candidates file): C05 `time_sequence_prediction_train` (LBFGS closure). A template PASS on that script means the converter changed the algorithm (LBFGS -> the harness optimizer), which is itself a finding to report, not a success.

### Run history and side effects

- The run was started at 21:45 (2026-09-22) with 24-thread children; after 8 rows it was stopped during the `siamese_network_main` evaluation and relaunched with `--resume --eval-threads 4` (OMP/MKL threads capped at 4 per child, one evaluation at a time) because the host was shared with four other Phase-2 runs. Rows already recorded were kept; `siamese_network_main` was re-evaluated from scratch. The first `word_language_model_main` row had been labelled `eval_crash` (the child died on a module-level `parse_args()`); the child-side handler was then extended to report this as `preflight/import_exit` with the syntax/interface fields filled, the row was deleted and the script re-evaluated (same generated file, same outcome, clearer label). `semantic_segmentation` later hit the same defect and was labelled by the new handler directly.
- `similarity_train` (torchvision resnet50 embedding net over FashionMNIST, batch 4, 1 local epoch = 15 000 steps on 4 CPU threads) exceeded the 3600 s evaluation timeout; recorded as `timeout`, not as a converter failure. Its preflight had passed before the e2e round started (see its .log).
- Side effect: evaluating `similarity_train` downloaded FashionMNIST (82 MB) into `FashionMNIST/raw/` under the repository root (data_root `.`), exactly as the primary harness had done for MNIST. No other file outside `benchmarks/expansion/`, `eval/_expansion/`, `eval/run_expansion.py` and `results/expansion_*` was written by this work. (`cifar10/` in the repo root predates the first expansion evaluation and belongs to another run.)

### Launching the LLM strategies later (not run here)

```
python eval/run_expansion.py --methods zero_shot few_shot_corrected structured --provider claude \
    --samples 1 --resume --out results/expansion.csv            # add --stratum holdout etc. to restrict
python eval/run_expansion.py --methods structured --dry-run --stratum adversarial   # pipeline test, 0 LLM calls
```

Every generated module (LLM or template) is evaluated in a child process (`--eval-timeout`, default 1800 s; 3600 s was used for the template baseline) with `OMP_NUM_THREADS=MKL_NUM_THREADS=--eval-threads` (default 4) and `CUDA_VISIBLE_DEVICES=""`; a timeout is recorded as `error_stage=timeout` and the child's process group is killed. Prompts come unchanged from `autofl.converter.llm_converter._build_prompt` (structured: spec v2 by default); calls go through `autofl.eval.llm_calls.generate_file` with `--max-calls` as the hard budget. `--resume` keys on (script_name, stratum, method, provider, sample). Output files: `eval/_expansion/<stratum>/<provider>/<stem>_<method>_s<k>.py` (+ `.log`).

## 5. Files created (no existing file was modified)

- `benchmarks/expansion/dev/<framework>/<stem>.py` (7), `benchmarks/expansion/holdout/<framework>/<stem>.py` (21), `benchmarks/expansion/adversarial/<framework>/<stem>.py` (5), `benchmarks/expansion/MANIFEST.sha256`
- `results/expansion_manifest.csv` (33 rows; columns: stratum, framework, stem, path, upstream_url, commit_sha, sha256, lines, task_type, dataset_needed, download_needed, loop_style, invariant_stress, difficulty, first_commit_date, post_cutoff, notes, plus candidate_id, repo, repo_path, file_last_commit_sha, file_last_commit_date, upstream_raw_sha256, byte_identical_to_upstream, materialization, fetch_date)
- `eval/run_expansion.py` (runner: template / zero_shot / few_shot_corrected / structured; --stratum, --frameworks, --scripts, --provider, --samples, --max-calls, --dry-run, --resume, --eval-timeout; rows appended to results/expansion.csv by default; generated files in eval/_expansion/<stratum>/<provider>/)
- `results/expansion_template_baseline.csv` and `eval/_expansion/*/none/*` (template baseline outputs + logs)
- `results/expansion_notes.md` (this file)

`post_cutoff` in the manifest is True only for K30 `brain_tumor_segmentation` (first commit 2026-02-02, after the January 2026 Claude Sonnet 4.6 training cutoff) and for the five constructed mutants. K31 (2025-02-06) and M13 (2024-06-27) are after the Gemini 2.5 Flash / Llama 3 cutoffs respectively but before the Claude cutoff.
