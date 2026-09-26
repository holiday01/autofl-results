# Source-to-converted differential tests (structured-prompt conversions)

**Date:** 2026-09-22
**Script:** `eval/differential_test.py` (new; no existing file modified)
**Raw results:** `results/differential_test.csv` (one row per pair per metric; columns
`pair_id, source_script, converted_script, stage, metric, source_value, converted_value,
comparison_value, verdict, notes, device`)
**Seed:** 42 everywhere (Python, NumPy, torch; re-seeded immediately before every compared
forward pass so that dropout masks and GAN noise `z` are identical on both sides)
**Devices:** P1, P2, P3, P5, P6, P7 on CPU (`CUDA_VISIBLE_DEVICES=""`, 16 threads).
P4 (DCGAN, 64x64 CIFAR-10) on CUDA (RTX 5070 Ti, `cudnn.deterministic=True`,
`cudnn.benchmark=False`) because one DCGAN step costs ~7 s on this CPU (900 steps for
three arms would have taken ~1.75 h); everything else about P4 is identical.
**Budget for stage (e):** 300 optimizer steps at batch size 64 (P5 additionally uses the
epoch count that sklearn's early stopping actually ran).

---

## 1. Why this test exists

The five-stage harness (`eval/evaluator.py`) certifies that a converted module is
*executable* under the FL runtime: syntax, interface, optimizer contract, preflight,
end-to-end run. A reviewer correctly pointed out that this does not certify *faithful
migration*. This test drives the original script's own code and the converted module on the
same real data and compares them at five levels: (a) preprocessing, (b) architecture,
(c) loss at initialisation, (d) one optimizer step, (e) task outcome after a fixed budget.

## 2. How the two sides are driven

**Source side.** Wherever a source component is importable it is executed as-is:

- P1 `mnist_main.py`: `Net`, and the script's own `train()` loop (called with a one-batch
  or step-capped loader), `optim.Adadelta(lr=1.0)` as in `main()`.
- P2 `mnist_lite.py`: the `GAN` LightningModule is imported through a shim that maps the
  `lightning.pytorch` 2.x namespace onto the installed `pytorch_lightning` 1.9.5 (the
  `lightning` package is not installed). `GAN.training_step` (manual optimisation, two Adam
  optimizers) is executed verbatim; the Trainer-backed services it uses (`optimizers()`,
  `toggle_optimizer`/`untoggle_optimizer`, `manual_backward`, `log_dict`) are provided on
  the instance with Lightning's semantics (toggle freezes the parameters not owned by the
  optimizer). Data come from Lightning's own `MNISTDataModule` pointed at the local MNIST.
- P3 `backbone_image_classifier.py`: `LitClassifier.training_step` and
  `configure_optimizers()` executed verbatim (automatic optimisation = zero_grad, step,
  backward, optimizer.step); data from the script's own `MyDataModule` with
  `DATASETS_PATH` redirected to the local MNIST copy.
- P4 `dcgan_main.py`: the script has module-level `argparse` (a required `--dataset`), so
  its `weights_init`, `Generator` and `Discriminator` definitions are compiled from the
  file's AST with the module globals they read (`nz=100, ngf=64, ndf=64, nc=3`), and the
  training-loop body (lines 213-239: D step on `errD_real + errD_fake`, then G step on
  `errG`) is transcribed verbatim. Dataset: CIFAR-10 through the script's own `cifar10`
  branch (Resize 64, ToTensor, Normalize 0.5).
- P5 `mlp_digits.py`: the script's split/scaler/`MLPClassifier` lines are reproduced; the
  converted model's initial parameters are injected into the sklearn estimator
  (`coefs_`/`intercepts_`, transposed) so that loss and gradient are compared at identical
  parameters; sklearn's own `_backprop` gives the source objective and gradient, and a
  `partial_fit` on the same 64-row batch with a fresh `AdamOptimizer` gives the source's
  one-step update.
- P6/P7: the script's estimator is trained as written (SVC / XGBClassifier) and only the
  outcome can be compared; no parameter-space loss or gradient exists.

**Converted side.** `build_model(config)`, `build_dataloader(config, split)` and
`train_step(model, batch, None, config)` are called exactly as the runtime calls them, and
every parameter update is produced by the unmodified `fl_runtime.client.FLClient.local_train`
(AdamW at `config["learning_rate"]`, `clip_grad_norm_(1.0)`, one `zero_grad`/`backward`/
`step` per batch). The loader handed to `local_train` is the converted module's own
`build_dataloader` output, capped at the step budget (or a single batch for stage (d)).
The config uses the key set of `eval/evaluator.py::_MINIMAL_CONFIG` with
`data_path=/home/holiday01/autofl`, `allow_synthetic_data=False`, `local.batch_size=64`
and the learning rate listed per pair below. P4 additionally passes
`model_kwargs={"workers": 0}` (in-process data loading; model hyper-parameters unchanged).

**Learning rates for the runtime arm.** The runtime's optimizer is always AdamW; its lr is
set to the source's lr where the source uses an Adam-family optimizer (P2/P4: 2e-4,
P3: 1e-4, P5: 1e-3) and to the harness default 1e-3 for P1 (the source uses
Adadelta lr=1.0, which is not transferable) and for P6/P7 (no lr in the source).

**Three arms in stage (e).** `source_native` = the source's own optimizer(s) and loop;
`source_adamw` = the source's loop with AdamW at the same lr substituted (optimizer-matched
control, so that a difference to the runtime arm cannot be attributed to the optimizer
type); `converted_runtime` = converted module under `FLClient.local_train`.

## 3. Pairs that cannot be run here

`keras`/`tensorflow` and MONAI's `nibabel`-based pipelines are not installed in this
environment (`tensorflow` import fails), so the TensorFlow and MONAI source scripts cannot be
executed and their pairs are not part of this test. The seven pairs below cover the
frameworks that can be executed: plain PyTorch (2), Lightning (2), scikit-learn (2, one
MLP and one surrogate), XGBoost (1, surrogate).

---
## 4. Summary

| Pair | Source -> converted | Device | (a) preprocessing | (b) architecture | (c) loss at init | (d) gradient cos / parameter-delta cos (runtime vs optimizer-matched source) | (e) outcome, 300 steps at batch 64 | Verdict |
|---|---|---|---|---|---|---|---|---|
| P1 | `pytorch/mnist_main.py` -> `mnist_main_fl_structured.py` | cpu | identical transform (max abs pixel diff 0 on aligned samples); 60000 train vs 54000/6000 split | identical: 1,199,882 params, same leaf sequence, state_dict keys/shapes equal | 2.31161 = 2.31161 (train mode, same dropout RNG); 2.30667 = 2.30667 (eval) | 1.000 / 0.99999 (vs source's own Adadelta step: 0.72, an optimizer effect) | MNIST test acc: 0.9738 (source, Adadelta lr 1.0), 0.9735 (source, AdamW), 0.9734 (converted under runtime) | semantics preserved |
| P2 | `lightning/mnist_lite.py` (GAN, manual optimisation) -> `mnist_lite_fl_structured.py` | cpu | identical (ToTensor; aligned diff 0); 55000/5000 vs 54000/6000 | identical G (1,510,032) and D (533,505); full state_dict keys equal | g_loss, d_real, d_fake all equal; source optimises D on (real+fake)/2 = 0.694, converted folds real+fake = 1.388 into one scalar | G: 1.000 / 1.000. D: 0.77 / 0.87. Leak term cos(dg/dD, d_fake/dD) = -1.000; fake-branch D signal is 3.1% of the source's | D accuracy on fakes: 1.000 (source, two AdamW), 0.977 (source native) vs 0.049 (converted under runtime); mean D prob that a fake is real 0.30 vs 0.74 | runtime-compliant, not semantics-preserving |
| P3 | `lightning/backbone_image_classifier.py` -> `backbone_image_classifier_fl_structured.py` | cpu | identical (ToTensor; aligned diff 0); 55000/5000 vs 54000/6000; source loader unshuffled | identical: 101,770 params | 2.30735 = 2.30735 | 1.000 / 1.000 (native Adam and AdamW alike) | test acc 0.7729 (source, own loader) vs 0.8312 (converted under runtime); source code fed the converted loader's batches: 0.8319 (delta -0.0007) | semantics preserved |
| P4 | `pytorch/dcgan_main.py` (two-optimizer DCGAN) -> `dcgan_main_fl_structured.py` | cuda | identical (CIFAR-10, Resize 64, Normalize 0.5; aligned diff 0); 50000 vs 45000/5000 | identical G (3,576,704) and D (2,765,568) | errD_real, errD_fake, errG, errD all equal (2.52697 total) | G gradient 1.000 but delta_G cos 0.65 (source updates D first, then G through the updated D). D: gradient cos 0.45, delta cos 0.35; leak cos -0.94; fake-branch D signal 34% of the source's | D accuracy on fakes (BN batch stats): 1.000 (both source arms) vs 0.000 (converted under runtime); mean D prob that a fake is real 0.001-0.05 vs 0.76 | runtime-compliant, not semantics-preserving |
| P5 | `sklearn/mlp_digits.py` (MLPClassifier) -> `mlp_digits_fl_structured.py` (torch port) | cpu | scaler fit on all 1797 rows (source: train split only); random 80/20 split, unstratified; held-out overlap Jaccard 0.10 | same topology, 17,226 params (layer-wise transposes); sklearn adds L2 alpha=1e-4, port has none | log-loss 2.30031 = 2.30031; sklearn objective incl. L2 2.30036 | 1.000 / 0.9991 (sklearn Adam vs runtime AdamW) | digits test acc (source split): 0.9667 sklearn vs 0.9778 (port, same 39 epochs) and 0.9694 (300 steps); own-split val 0.9666 | model and objective preserved; preprocessing/split policy deviates |
| P6 | `sklearn/svm_iris.py` (SVC rbf) -> `svm_iris_fl_structured.py` (MLP surrogate) | cpu | scaler on all rows with ddof=1 std (max abs diff 0.079); random split, Jaccard 0.07 | different family by design (SVC: 285 stored values; MLP: 4,931 params) | n/a | n/a | iris test acc 0.9667 vs 0.9667 | surrogate; outcome equal |
| P7 | `xgboost/xgb_breast_cancer.py` -> `xgb_breast_cancer_fl_structured.py` (MLP surrogate) | cpu | scaler on all rows; random split, Jaccard 0.13 | different family by design (200 trees, 787 leaves; MLP 14,785 params) | n/a | n/a | acc 0.9561 / AUC 0.9940 vs 0.9649 / 0.9977 | surrogate; outcome equal or better |

Per-metric rows for every pair are in `results/differential_test.csv`; the same rows are
reproduced as tables in the appendix.

## 5. Interpretation

### 5.1 Conventional classifiers (P1, P3): faithful

Every level agrees. The converted `build_dataloader` instantiates torchvision's real
`MNIST` (not the synthetic fallback) with the same transform: the same index yields the
same tensor (max abs difference 0 over 256 aligned examples, identical labels, same dtype
and value range). `build_model` produces the same module: same parameter count, same
leaf-module sequence, identical `state_dict` keys and shapes, so the source's parameters
load into the converted model without mapping. With identical parameters and the same real
batch, the source's own loss computation (`F.nll_loss` in `mnist_main.train()`,
`LitClassifier.training_step`) and the converted `train_step` return the same value
(dropout RNG re-seeded for P1), and the gradients have cosine 1.000.

For the update, the runtime's `FLClient.local_train` (AdamW + `clip_grad_norm_(1.0)`)
reproduces the source loop's parameter change with cosine 0.99999 (P1) and 1.000 (P3) when
the source loop is run with AdamW at the same learning rate. The only visible difference in
P1 is optimizer choice: the script uses `Adadelta(lr=1.0)`, the runtime uses AdamW, and the
one-step parameter-delta cosine between the two is 0.72. That is a property of the runtime's
fixed optimizer, not of the conversion, and after 300 steps it does not matter for the task:
0.9738 (Adadelta) / 0.9735 (AdamW) / 0.9734 (converted under the runtime).

P3 shows how easily a naive outcome comparison misleads. The source `MyDataModule`
returns an *unshuffled* loader over its 55000-example subset, the converted loader shuffles
a 54000-example subset, and at lr 1e-4 after 300 steps the model is far from converged
(77-83% accuracy). The 0.058 gap between "source on its own loader" (0.7729) and "converted
under the runtime" (0.8312) disappears when the source's `training_step` + AdamW are fed
the converted loader's batches (0.8319, delta -0.0007): the gap is batch order, not code
semantics.

### 5.2 GANs (P2, P4): executable, but the training procedure is not preserved

Both GAN conversions pass the harness because they satisfy its contract: one `nn.Module`,
one `train_step` that returns one differentiable scalar, no optimizer calls inside. Both
reproduce the source architecture exactly and, at identical parameters and identical noise
`z`, every loss term the source computes: g_loss, d_real, d_fake (P2) and errD_real,
errD_fake, errG (P4) agree to float precision. What is not preserved is *what is optimised
with what*.

The source scripts alternate two optimizers. In P2 (`GAN.training_step`, manual
optimisation) the generator step back-propagates `g_loss` with the discriminator frozen
(`toggle_optimizer`) and the discriminator step back-propagates `(real_loss + fake_loss)/2`
after `opt_d.zero_grad()`, so no part of the generator loss ever reaches the discriminator's
update. In P4 (`dcgan_main.py`) the discriminator is stepped on `errD_real + errD_fake`
before `errG.backward()` runs, and the gradient that `errG` deposits in `netD` is discarded
by the next iteration's `netD.zero_grad()`. The converted modules instead return
`d_real + d_fake + g_loss` (P2) / `errD + errG` (P4), and the runtime applies one AdamW
step over generator *and* discriminator parameters.

The reviewer's specific concern, that the generator-loss gradient flows into D through the
non-detached fake images, is confirmed and quantified. For a fake sample with discriminator
logit x, `d_fake = -log(1 - sigmoid(x))` has derivative `sigmoid(x)` and
`g_loss = -log sigmoid(x)` has derivative `sigmoid(x) - 1`; their sum is `2 sigmoid(x) - 1`,
which pushes D(fake) towards 0.5 instead of 0 and vanishes there. (The same algebra holds
for P4's `Sigmoid + BCELoss`.) Measured at identical parameters:

- cosine between the leaked generator-loss gradient on D and the source's fake-branch
  gradient on D: -1.000 (P2), -0.943 (P4);
- norm of the fake-branch signal that D actually receives in the summed loss, relative to
  the source's: 3.1% (P2), 34% (P4);
- cosine of the full D gradient, converted vs source: 0.77 (P2), 0.45 (P4); after the
  optimizer step, D parameter-delta cosine 0.87 (P2) and 0.35 (P4).

The generator gradient is identical in both pairs (cosine 1.000) because `d_fake` is
detached and `d_real` does not depend on G. In P2 the generator step therefore matches
the source exactly (delta cosine 1.000); in P4 it does not (0.65) for a second, ordering
reason: the source evaluates `errG` through the discriminator *after* the discriminator's
update, whereas one summed backward can only see the pre-update discriminator.

After 300 steps the consequence is not subtle. Evaluated with BatchNorm batch statistics
(how D saw its batches during training; eval-mode numbers are also in the CSV) the source
discriminators reject fakes at 0.977-1.000 (P2) and 1.000 (P4), while the discriminators
trained through the converted modules reject 4.9% (P2) and 0.0% (P4) of fakes and assign
them a mean "real" probability of 0.74 (P2) and 0.76 (P4). Real images are still recognised
(0.78 and 0.998), i.e. D has learned only the real branch; the fake branch was cancelled.
The optimizer-matched control (the source loop with two AdamW optimizers at the same lr)
still trains a working discriminator, so the failure is attributable to the
summed-loss/single-step conversion, not to the switch from Adam(beta1=0.5) to AdamW.
Consistently, P2's converted generator drifts to a mean pixel value of 0.33 (real MNIST:
0.13; source generators: 0.07-0.12).

Two secondary deviations are recorded as well: P2's source halves the discriminator loss
(`(real+fake)/2`), the converted module does not, which changes the relative weighting of
D terms vs g_loss inside the joint loss; and the runtime clips the norm of the *joint*
gradient vector, so the discriminator's and generator's steps are no longer independent.

Nothing in this test suggests the GAN modules are wrong *as runtime clients*; they are the
faithful expression of a two-player alternating procedure into a contract that only has one
loss and one optimizer. Preserving the source semantics would require the runtime contract
to admit multiple optimizers / alternating steps (or a `train_step` that performs its own
alternation, which the harness's optimizer rule deliberately rejects).

### 5.3 sklearn MLP port (P5): model and objective preserved, data protocol deviates

The port is a faithful re-expression of `MLPClassifier(hidden_layer_sizes=(128, 64))`:
same topology (layer-wise transposed weights, 17,226 parameters), and with the port's
initial weights injected into the sklearn estimator the two compute the same log-loss on
the same 64-row batch (2.30031; sklearn's training objective adds 0.5*alpha*||W||^2/n =
5e-5) and the same gradient (cosine 1.000, computed with sklearn's own `_backprop`). One
sklearn `partial_fit` Adam step and one runtime AdamW step move the parameters in the same
direction (cosine 0.9991; the residual is sklearn's L2 term vs AdamW's decoupled decay and
the runtime's clipping). Trained on the source's train split for the 39 epochs that
sklearn's early stopping actually ran, the port reaches 0.9778 on the source's test split
vs 0.9667 for sklearn (0.9694 with the fixed 300-step budget).

What the conversion did change is the data protocol, and this is worth stating in the
paper: the converted `build_dataloader` fits `StandardScaler` on all 1797 rows before
splitting (the source fits on the train split only) and draws an unstratified random 80/20
split whose held-out set overlaps the source's test set only at Jaccard 0.10. The aligned
per-feature difference is small on average (0.009) but large on near-constant border pixels
(max 21.6), where the two fit populations give different standard deviations. The same
pattern (whole-data scaler, random split) appears in P6 and P7. It does not change the
learning problem, but it means the converted module's "val" split is not the script's test
split, and its scaler has seen the validation rows.

### 5.4 Surrogates (P6, P7): different models by design; outcomes match

The SVC and XGBoost sources have no parameter-space loss or gradient, so levels (c) and (d)
are not defined and are recorded as n/a. Level (a) shows the same data-protocol deviations
as P5 (P6 additionally standardises with ddof=1 via `torch.std`, max aligned difference
0.079). Level (e), the only meaningful comparison, is neutral or favourable to the
surrogates on the source's own test split: iris 0.9667 vs 0.9667; breast cancer accuracy
0.9561 -> 0.9649 and ROC-AUC 0.9940 -> 0.9977. This agrees with
`results/non_dl_proxy_validation.csv`.

## 6. Caveats

- 300 steps at batch 64 is a short budget chosen for a same-optimizer comparison, not for
  convergence (P3 is at 77-83%; the GANs are early). The GAN outcome metrics are
  discriminator real/fake accuracy and generator output statistics, deliberately not FID.
- The Lightning sources ran on `pytorch_lightning` 1.9.5 through a namespace shim, with the
  Trainer-provided services of manual optimisation emulated on the instance; the DCGAN
  classes were compiled from the script's AST and its loop body transcribed (the script
  cannot be imported). All other source code was executed as written.
- P4 ran on CUDA with cuDNN in deterministic mode; all other pairs on CPU. Elapsed times in
  the CSV were measured on a shared machine and are not benchmarks.
- TensorFlow/Keras and MONAI pairs are not covered (packages not installed).

---

## Appendix: all recorded metrics

### P1: `benchmarks/pytorch/mnist_main.py` -> `benchmarks/pytorch/mnist_main_fl_structured.py`

| stage | metric | source | converted | comparison | verdict | notes |
|---|---|---|---|---|---|---|
| a_preprocessing | dataset_class | MNIST | MNIST |  | match | class of the dataset actually instantiated (synthetic fallback would be TensorDataset) |
| a_preprocessing | converted_used_real_data |  | True |  | match | True when the converted loader wraps torchvision's real dataset |
| a_preprocessing | dataset_len_underlying | 60000 | 60000 |  | match | length of the underlying dataset |
| a_preprocessing | train_split_len | 60000 | 54000 |  | info | source: no split (60000 train; separate 10k test set); converted: 90/10 random_split(seed 42) |
| a_preprocessing | val_split_len | - | 6000 |  | info |  |
| a_preprocessing | example_shape | (1, 28, 28) | (1, 28, 28) |  | match |  |
| a_preprocessing | example_dtype | torch.float32 | torch.float32 |  | match |  |
| a_preprocessing | value_min_first256 | -0.424213 | -0.424213 | 0 | match | over the first 256 examples |
| a_preprocessing | value_max_first256 | 2.82149 | 2.82149 | 0 | match | over the first 256 examples |
| a_preprocessing | value_mean_first256 | -0.00788985 | -0.00788985 | 0 | match | over the first 256 examples |
| a_preprocessing | value_std_first256 | 0.990687 | 0.990687 | 0 | match | over the first 256 examples |
| a_preprocessing | max_abs_pixel_diff_first256 |  |  | 0 | match | sample-aligned: same index in both datasets |
| a_preprocessing | labels_identical_first256 |  |  | 1 | match |  |
| a_preprocessing | label_set | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] |  | match |  |
| a_preprocessing | loader_batch_size_used |  | 64 |  | info | converted loader batch size from config['local']['batch_size'] |
| a_preprocessing | converted_train_shuffle |  | True |  | info |  |
| b_architecture | param_count | 1199882 | 1199882 | 0 | match |  |
| b_architecture | leaf_modules | Conv2d>Conv2d>Dropout>Dropout>Linear>Linear | Conv2d>Conv2d>Dropout>Dropout>Linear>Linear |  | match | leaf nn.Module types in registration order |
| b_architecture | state_dict_keys_match | 8 | 8 |  | match |  |
| b_architecture | state_dict_shapes_match |  |  | True | match |  |
| c_loss_init | loss_train_mode | 2.31161 | 2.31161 | 0 | match | identical params, same real batch, dropout RNG re-seeded |
| c_loss_init | loss_eval_mode | 2.30667 | 2.30667 | 0 | match | dropout disabled |
| d_one_step | grad_cosine |  |  | 1 | match | gradient of source loss vs gradient of converted train_step, same params/batch |
| d_one_step | param_delta_runtime_vs_source_native_cosine |  |  | 0.719912 | differs | source: Adadelta(lr=1.0) via mnist_main.train(); runtime: AdamW(lr=1e-3)+clip_grad_norm(1.0) |
| d_one_step | param_delta_runtime_vs_source_native_rel_l2 | 1.06649 | 0.93962 | 0.712524 | info | \|\|delta_runtime - delta_source\|\| / \|\|delta_source\|\| |
| d_one_step | param_delta_runtime_vs_source_adamw_cosine |  |  | 0.999997 | match | source loop with AdamW(lr=1e-3) substituted (optimizer-matched control) |
| d_one_step | param_delta_runtime_vs_source_adamw_rel_l2 | 0.939777 | 0.93962 | 0.00235392 | info | \|\|delta_runtime - delta_source\|\| / \|\|delta_source\|\| |
| e_budget | test_acc_300steps_bs64_source_native | 0.9738 |  |  | info | Adadelta(lr=1.0), source train(); MNIST test 10k; 325s |
| e_budget | test_acc_300steps_bs64_source_adamw | 0.9735 |  |  | info | source train() with AdamW(lr=0.001); 307s |
| e_budget | test_acc_300steps_bs64_converted_runtime |  | 0.9734 | -0.0001 | match | FLClient.local_train, AdamW(lr=0.001)+clip; delta vs optimizer-matched source; 336s |
| run | elapsed_sec |  |  | 981.9 | info |  |

### P2: `benchmarks/lightning/mnist_lite.py` -> `benchmarks/lightning/mnist_lite_fl_structured.py`

| stage | metric | source | converted | comparison | verdict | notes |
|---|---|---|---|---|---|---|
| a_preprocessing | dataset_class | MNIST | MNIST |  | match | class of the dataset actually instantiated (synthetic fallback would be TensorDataset) |
| a_preprocessing | converted_used_real_data |  | True |  | match | True when the converted loader wraps torchvision's real dataset |
| a_preprocessing | dataset_len_underlying | 60000 | 60000 |  | match | length of the underlying dataset |
| a_preprocessing | train_split_len | 55000 | 54000 |  | info | source: random_split([55000, 5000], global RNG, MNISTDataModule.setup); converted: 90/10 random_split(seed 42) |
| a_preprocessing | val_split_len | 5000 | 6000 |  | info |  |
| a_preprocessing | example_shape | (1, 28, 28) | (1, 28, 28) |  | match |  |
| a_preprocessing | example_dtype | torch.float32 | torch.float32 |  | match |  |
| a_preprocessing | value_min_first256 | 0 | 0 | 0 | match | over the first 256 examples |
| a_preprocessing | value_max_first256 | 1 | 1 | 0 | match | over the first 256 examples |
| a_preprocessing | value_mean_first256 | 0.128269 | 0.128269 | 0 | match | over the first 256 examples |
| a_preprocessing | value_std_first256 | 0.305231 | 0.305231 | 0 | match | over the first 256 examples |
| a_preprocessing | max_abs_pixel_diff_first256 |  |  | 0 | match | sample-aligned: same index in both datasets |
| a_preprocessing | labels_identical_first256 |  |  | 1 | match |  |
| a_preprocessing | label_set | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] |  | match |  |
| a_preprocessing | loader_batch_size_used |  | 64 |  | info | converted loader batch size from config['local']['batch_size'] |
| a_preprocessing | converted_train_shuffle |  | True |  | info |  |
| a_preprocessing | source_default_batch_size | 32 | 64 |  | info | MNISTDataModule default batch_size=32; both sides use bs for this test |
| b_architecture | param_count_generator | 1510032 | 1510032 | 0 | match |  |
| b_architecture | leaf_modules_generator | Linear>LeakyReLU>Linear>BatchNorm1d>LeakyReLU>Linear>BatchNorm1d>LeakyReLU>Linear>BatchNorm1d>LeakyReLU>Linear>Tanh | Linear>LeakyReLU>Linear>BatchNorm1d>LeakyReLU>Linear>BatchNorm1d>LeakyReLU>Linear>BatchNorm1d>LeakyReLU>Linear>Tanh |  | match | leaf nn.Module types in registration order |
| b_architecture | state_dict_keys_match_generator | 25 | 25 |  | match |  |
| b_architecture | state_dict_shapes_match_generator |  |  | True | match |  |
| b_architecture | param_count_discriminator | 533505 | 533505 | 0 | match |  |
| b_architecture | leaf_modules_discriminator | Linear>LeakyReLU>Linear>LeakyReLU>Linear | Linear>LeakyReLU>Linear>LeakyReLU>Linear |  | match | leaf nn.Module types in registration order |
| b_architecture | state_dict_keys_match_discriminator | 6 | 6 |  | match |  |
| b_architecture | state_dict_shapes_match_discriminator |  |  | True | match |  |
| b_architecture | state_dict_keys_match_full_model |  |  | True | match | GAN(LightningModule) vs GANModel(nn.Module) |
| c_loss_init | g_loss | 0.677422 | 0.677422 | 0 | match | BCE-with-logits(D(G(z)), 1) |
| c_loss_init | d_real_loss | 0.678824 | 0.678824 | 0 | match |  |
| c_loss_init | d_fake_loss | 0.709125 | 0.709125 | 0 | match |  |
| c_loss_init | d_loss_as_optimised | 0.693974 | 1.38795 | 0.693974 | differs | source optimises (real+fake)/2 with opt_d; converted sums real+fake (no /2) into the joint loss |
| c_loss_init | returned_total_loss | 2.06537 | 2.06537 | -1.19209e-07 | match | converted returns d_real+d_fake+g_loss; source never forms this scalar (compared to the same sum of source components) |
| c_loss_init | n_loss_calls_in_train_step | 2 | 3 |  | info | source: two backward targets (g_loss, d_loss); converted: three BCE terms in one scalar |
| d_one_step | grad_G_cosine |  |  | 1 | match | generator gradient: converted summed loss vs source g_loss (same z) |
| d_one_step | grad_D_cosine |  |  | 0.772316 | differs | discriminator gradient: converted summed loss vs source d_loss=(real+fake)/2 |
| d_one_step | grad_D_leak_cosine(gloss_vs_dfake) |  |  | -1 | opposite | cos(dL_g/dD, dL_fake/dD): the g_loss gradient that leaks into D vs the source's fake-branch D gradient |
| d_one_step | grad_D_fake_branch_norm_ratio | 0.589678 | 0.0184015 | 0.031206 | info | \|\|dL_fake/dD + dL_g/dD\|\| / \|\|dL_fake/dD\|\|: <1 means the fake-branch signal to D is cancelled in the summed loss |
| d_one_step | grad_D_leak_norm_over_source_norm | 0.344264 | 0.571276 | 1.65941 | info | \|\|dL_g/dD\|\| / \|\|d(source D loss)/dD\|\| |
| d_one_step | source_training_step_logged_d_loss | 0.695882 |  |  | info | d_loss logged by the source training_step (computed after the G update) |
| d_one_step | source_training_step_logged_g_loss | 0.677422 |  |  | info |  |
| d_one_step | G_param_delta_runtime_vs_source_native_cosine |  |  | 1 | match | source: opt_g=Adam(2e-4,(0.5,0.999)); runtime: single AdamW(2e-4)+clip over G and D |
| d_one_step | G_param_delta_runtime_vs_source_native_rel_l2 | 0.243978 | 0.243977 | 0.000431333 | info | \|\|delta_runtime - delta_source\|\| / \|\|delta_source\|\| |
| d_one_step | D_param_delta_runtime_vs_source_native_cosine |  |  | 0.86713 | differs | source: opt_d=Adam step on (real+fake)/2 only; runtime: joint step on real+fake+g_loss |
| d_one_step | D_param_delta_runtime_vs_source_native_rel_l2 | 0.145752 | 0.144231 | 0.512909 | info | \|\|delta_runtime - delta_source\|\| / \|\|delta_source\|\| |
| d_one_step | G_param_delta_runtime_vs_source_adamw_cosine |  |  | 1 | match | source training_step with two AdamW(2e-4) optimizers substituted |
| d_one_step | G_param_delta_runtime_vs_source_adamw_rel_l2 | 0.243977 | 0.243977 | 0 | info | \|\|delta_runtime - delta_source\|\| / \|\|delta_source\|\| |
| d_one_step | D_param_delta_runtime_vs_source_adamw_cosine |  |  | 0.867131 | differs |  |
| d_one_step | D_param_delta_runtime_vs_source_adamw_rel_l2 | 0.145752 | 0.144231 | 0.512909 | info | \|\|delta_runtime - delta_source\|\| / \|\|delta_source\|\| |
| e_budget | d_acc_real_300steps_bs64_source_native | 0.521 |  |  | info | two Adam(2e-4,(0.5,0.999)); 790s |
| e_budget | d_acc_fake_300steps_bs64_source_native | 0.985 |  |  | info | two Adam(2e-4,(0.5,0.999)); 790s |
| e_budget | d_acc_mean_300steps_bs64_source_native | 0.753 |  |  | info | two Adam(2e-4,(0.5,0.999)); 790s |
| e_budget | d_prob_real_mean_300steps_bs64_source_native | 0.502469 |  |  | info | two Adam(2e-4,(0.5,0.999)); 790s |
| e_budget | d_prob_fake_mean_300steps_bs64_source_native | 0.328954 |  |  | info | two Adam(2e-4,(0.5,0.999)); 790s |
| e_budget | g_out_mean_300steps_bs64_source_native | 0.131757 |  |  | info | two Adam(2e-4,(0.5,0.999)); 790s |
| e_budget | g_out_std_300steps_bs64_source_native | 0.298679 |  |  | info | two Adam(2e-4,(0.5,0.999)); 790s |
| e_budget | d_acc_real_bn_batchstats_300steps_bs64_source_native | 0.521 |  |  | info | two Adam(2e-4,(0.5,0.999)); 790s |
| e_budget | d_acc_fake_bn_batchstats_300steps_bs64_source_native | 0.977 |  |  | info | two Adam(2e-4,(0.5,0.999)); 790s |
| e_budget | d_acc_mean_bn_batchstats_300steps_bs64_source_native | 0.749 |  |  | info | two Adam(2e-4,(0.5,0.999)); 790s |
| e_budget | d_prob_real_mean_bn_batchstats_300steps_bs64_source_native | 0.502469 |  |  | info | two Adam(2e-4,(0.5,0.999)); 790s |
| e_budget | d_prob_fake_mean_bn_batchstats_300steps_bs64_source_native | 0.340875 |  |  | info | two Adam(2e-4,(0.5,0.999)); 790s |
| e_budget | g_out_mean_bn_batchstats_300steps_bs64_source_native | 0.124339 |  |  | info | two Adam(2e-4,(0.5,0.999)); 790s |
| e_budget | g_out_std_bn_batchstats_300steps_bs64_source_native | 0.289708 |  |  | info | two Adam(2e-4,(0.5,0.999)); 790s |
| e_budget | d_acc_real_300steps_bs64_source_adamw | 0.757 |  |  | info | two AdamW(2e-4); 425s |
| e_budget | d_acc_fake_300steps_bs64_source_adamw | 1 |  |  | info | two AdamW(2e-4); 425s |
| e_budget | d_acc_mean_300steps_bs64_source_adamw | 0.8785 |  |  | info | two AdamW(2e-4); 425s |
| e_budget | d_prob_real_mean_300steps_bs64_source_adamw | 0.663319 |  |  | info | two AdamW(2e-4); 425s |
| e_budget | d_prob_fake_mean_300steps_bs64_source_adamw | 0.257012 |  |  | info | two AdamW(2e-4); 425s |
| e_budget | g_out_mean_300steps_bs64_source_adamw | 0.0672221 |  |  | info | two AdamW(2e-4); 425s |
| e_budget | g_out_std_300steps_bs64_source_adamw | 0.147519 |  |  | info | two AdamW(2e-4); 425s |
| e_budget | d_acc_real_bn_batchstats_300steps_bs64_source_adamw | 0.757 |  |  | info | two AdamW(2e-4); 425s |
| e_budget | d_acc_fake_bn_batchstats_300steps_bs64_source_adamw | 1 |  |  | info | two AdamW(2e-4); 425s |
| e_budget | d_acc_mean_bn_batchstats_300steps_bs64_source_adamw | 0.8785 |  |  | info | two AdamW(2e-4); 425s |
| e_budget | d_prob_real_mean_bn_batchstats_300steps_bs64_source_adamw | 0.663319 |  |  | info | two AdamW(2e-4); 425s |
| e_budget | d_prob_fake_mean_bn_batchstats_300steps_bs64_source_adamw | 0.298741 |  |  | info | two AdamW(2e-4); 425s |
| e_budget | g_out_mean_bn_batchstats_300steps_bs64_source_adamw | 0.0714103 |  |  | info | two AdamW(2e-4); 425s |
| e_budget | g_out_std_bn_batchstats_300steps_bs64_source_adamw | 0.158788 |  |  | info | two AdamW(2e-4); 425s |
| e_budget | d_acc_real_300steps_bs64_converted_runtime |  | 0.778 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.825323; 38s |
| e_budget | d_acc_fake_300steps_bs64_converted_runtime |  | 0.171 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.825323; 38s |
| e_budget | d_acc_mean_300steps_bs64_converted_runtime |  | 0.4745 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.825323; 38s |
| e_budget | d_prob_real_mean_300steps_bs64_converted_runtime |  | 0.594812 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.825323; 38s |
| e_budget | d_prob_fake_mean_300steps_bs64_converted_runtime |  | 0.620125 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.825323; 38s |
| e_budget | g_out_mean_300steps_bs64_converted_runtime |  | 0.313348 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.825323; 38s |
| e_budget | g_out_std_300steps_bs64_converted_runtime |  | 0.463163 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.825323; 38s |
| e_budget | d_acc_real_bn_batchstats_300steps_bs64_converted_runtime |  | 0.778 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.825323; 38s |
| e_budget | d_acc_fake_bn_batchstats_300steps_bs64_converted_runtime |  | 0.049 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.825323; 38s |
| e_budget | d_acc_mean_bn_batchstats_300steps_bs64_converted_runtime |  | 0.4135 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.825323; 38s |
| e_budget | d_prob_real_mean_bn_batchstats_300steps_bs64_converted_runtime |  | 0.594812 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.825323; 38s |
| e_budget | d_prob_fake_mean_bn_batchstats_300steps_bs64_converted_runtime |  | 0.737938 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.825323; 38s |
| e_budget | g_out_mean_bn_batchstats_300steps_bs64_converted_runtime |  | 0.332034 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.825323; 38s |
| e_budget | g_out_std_bn_batchstats_300steps_bs64_converted_runtime |  | 0.481105 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.825323; 38s |
| e_budget | final_losses_300steps_source_native | d=0.6734,g=1.1174 |  |  | info | last logged d_loss/g_loss of the source training_step |
| e_budget | d_acc_mean_300steps_delta | 0.8785 | 0.4745 | -0.404 | differs | discriminator real/fake accuracy on 1000 MNIST test images + 1000 samples: converted-runtime minus optimizer-matched source |
| e_budget | d_acc_mean_bn_batchstats_300steps_delta | 0.8785 | 0.4135 | -0.465 | differs | discriminator real/fake accuracy on 1000 MNIST test images + 1000 samples: converted-runtime minus optimizer-matched source |
| e_budget | d_acc_fake_bn_batchstats_300steps_delta | 1 | 0.049 | -0.951 | differs | discriminator real/fake accuracy on 1000 MNIST test images + 1000 samples: converted-runtime minus optimizer-matched source |
| run | elapsed_sec |  |  | 1274.5 | info |  |

### P3: `benchmarks/lightning/backbone_image_classifier.py` -> `benchmarks/lightning/backbone_image_classifier_fl_structured.py`

| stage | metric | source | converted | comparison | verdict | notes |
|---|---|---|---|---|---|---|
| a_preprocessing | dataset_class | MNIST | MNIST |  | match | class of the dataset actually instantiated (synthetic fallback would be TensorDataset) |
| a_preprocessing | converted_used_real_data |  | True |  | match | True when the converted loader wraps torchvision's real dataset |
| a_preprocessing | dataset_len_underlying | 60000 | 60000 |  | match | length of the underlying dataset |
| a_preprocessing | train_split_len | 55000 | 54000 |  | info | source: random_split([55000, 5000], seed 42); converted: 90/10 random_split(seed 42) |
| a_preprocessing | val_split_len | 5000 | 6000 |  | info |  |
| a_preprocessing | example_shape | (1, 28, 28) | (1, 28, 28) |  | match |  |
| a_preprocessing | example_dtype | torch.float32 | torch.float32 |  | match |  |
| a_preprocessing | value_min_first256 | 0 | 0 | 0 | match | over the first 256 examples |
| a_preprocessing | value_max_first256 | 1 | 1 | 0 | match | over the first 256 examples |
| a_preprocessing | value_mean_first256 | 0.128269 | 0.128269 | 0 | match | over the first 256 examples |
| a_preprocessing | value_std_first256 | 0.305231 | 0.305231 | 0 | match | over the first 256 examples |
| a_preprocessing | max_abs_pixel_diff_first256 |  |  | 0 | match | sample-aligned: same index in both datasets |
| a_preprocessing | labels_identical_first256 |  |  | 1 | match |  |
| a_preprocessing | label_set | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] |  | match |  |
| a_preprocessing | loader_batch_size_used |  | 64 |  | info | converted loader batch size from config['local']['batch_size'] |
| a_preprocessing | converted_train_shuffle |  | True |  | info |  |
| a_preprocessing | source_train_shuffle | False |  |  | info | MyDataModule.train_dataloader() passes no shuffle=True; converted shuffles |
| b_architecture | param_count | 101770 | 101770 | 0 | match | source Backbone is LitClassifier.backbone (state_dict keys carry 'backbone.' prefix at LightningModule level) |
| b_architecture | leaf_modules | Linear>Linear | Linear>Linear |  | match | leaf nn.Module types in registration order |
| b_architecture | state_dict_keys_match | 4 | 4 |  | match |  |
| b_architecture | state_dict_shapes_match |  |  | True | match |  |
| c_loss_init | loss_train_mode | 2.30735 | 2.30735 | 0 | match | LitClassifier.training_step vs converted train_step, identical params |
| d_one_step | grad_cosine |  |  | 1 | match |  |
| d_one_step | param_delta_runtime_vs_source_native_cosine |  |  | 1 | match | source: configure_optimizers()=Adam(lr=1e-4); runtime: AdamW(lr=1e-4)+clip(1.0) |
| d_one_step | param_delta_runtime_vs_source_native_rel_l2 | 0.0246816 | 0.0246816 | 0.000279637 | info | \|\|delta_runtime - delta_source\|\| / \|\|delta_source\|\| |
| d_one_step | param_delta_runtime_vs_source_adamw_cosine |  |  | 1 | match | source loop with AdamW(lr=1e-4) substituted |
| d_one_step | param_delta_runtime_vs_source_adamw_rel_l2 | 0.0246816 | 0.0246816 | 0 | info | \|\|delta_runtime - delta_source\|\| / \|\|delta_source\|\| |
| e_budget | test_acc_300steps_bs64_source_native | 0.7729 |  |  | info | Adam(lr=0.0001); source loader (unshuffled 55000-subset); MNIST test 10k; 27s |
| e_budget | test_acc_300steps_bs64_source_adamw | 0.7729 |  |  | info | source loop with AdamW(lr=0.0001); source loader; 26s |
| e_budget | test_acc_300steps_bs64_source_adamw_on_converted_loader | 0.8319 |  |  | info | source training_step + AdamW fed by the converted build_dataloader (same seed, same batches as the runtime arm) |
| e_budget | test_acc_300steps_bs64_converted_runtime |  | 0.8312 | -0.0007 | match | FLClient.local_train, AdamW(lr=0.0001)+clip; delta vs the same-batches source control (delta vs source_adamw on its own loader: +0.0583); 26s |
| run | elapsed_sec |  |  | 108.5 | info |  |

### P4: `benchmarks/pytorch/dcgan_main.py` -> `benchmarks/pytorch/dcgan_main_fl_structured.py`

| stage | metric | source | converted | comparison | verdict | notes |
|---|---|---|---|---|---|---|
| a_preprocessing | dataset_class | CIFAR10 | CIFAR10 |  | match | class of the dataset actually instantiated (synthetic fallback would be TensorDataset) |
| a_preprocessing | converted_used_real_data |  | True |  | match | True when the converted loader wraps torchvision's real dataset |
| a_preprocessing | dataset_len_underlying | 50000 | 50000 |  | match | length of the underlying dataset |
| a_preprocessing | train_split_len | 50000 | 45000 |  | info | source: no split (50000 train); converted: 90/10 random_split(seed 42) |
| a_preprocessing | val_split_len | - | 5000 |  | info |  |
| a_preprocessing | example_shape | (3, 64, 64) | (3, 64, 64) |  | match |  |
| a_preprocessing | example_dtype | torch.float32 | torch.float32 |  | match |  |
| a_preprocessing | value_min_first256 | -1 | -1 | 0 | match | over the first 256 examples |
| a_preprocessing | value_max_first256 | 1 | 1 | 0 | match | over the first 256 examples |
| a_preprocessing | value_mean_first256 | -0.0627353 | -0.0627353 | 0 | match | over the first 256 examples |
| a_preprocessing | value_std_first256 | 0.479676 | 0.479676 | 0 | match | over the first 256 examples |
| a_preprocessing | max_abs_pixel_diff_first256 |  |  | 0 | match | sample-aligned: same index in both datasets |
| a_preprocessing | labels_identical_first256 |  |  | 1 | match |  |
| a_preprocessing | label_set | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] |  | match |  |
| a_preprocessing | loader_batch_size_used |  | 64 |  | info | converted loader batch size from config['local']['batch_size'] |
| a_preprocessing | converted_train_shuffle |  | True |  | info |  |
| a_preprocessing | dataset_choice | cifar10 (source supports it natively) | cifar10 (converted default) |  | info | no ImageFolder needed; converted adds drop_last=True |
| b_architecture | param_count_generator | 3576704 | 3576704 | 0 | match |  |
| b_architecture | leaf_modules_generator | ConvTranspose2d>BatchNorm2d>ReLU>ConvTranspose2d>BatchNorm2d>ReLU>ConvTranspose2d>BatchNorm2d>ReLU>ConvTranspose2d>BatchNorm2d>ReLU>ConvTranspose2d>Tanh | ConvTranspose2d>BatchNorm2d>ReLU>ConvTranspose2d>BatchNorm2d>ReLU>ConvTranspose2d>BatchNorm2d>ReLU>ConvTranspose2d>BatchNorm2d>ReLU>ConvTranspose2d>Tanh |  | match | leaf nn.Module types in registration order |
| b_architecture | state_dict_keys_match_generator | 25 | 25 |  | match |  |
| b_architecture | state_dict_shapes_match_generator |  |  | True | match |  |
| b_architecture | param_count_discriminator | 2765568 | 2765568 | 0 | match |  |
| b_architecture | leaf_modules_discriminator | Conv2d>LeakyReLU>Conv2d>BatchNorm2d>LeakyReLU>Conv2d>BatchNorm2d>LeakyReLU>Conv2d>BatchNorm2d>LeakyReLU>Conv2d>Sigmoid | Conv2d>LeakyReLU>Conv2d>BatchNorm2d>LeakyReLU>Conv2d>BatchNorm2d>LeakyReLU>Conv2d>BatchNorm2d>LeakyReLU>Conv2d>Sigmoid |  | match | leaf nn.Module types in registration order |
| b_architecture | state_dict_keys_match_discriminator | 20 | 20 |  | match |  |
| b_architecture | state_dict_shapes_match_discriminator |  |  | True | match |  |
| c_loss_init | g_loss | 0.77508 | 0.77508 | 0 | match | BCELoss(D(G(z)), 1) on sigmoid outputs |
| c_loss_init | d_real_loss | 0.841747 | 0.841747 | 0 | match |  |
| c_loss_init | d_fake_loss | 0.910146 | 0.910146 | 0 | match |  |
| c_loss_init | d_loss_as_optimised | 1.75189 | 1.75189 | 0 | match | source errD = errD_real + errD_fake (optimizerD only); converted folds it into the joint loss |
| c_loss_init | returned_total_loss | 2.52697 | 2.52697 | 0 | match | converted returns errD + errG; source never forms this scalar |
| c_loss_init | n_loss_calls_in_train_step | 3 | 3 |  | info | source: three criterion calls but two separate optimizer steps |
| d_one_step | grad_G_cosine |  |  | 1 | match | generator gradient: converted summed loss vs source g_loss (same z) |
| d_one_step | grad_D_cosine |  |  | 0.447846 | differs | discriminator gradient: converted summed loss vs source errD=errD_real+errD_fake |
| d_one_step | grad_D_leak_cosine(gloss_vs_dfake) |  |  | -0.943443 | opposite | cos(dL_g/dD, dL_fake/dD): the g_loss gradient that leaks into D vs the source's fake-branch D gradient |
| d_one_step | grad_D_fake_branch_norm_ratio | 29.4138 | 9.87374 | 0.335684 | info | \|\|dL_fake/dD + dL_g/dD\|\| / \|\|dL_fake/dD\|\|: <1 means the fake-branch signal to D is cancelled in the summed loss |
| d_one_step | grad_D_leak_norm_over_source_norm | 26.7517 | 26.2024 | 0.979468 | info | \|\|dL_g/dD\|\| / \|\|d(source D loss)/dD\|\| |
| d_one_step | source_loop_losses | errD=1.7519,errG=6.0228 |  |  | info | errG is evaluated with the already-updated D (source ordering: D step, then G step) |
| d_one_step | G_param_delta_runtime_vs_source_native_cosine |  |  | 0.647153 | differs | source: optimizerG=Adam(2e-4,(0.5,0.999)); runtime: single AdamW(2e-4)+clip over G and D |
| d_one_step | G_param_delta_runtime_vs_source_native_rel_l2 | 0.378241 | 0.378124 | 0.839926 | info | \|\|delta_runtime - delta_source\|\| / \|\|delta_source\|\| |
| d_one_step | D_param_delta_runtime_vs_source_native_cosine |  |  | 0.350506 | differs | source: optimizerD step on errD only; runtime: joint step on errD+errG |
| d_one_step | D_param_delta_runtime_vs_source_native_rel_l2 | 0.332595 | 0.332473 | 1.13952 | info | \|\|delta_runtime - delta_source\|\| / \|\|delta_source\|\| |
| d_one_step | G_param_delta_runtime_vs_source_adamw_cosine |  |  | 0.64722 | differs | two AdamW(2e-4) substituted in the source loop |
| d_one_step | G_param_delta_runtime_vs_source_adamw_rel_l2 | 0.378241 | 0.378124 | 0.839846 | info | \|\|delta_runtime - delta_source\|\| / \|\|delta_source\|\| |
| d_one_step | D_param_delta_runtime_vs_source_adamw_cosine |  |  | 0.350506 | differs |  |
| d_one_step | D_param_delta_runtime_vs_source_adamw_rel_l2 | 0.332596 | 0.332473 | 1.13952 | info | \|\|delta_runtime - delta_source\|\| / \|\|delta_source\|\| |
| e_budget | d_acc_real_300steps_bs64_source_native | 0.333984 |  |  | info | two Adam(2e-4,(0.5,0.999)); 12s |
| e_budget | d_acc_fake_300steps_bs64_source_native | 0.642578 |  |  | info | two Adam(2e-4,(0.5,0.999)); 12s |
| e_budget | d_acc_mean_300steps_bs64_source_native | 0.488281 |  |  | info | two Adam(2e-4,(0.5,0.999)); 12s |
| e_budget | d_prob_real_mean_300steps_bs64_source_native | 0.364387 |  |  | info | two Adam(2e-4,(0.5,0.999)); 12s |
| e_budget | d_prob_fake_mean_300steps_bs64_source_native | 0.426447 |  |  | info | two Adam(2e-4,(0.5,0.999)); 12s |
| e_budget | g_out_mean_300steps_bs64_source_native | -0.230363 |  |  | info | two Adam(2e-4,(0.5,0.999)); 12s |
| e_budget | g_out_std_300steps_bs64_source_native | 0.25657 |  |  | info | two Adam(2e-4,(0.5,0.999)); 12s |
| e_budget | d_acc_real_bn_batchstats_300steps_bs64_source_native | 0.982422 |  |  | info | two Adam(2e-4,(0.5,0.999)); 12s |
| e_budget | d_acc_fake_bn_batchstats_300steps_bs64_source_native | 1 |  |  | info | two Adam(2e-4,(0.5,0.999)); 12s |
| e_budget | d_acc_mean_bn_batchstats_300steps_bs64_source_native | 0.991211 |  |  | info | two Adam(2e-4,(0.5,0.999)); 12s |
| e_budget | d_prob_real_mean_bn_batchstats_300steps_bs64_source_native | 0.955161 |  |  | info | two Adam(2e-4,(0.5,0.999)); 12s |
| e_budget | d_prob_fake_mean_bn_batchstats_300steps_bs64_source_native | 0.0498065 |  |  | info | two Adam(2e-4,(0.5,0.999)); 12s |
| e_budget | g_out_mean_bn_batchstats_300steps_bs64_source_native | -0.106925 |  |  | info | two Adam(2e-4,(0.5,0.999)); 12s |
| e_budget | g_out_std_bn_batchstats_300steps_bs64_source_native | 0.247802 |  |  | info | two Adam(2e-4,(0.5,0.999)); 12s |
| e_budget | d_acc_real_300steps_bs64_source_adamw | 0.347656 |  |  | info | two AdamW(2e-4); 15s |
| e_budget | d_acc_fake_300steps_bs64_source_adamw | 1 |  |  | info | two AdamW(2e-4); 15s |
| e_budget | d_acc_mean_300steps_bs64_source_adamw | 0.673828 |  |  | info | two AdamW(2e-4); 15s |
| e_budget | d_prob_real_mean_300steps_bs64_source_adamw | 0.369913 |  |  | info | two AdamW(2e-4); 15s |
| e_budget | d_prob_fake_mean_300steps_bs64_source_adamw | 0.0227452 |  |  | info | two AdamW(2e-4); 15s |
| e_budget | g_out_mean_300steps_bs64_source_adamw | -0.0697408 |  |  | info | two AdamW(2e-4); 15s |
| e_budget | g_out_std_300steps_bs64_source_adamw | 0.376477 |  |  | info | two AdamW(2e-4); 15s |
| e_budget | d_acc_real_bn_batchstats_300steps_bs64_source_adamw | 0.994141 |  |  | info | two AdamW(2e-4); 15s |
| e_budget | d_acc_fake_bn_batchstats_300steps_bs64_source_adamw | 1 |  |  | info | two AdamW(2e-4); 15s |
| e_budget | d_acc_mean_bn_batchstats_300steps_bs64_source_adamw | 0.99707 |  |  | info | two AdamW(2e-4); 15s |
| e_budget | d_prob_real_mean_bn_batchstats_300steps_bs64_source_adamw | 0.967621 |  |  | info | two AdamW(2e-4); 15s |
| e_budget | d_prob_fake_mean_bn_batchstats_300steps_bs64_source_adamw | 0.00123269 |  |  | info | two AdamW(2e-4); 15s |
| e_budget | g_out_mean_bn_batchstats_300steps_bs64_source_adamw | -0.0463706 |  |  | info | two AdamW(2e-4); 15s |
| e_budget | g_out_std_bn_batchstats_300steps_bs64_source_adamw | 0.388483 |  |  | info | two AdamW(2e-4); 15s |
| e_budget | d_acc_real_300steps_bs64_converted_runtime |  | 0.369141 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.961541; 10s |
| e_budget | d_acc_fake_300steps_bs64_converted_runtime |  | 0 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.961541; 10s |
| e_budget | d_acc_mean_300steps_bs64_converted_runtime |  | 0.18457 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.961541; 10s |
| e_budget | d_prob_real_mean_300steps_bs64_converted_runtime |  | 0.383521 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.961541; 10s |
| e_budget | d_prob_fake_mean_300steps_bs64_converted_runtime |  | 0.947643 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.961541; 10s |
| e_budget | g_out_mean_300steps_bs64_converted_runtime |  | -0.00134568 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.961541; 10s |
| e_budget | g_out_std_300steps_bs64_converted_runtime |  | 0.267992 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.961541; 10s |
| e_budget | d_acc_real_bn_batchstats_300steps_bs64_converted_runtime |  | 0.998047 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.961541; 10s |
| e_budget | d_acc_fake_bn_batchstats_300steps_bs64_converted_runtime |  | 0 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.961541; 10s |
| e_budget | d_acc_mean_bn_batchstats_300steps_bs64_converted_runtime |  | 0.499023 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.961541; 10s |
| e_budget | d_prob_real_mean_bn_batchstats_300steps_bs64_converted_runtime |  | 0.961889 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.961541; 10s |
| e_budget | d_prob_fake_mean_bn_batchstats_300steps_bs64_converted_runtime |  | 0.764514 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.961541; 10s |
| e_budget | g_out_mean_bn_batchstats_300steps_bs64_converted_runtime |  | -0.00545853 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.961541; 10s |
| e_budget | g_out_std_bn_batchstats_300steps_bs64_converted_runtime |  | 0.266156 |  | info | single AdamW(2e-4)+clip on summed loss; mean runtime loss 1.961541; 10s |
| e_budget | final_losses_300steps_source_native | errD=0.0628,errG=6.2599 |  |  | info |  |
| e_budget | d_acc_mean_300steps_delta | 0.673828 | 0.18457 | -0.489258 | differs | discriminator real/fake accuracy on 512 CIFAR-10 test images + 512 samples: converted-runtime minus optimizer-matched source |
| e_budget | d_acc_mean_bn_batchstats_300steps_delta | 0.99707 | 0.499023 | -0.498047 | differs | discriminator real/fake accuracy on 512 CIFAR-10 test images + 512 samples: converted-runtime minus optimizer-matched source |
| e_budget | d_acc_fake_bn_batchstats_300steps_delta | 1 | 0 | -1 | differs | discriminator real/fake accuracy on 512 CIFAR-10 test images + 512 samples: converted-runtime minus optimizer-matched source |
| run | elapsed_sec |  |  | 45.8 | info |  |

### P5: `benchmarks/sklearn/mlp_digits.py` -> `benchmarks/sklearn/mlp_digits_fl_structured.py`

| stage | metric | source | converted | comparison | verdict | notes |
|---|---|---|---|---|---|---|
| a_preprocessing | dataset_class | ndarray (sklearn loader) | TensorDataset |  | info | converted wraps the sklearn arrays in a TensorDataset |
| a_preprocessing | converted_used_real_data |  | True |  | match | converted tensors equal the re-scaled sklearn arrays (not the randn fallback) |
| a_preprocessing | dataset_len_underlying | 1797 | 1797 |  | match |  |
| a_preprocessing | scaler_fit_population | train split (1437) | all rows (1797) |  | differs | converted fits StandardScaler on all 1797 rows before splitting (val rows leak into the scaler) |
| a_preprocessing | max_abs_feature_diff_sample_aligned |  |  | 21.5778 | differs | source scaler applied to every row vs the converted tensor, same row index |
| a_preprocessing | mean_abs_feature_diff_sample_aligned |  |  | 0.00919517 | info |  |
| a_preprocessing | train_split_len | 1437 | 1438 |  | differs | source: stratified train_test_split(0.2, seed 42); converted: random_split |
| a_preprocessing | val_split_len | 360 | 359 |  | differs |  |
| a_preprocessing | heldout_set_overlap_jaccard |  |  | 0.104455 | differs | source test split vs converted val split (row indices) |
| a_preprocessing | split_stratified | True | False |  | differs |  |
| a_preprocessing | value_min_first256 | -3.00994 | -3.0126 | -0.00265741 | info | first 256 rows of each pipeline's own train split (not sample-aligned) |
| a_preprocessing | value_max_first256 | 37.8946 | 21.1734 | -16.7212 | info | first 256 rows of each pipeline's own train split (not sample-aligned) |
| a_preprocessing | value_mean_first256 | -0.0109765 | -0.00556348 | 0.00541304 | info | first 256 rows of each pipeline's own train split (not sample-aligned) |
| a_preprocessing | value_std_first256 | 0.92943 | 0.928935 | -0.000495017 | info | first 256 rows of each pipeline's own train split (not sample-aligned) |
| a_preprocessing | example_shape | (64,) | (64,) |  | match |  |
| a_preprocessing | example_dtype | float64 (numpy) | torch.float32 |  | info |  |
| a_preprocessing | label_set | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] |  | match |  |
| b_architecture | param_count | 17226 | 17226 | 0 | match | sklearn coefs_+intercepts_ vs torch parameters |
| b_architecture | layer_weight_shapes | [(64, 128), (128, 64), (64, 10)] | [(128, 64), (64, 128), (10, 64)] |  | match | sklearn stores (in,out); torch stores (out,in): layer-wise transposes |
| b_architecture | leaf_modules | Linear>relu>Linear>relu>Linear>softmax/log_loss | Linear>ReLU>Linear>ReLU>Linear + cross_entropy |  | match | same topology; softmax+log-loss == logits+cross-entropy |
| b_architecture | state_dict_keys_match | coefs_[0..2], intercepts_[0..2] | network.0.weight,network.0.bias,network.2.weight,network.2.bias,network.4.weight,network.4.bias |  | n/a | no state_dict on the sklearn side; correspondence established layer-wise |
| b_architecture | l2_regularisation | alpha=1e-4 (in objective) | none (AdamW weight_decay=0.01 in runtime) |  | differs | sklearn adds 0.5*alpha*\|\|W\|\|^2/n to the loss; converted train_step has no penalty |
| c_loss_init | log_loss | 2.30031 | 2.30031 | 2.9207e-08 | match | sklearn softmax log-loss vs torch cross_entropy, identical parameters, same 64-row batch |
| c_loss_init | objective_with_l2 | 2.30036 | 2.30031 | -5.25841e-05 | match | sklearn _backprop objective includes 0.5*alpha*\|\|W\|\|^2/n |
| d_one_step | grad_cosine |  |  | 1 | match | sklearn _backprop gradient (incl. L2 term) vs torch autograd of train_step |
| d_one_step | param_delta_runtime_vs_source_native_cosine |  |  | 0.999102 | match | source: sklearn AdamOptimizer(lr=1e-3) via partial_fit; runtime: AdamW(lr=1e-3)+clip(1.0) |
| d_one_step | param_delta_runtime_vs_source_native_rel_l2 | 0.12938 | 0.129451 | 0.0423946 | info | \|\|delta_runtime - delta_source\|\| / \|\|delta_source\|\| |
| d_one_step | param_delta_runtime_vs_torch_adam_cosine |  |  | 1 | match | torch.optim.Adam(lr=1e-3) on the converted train_step without clipping (optimizer-type control) |
| d_one_step | param_delta_runtime_vs_torch_adam_rel_l2 | 0.129452 | 0.129451 | 0.000639207 | info | \|\|delta_runtime - delta_source\|\| / \|\|delta_source\|\| |
| d_one_step | param_delta_torch_adam_vs_source_native_cosine |  |  | 0.999098 | match | torch Adam vs sklearn Adam from identical parameters/batch |
| d_one_step | param_delta_torch_adam_vs_source_native_rel_l2 | 0.12938 | 0.129452 | 0.0425007 | info | \|\|delta_runtime - delta_source\|\| / \|\|delta_source\|\| |
| e_budget | test_acc_source_native | 0.966667 |  |  | info | sklearn MLPClassifier as in the script (early stopping); n_iter_=39 epochs, batch 200; 0s |
| e_budget | test_acc_converted_runtime_39epochs_source_split |  | 0.977778 | 0.0111111 | match | torch port trained by FLClient.local_train on the source's scaled train split for the same 39 epochs (bs 64); 1s |
| e_budget | test_acc_converted_runtime_300steps_source_split |  | 0.969444 | 0.00277778 | match | fixed 300-step budget; 0s |
| e_budget | val_acc_converted_runtime_39epochs_own_split |  | 0.966574 | -9.28505e-05 | info | converted pipeline end-to-end (its own random 80/20 split, whole-data scaler); 0s |
| run | elapsed_sec |  |  | 1.5 | info |  |

### P6: `benchmarks/sklearn/svm_iris.py` -> `benchmarks/sklearn/svm_iris_fl_structured.py`

| stage | metric | source | converted | comparison | verdict | notes |
|---|---|---|---|---|---|---|
| a_preprocessing | dataset_class | ndarray (sklearn loader) | TensorDataset |  | info | converted wraps the sklearn arrays in a TensorDataset |
| a_preprocessing | converted_used_real_data |  | True |  | match | converted tensors equal the re-scaled sklearn arrays (not the randn fallback) |
| a_preprocessing | dataset_len_underlying | 150 | 150 |  | match |  |
| a_preprocessing | scaler_fit_population | train split (120) | all rows (150) |  | differs | converted standardises all 150 rows with torch .std() (ddof=1); source uses sklearn StandardScaler (ddof=0) on the train split |
| a_preprocessing | max_abs_feature_diff_sample_aligned |  |  | 0.0787234 | differs | source scaler applied to every row vs the converted tensor, same row index |
| a_preprocessing | mean_abs_feature_diff_sample_aligned |  |  | 0.0124968 | info |  |
| a_preprocessing | train_split_len | 120 | 120 |  | match | source: stratified train_test_split(0.2, seed 42); converted: random_split |
| a_preprocessing | val_split_len | 30 | 30 |  | match |  |
| a_preprocessing | heldout_set_overlap_jaccard |  |  | 0.0714286 | differs | source test split vs converted val split (row indices) |
| a_preprocessing | split_stratified | True | False |  | differs |  |
| a_preprocessing | value_min_first256 | -2.3471 | -1.96696 | 0.380133 | info | first 256 rows of each pipeline's own train split (not sample-aligned) |
| a_preprocessing | value_max_first256 | 3.02622 | 3.08046 | 0.0542309 | info | first 256 rows of each pipeline's own train split (not sample-aligned) |
| a_preprocessing | value_mean_first256 | -5.96046e-09 | -0.0603102 | -0.0603102 | info | first 256 rows of each pipeline's own train split (not sample-aligned) |
| a_preprocessing | value_std_first256 | 1.00104 | 0.993425 | -0.00761819 | info | first 256 rows of each pipeline's own train split (not sample-aligned) |
| a_preprocessing | example_shape | (4,) | (4,) |  | match |  |
| a_preprocessing | example_dtype | float64 (numpy) | torch.float32 |  | info |  |
| a_preprocessing | label_set | [0, 1, 2] | [0, 1, 2] |  | match |  |
| b_architecture | model_family | sklearn.svm.SVC(rbf) | SVMClassifier (MLP 4-64-64-3 with BatchNorm) |  | surrogate | no parameter correspondence by design |
| b_architecture | param_count | 285 | 4931 |  | surrogate | SVC: 47 support vectors x4 features + dual_coef_ (2, 47) + intercepts; MLP: trainable tensors |
| b_architecture | leaf_modules | kernel QP (no layers) | Linear>BatchNorm1d>ReLU>Linear>BatchNorm1d>ReLU>Linear |  | surrogate |  |
| c_loss_init | loss_at_init |  |  |  | n/a | SVC has no per-batch differentiable loss or gradient step (dual QP solved by libsvm) |
| d_one_step | grad_cosine |  |  |  | n/a | SVC has no per-batch differentiable loss or gradient step (dual QP solved by libsvm) |
| d_one_step | param_delta_cosine |  |  |  | n/a | SVC has no per-batch differentiable loss or gradient step (dual QP solved by libsvm) |
| e_budget | test_acc_source_native | 0.966667 |  |  | info | SVC as in the script, source test split (30 rows) |
| e_budget | test_acc_converted_runtime_300steps_source_split |  | 0.966667 | 0 | match | surrogate MLP trained by FLClient.local_train on the source train split, AdamW(lr=0.001); 0s |
| run | elapsed_sec |  |  | 0.3 | info |  |

### P7: `benchmarks/xgboost/xgb_breast_cancer.py` -> `benchmarks/xgboost/xgb_breast_cancer_fl_structured.py`

| stage | metric | source | converted | comparison | verdict | notes |
|---|---|---|---|---|---|---|
| a_preprocessing | dataset_class | ndarray (sklearn loader) | TensorDataset |  | info | converted wraps the sklearn arrays in a TensorDataset |
| a_preprocessing | converted_used_real_data |  | True |  | match | converted tensors equal the re-scaled sklearn arrays (not the randn fallback) |
| a_preprocessing | dataset_len_underlying | 569 | 569 |  | match |  |
| a_preprocessing | scaler_fit_population | train split (455) | all rows (569) |  | differs | converted fits StandardScaler on all 569 rows before splitting |
| a_preprocessing | max_abs_feature_diff_sample_aligned |  |  | 1.01446 | differs | source scaler applied to every row vs the converted tensor, same row index |
| a_preprocessing | mean_abs_feature_diff_sample_aligned |  |  | 0.0226733 | info |  |
| a_preprocessing | train_split_len | 455 | 456 |  | differs | source: stratified train_test_split(0.2, seed 42); converted: random_split |
| a_preprocessing | val_split_len | 114 | 113 |  | differs |  |
| a_preprocessing | heldout_set_overlap_jaccard |  |  | 0.129353 | differs | source test split vs converted val split (row indices) |
| a_preprocessing | split_stratified | True | False |  | differs |  |
| a_preprocessing | value_min_first256 | -2.71511 | -2.74412 | -0.0290096 | info | first 256 rows of each pipeline's own train split (not sample-aligned) |
| a_preprocessing | value_max_first256 | 11.6584 | 11.0418 | -0.616547 | info | first 256 rows of each pipeline's own train split (not sample-aligned) |
| a_preprocessing | value_mean_first256 | 0.0880783 | -0.0125075 | -0.100586 | info | first 256 rows of each pipeline's own train split (not sample-aligned) |
| a_preprocessing | value_std_first256 | 1.06651 | 1.00489 | -0.0616195 | info | first 256 rows of each pipeline's own train split (not sample-aligned) |
| a_preprocessing | example_shape | (30,) | (30,) |  | match |  |
| a_preprocessing | example_dtype | float64 (numpy) | torch.float32 |  | info |  |
| a_preprocessing | label_set | [0.0, 1.0] | [0, 1] |  | match |  |
| b_architecture | model_family | xgboost.XGBClassifier (200 trees, depth 4) | BreastCancerMLP (MLP 30-128-64-32-1, BatchNorm+Dropout) |  | surrogate | no parameter correspondence by design |
| b_architecture | param_count | 787 | 14785 |  | surrogate | XGB: 200 trees with 787 leaf values; MLP: trainable tensors |
| b_architecture | leaf_modules | boosted decision trees (no layers) | Linear>BatchNorm1d>ReLU>Dropout>Linear>BatchNorm1d>ReLU>Dropout>Linear>BatchNorm1d>ReLU>Dropout>Linear |  | surrogate |  |
| c_loss_init | loss_at_init |  |  |  | n/a | gradient boosting has no parameter-space gradient step comparable to an nn.Module update |
| d_one_step | grad_cosine |  |  |  | n/a | gradient boosting has no parameter-space gradient step comparable to an nn.Module update |
| d_one_step | param_delta_cosine |  |  |  | n/a | gradient boosting has no parameter-space gradient step comparable to an nn.Module update |
| e_budget | test_acc_source_native | 0.95614 |  |  | info | XGBClassifier as in the script, source test split (114 rows) |
| e_budget | test_auc_source_native | 0.994048 |  |  | info |  |
| e_budget | test_acc_converted_runtime_300steps_source_split |  | 0.964912 | 0.00877193 | match | surrogate MLP trained by FLClient.local_train on the source train split, AdamW(lr=0.001); 1s |
| e_budget | test_auc_converted_runtime_300steps_source_split |  | 0.997685 | 0.00363757 | match |  |
| run | elapsed_sec |  |  | 0.8 | info |  |
