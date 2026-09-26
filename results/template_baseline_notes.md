# Template (rule-based) converter baseline — results and notes

Generated 2026-09-22 by `eval/run_template_baseline.py` (CPU only, `CUDA_VISIBLE_DEVICES=""`, `data_root="."`, `strict_data=False`, i.e. `allow_synthetic_data=True`, the same harness settings as `results/benchmark.csv`). The converter is `converter/template_converter.py`; it makes **no LLM calls**. Generated modules live next to their sources as `benchmarks/<fw>/<stem>_fl_template.py`; the CSV is `results/template_baseline.csv` (same columns as `results/benchmark.csv` plus the evaluator's provenance columns and `template_model`, `template_train_step`, `n_rules`, `n_fallbacks`, `rules_fired`, `fallbacks_used`, `converter_warnings`, `convert_error`).

## 0. Headline numbers

* e2e_runnable: **13/13** scripts (preflight_pass 13/13); DL scripts: **10/10**.
* Of these, 2 pass only through the *generic-MLP* fallback (`TEMPLATE_FALLBACK="generic"`: image_classification_from_scratch, lstm_seq2seq) and 3 through the documented *surrogate* fallback for non-neural estimators (mlp_digits, svm_iris, xgb_breast_cancer).
* Counting a generic-fallback pass as a failed conversion gives **11/13** overall and **8/10** on the DL scripts.
* For comparison (`results/benchmark.csv`, `results/benchmark_extended.csv`): the naive AST baseline is 0/13 (0/10 DL), the LLM structured strategy is 13/13.

## 1. Rule set (`converter/template_converter.py`)

The converter is a deterministic AST rewriter: it never executes the source
script, never calls a model of any kind, and never special-cases a script by
name. Every rule below is applied to every input; the generated file's header
lists which rules fired and which fallbacks were used, and exposes the same
information as module constants (`TEMPLATE_RULES`, `TEMPLATE_FALLBACK`,
`TEMPLATE_TRAIN_STEP`, `TEMPLATE_FALLBACKS`, `TEMPLATE_WARNINGS`).

### R1 model discovery
| id | rule |
|----|------|
| R1.1 | `nn.Module`/`LightningModule` subclasses are indexed. The *training* model is the callable applied to batch data inside the training loop; its name is resolved through function parameters -> call sites -> assignment sites (e.g. `train(args, model, ...)` in `main()` -> `model = Net().to(device)`). `.to()/.cuda()/DataParallel` wrappers are stripped. |
| R1.2 | Library model factories are kept verbatim: `torchvision.models.*`, `models.__dict__[args.arch]()`, `monai.networks.nets.*`, `timm.create_model`, `nn.Sequential`. |
| R1.3 | If several networks are called in one loop (GANs), they are wrapped in `TemplateModelBundle(nn.Module)` so the FL runtime aggregates one `state_dict`; loop-body references are rewritten to `model.<name>`. |
| R1.4 | Post-construction calls `<model>.apply(<init_fn>)` are kept and `<init_fn>` is retained. |
| R1 opt | The first optimizer constructor (`torch.optim.*`, `keras.compile(optimizer=...)`) is recorded (class + resolvable kwargs) and emitted as an informational `build_optimizer()`; the runtime keeps its own AdamW. |

### R2 data discovery
| id | rule |
|----|------|
| R2.1 | Dataset constructors are collected (torchvision classes, `monai.data.*`, `TensorDataset`, local `Dataset` subclasses, keras loaders, `sklearn.datasets.load_*`). The loop's iterable is followed through `DataLoader(...)`/`random_split`/`Subset` and parameters -> call sites to find the constructor(s) that *feed the training loader*; only those are emitted if the link succeeds, otherwise every constructor. Unresolvable `if` conditions (e.g. `opt.dataset`) keep all branches as candidates, ordered: downloadable built-ins > local/MONAI/torch > folder datasets > `FakeData`. |
| R2.2 | Data-root substitution: a string-literal root, an argparse attribute named like a root (`args.data`, `opt.dataroot`) or a variable named like a root (`data_path`, `DATASETS_PATH`, `root`) is bound to `config["data_path"]`; unresolvable names used as arguments of path functions (`os.path.join`, `glob`, ...) are likewise bound (e.g. a `tempdir` parameter). |
| R2.3 | Every candidate is built **and probed** (`len(ds)` and `ds[0]`) inside `try/except`, so lazily failing datasets (missing NIfTI files, missing `nibabel`) are caught. When no candidate works, synthetic data is used **only if** `config["allow_synthetic_data"]`; otherwise `FileNotFoundError` is raised (strict mode). A 90/10 `random_split` gives the `train`/`val` splits. |
| R2.4 | Framework aliases: `keras.datasets.<x>.load_data()` -> `torchvision.datasets.<X>`; `image_dataset_from_directory(dir, image_size)` -> `ImageFolder(join(data_path, dir), Compose([Resize, ToTensor]))`; `lightning.pytorch.demos` `MNIST`/`MNISTDataModule` -> torchvision `MNIST`; `sklearn.datasets.load_*` -> `TensorDataset` with `StandardScaler` mirrored as per-feature standardisation. |

### R3 train-step extraction
| id | rule |
|----|------|
| R3.1 | The innermost `for`/`while` whose own body contains `.backward()` / `manual_backward()` / `scaler.scale(loss).backward()` is the training loop; for Lightning the `training_step` body is used (loss = the back-propagated variable or the returned expression). The loop target (`(data, target)`, `batch_data`, `i, data` from `enumerate`) becomes the batch binding. |
| R3.2 | `with torch.autocast(...)`/`amp.autocast` blocks are inlined (the runtime owns AMP). |
| R3.3 | Call-site scrubbing: statements calling `zero_grad`, `backward`, `manual_backward`, `step`, `toggle_optimizer`, `untoggle_optimizer`, `log`, `log_dict`, `unscale_`, `scaler.update()`, `clip_grad_norm_/value_`, TensorBoard `add_*`, progress-bar/`print` calls, and assignments from `self.optimizers()` are deleted. |
| R3.4 | The body is cut after the assignment of the last back-propagated loss; a backward slice from the loss statement(s) keeps only statements the loss depends on (assignments and in-place method calls on needed names). Everything else (metrics, logging, checkpointing) disappears. |
| R3.5 | Free names of the slice are hoisted from enclosing scopes: the loop's function (before the loop), its callers via parameter->argument mapping, and module level. Argparse defaults are folded (`args.lr` -> `1.0`; `store_true` -> `False`), `if` conditions on folded constants select branches, unresolvable conditions fall back to majority-vote constants (e.g. `nc = 3`), and self-referential assignment chains (`images = [...]; images = [join(root, f) for f in images]`) are emitted in order. Names that are mutated in place (`set.add`, `+=`, item assignment) are never constant-folded. |
| R3.6 | `.cuda(...)`/`.to(dev, non_blocking=True)` -> `.to(device)` with `device = next(model.parameters()).device`; `self` -> `model`; in-place `x.fill_(v)`/`x.zero_()` -> `x = torch.full_like(x, v)` so a single combined backward is autograd-safe; `.item()/.detach()` are stripped from the returned loss. |
| R3.7 | Several back-propagated losses in one loop are returned as their sum (`errD_real + errD_fake + errG`, `g_loss + d_loss`). |
| R3.8 | If the loss cannot be isolated (no loop, or its statement depends on unresolvable names) a generic `logits = model(x); loss = <detected loss>(logits, y)` is emitted, with the loss detected from the loop, a `criterion`/`loss_function` variable, a `keras.compile(loss=...)` argument, or any known loss constructor in the module (default `cross_entropy`). |

### R4 synthetic-shape inference (evidence in priority order)
1. explicit shapes: `keras.Input(shape=...)` (NHWC -> NCHW), `FakeData(image_size=..., num_classes=...)`, MONAI `in_channels`/`spatial_dims` + `Resize`/`spatial_size`/`roi_size`, model `__init__` defaults such as `img_shape=(1, 28, 28)`;
2. torchvision transforms: channel count from `Normalize(mean)`/`Grayscale`, spatial size from the last `Resize`/`CenterCrop`/`RandomResizedCrop`/`RandomCrop` of each `Compose` (majority vote);
3. dataset catalogue (MNIST family 1x28x28/10, CIFAR 3x32x32, STL10, ImageNet, sklearn toy sets with feature and class counts, ...);
4. model introspection: first `Conv{1,2,3}d` `in_channels`, first `Linear` `in_features` for flat inputs, last `Linear` `out_features` / `out_channels` / `num_classes` kwargs for the label space;
5. defaults (32x32 spatial, 64 flat features, 10 classes) — flagged as a fallback.
The target kind follows the detected loss: class labels (`long`) for cross-entropy/NLL, `{0,1}` floats for BCE, floats for MSE/L1, binary masks with `out_channels` channels for Dice-type losses. If the loop indexes the batch with string keys (`batch["img"]`, `batch["seg"]`) the synthetic dataset yields dicts with the same keys in source order. The sample count is bounded by an element budget (16M floats), so 3-D volumes get ~18 samples and MNIST-sized inputs 256.

### R5 framework translation
| id | rule |
|----|------|
| R5.1 | Keras `Sequential([...])`, `model.add(...)` chains and *linear* functional chains (`x = L(...)(x)`) are translated layer by layer into `torch.nn.Sequential` with static shape tracking (Conv2D/SeparableConv2D/DepthwiseConv2D, Max/AveragePooling2D, Global pooling, BatchNormalization, Activation/ReLU/LeakyReLU/Softmax, Dropout, Flatten, Dense, Rescaling, ZeroPadding2D, Embedding, LSTM/GRU/Bidirectional, Conv1D/pooling1D, augmentation layers as identity). A final softmax/sigmoid is folded into the loss. Anything else (loops, residual `add`, multiple `keras.Input`, `return_state=True`) yields a generic MLP on the inferred input shape marked `TEMPLATE_FALLBACK = "generic"`. |
| R5.2 | Lightning: `LightningModule` base -> `nn.Module`; `save_hyperparameters()` -> `self.hparams = SimpleNamespace(...)` from the `__init__` signature; hooks (`*_step`, `configure_*`, `on_*`, `*_dataloader`, ...) removed; `self.log*/manual_backward/toggle_optimizer` stripped; `self.device` -> `next(self.parameters()).device`; the model is instantiated with its call-site kwargs (argparse-folded) or class defaults. |
| R5.3 | MONAI: the network constructor is kept verbatim; synthetic N-D volumes are shaped from `spatial_dims`/`in_channels` and the transform crop/resize size; dict batches are preserved. |
| R5.4 | sklearn/XGBoost estimators (`*Classifier`, `*Regressor`, `SVC`, ...) become `TemplateTabularSurrogate`, an MLP whose hidden sizes mirror `hidden_layer_sizes` when present (default (64, 64)), on the catalogue feature count with cross-entropy/MSE; marked `TEMPLATE_FALLBACK = "surrogate"`. |

### R6 module hygiene
Imports are tree-shaken to the names the generated module references; Lightning/Keras/TensorFlow imports are dropped (Lightning's demo `MNIST` is aliased to torchvision); imports that the source guarded with `if`/`try` or placed inside functions are re-emitted inside `try/except ImportError`; `from __future__` lines are hoisted; module-level constants that kept classes need (`nz`, `ngf`, `nc`, ...) are folded and emitted before the classes; kept classes/functions are the transitive closure of what the generated interface references, emitted verbatim (Lightning classes are re-serialised from the cleaned AST).


## 2. Per-script outcome

| script | fw | stage reached | preflight | e2e | dataset used (class/len) | model | train_step | elapsed s | rules fired (ids) | fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|
| dcgan_main | pytorch | all 5 stages + e2e round | True | True | CIFAR10/50000 | none | extracted | 481 | R1.4, R1.1, R1.3, R3.5, R2.2, R2.1, R2.3, R4, R3.1, R3.7, R3.3, R3.6, R3.4, R6, R1 | none |
| imagenet_main | pytorch | all 5 stages + e2e round | True | True | TensorDataset/111 | none | extracted | 11 | R1.2, R2.2, R3.5, R2.1, R2.3, R4, R3.1, R3.4, R1, R6 | none |
| mnist_main | pytorch | all 5 stages + e2e round | True | True | MNIST/60000 | none | extracted | 198 | R1.1, R2.2, R3.5, R2.1, R2.3, R4, R3.1, R3.3, R1, R6 | none |
| image_classification_from_scratch | tensorflow | all 5 stages + e2e round | True | True | TensorDataset/172 | generic | generic | 9 | R2.4, R2.1, R2.3, R4, R3.8, R1, R6 | generic model: non-linear functional graph (layers applied inside a loop); generic train_step (no training loop) |
| lstm_seq2seq | tensorflow | all 5 stages + e2e round | True | True | TensorDataset/256 | generic | generic | 3 | R2.3, R4, R3.8, R1, R6 | generic model: multi-input functional graph (4 keras.Input calls); no usable dataset construction found; synthetic data only; synthetic input shape could not be fully resolved from the source; defaults used; generic train_step (no training loop) |
| mnist_convnet | tensorflow | all 5 stages + e2e round | True | True | MNIST/60000 | none | generic | 38 | R5.1, R2.4, R2.1, R2.3, R4, R3.8, R1, R6 | generic train_step (no training loop) |
| mednist_tutorial | monai | all 5 stages + e2e round | True | True | TensorDataset/18 | none | extracted | 37 | R1.2, R3.5, R2.1, R2.3, R4, R3.1, R3.3, R3.4, R1, R6 | none |
| spleen_segmentation_3d | monai | all 5 stages + e2e round | True | True | Dataset/20 | none | extracted | 67 | R1.2, R2.2, R3.5, R2.5, R2.1, R2.3, R4, R5.3, R3.1, R3.3, R3.4, R1, R6 | none |
| backbone_image_classifier | lightning | all 5 stages + e2e round | True | True | MNIST/60000 | none | extracted | 12 | R5.2, R2.2, R2.1, R2.3, R4, R3.1, R1, R2.4, R6 | none |
| mnist_lite | lightning | all 5 stages + e2e round | True | True | MNIST/60000 | none | extracted | 198 | R5.2, R2.4, R2.1, R2.3, R4, R3.1, R3.7, R3.3, R1, R6 | none |
| mlp_digits | sklearn | all 5 stages + e2e round | True | True | TensorDataset/1797 | surrogate | generic | 8 | R5.4, R2.4, R2.1, R2.3, R4, R3.8, R1, R6 | surrogate model for a non-neural estimator; generic train_step (no training loop) |
| svm_iris | sklearn | all 5 stages + e2e round | True | True | TensorDataset/150 | surrogate | generic | 3 | R5.4, R2.4, R2.1, R2.3, R4, R3.8, R1, R6 | surrogate model for a non-neural estimator; generic train_step (no training loop) |
| xgb_breast_cancer | xgboost | all 5 stages + e2e round | True | True | TensorDataset/569 | surrogate | generic | 4 | R5.4, R2.4, R2.1, R2.3, R4, R3.8, R1, R6 | surrogate model for a non-neural estimator; generic train_step (no training loop) |

Elapsed times were measured on a machine shared with other jobs (load average 25-100 on 24 cores), so they are not comparable with the timings in `results/benchmark.csv`.

### 2.1 Rules and fallbacks that fired, per script (from the generated headers)

**dcgan_main** (pytorch) — model: `none`, train_step: `extracted`
* R1.4 kept post-construction call netD.apply(weights_init)
* R1.1 model class Discriminator (instantiated at module level as netD)
* R1.4 kept post-construction call netG.apply(weights_init)
* R1.1 model class Generator (instantiated at module level as netG)
* R1.3 several networks trained in one loop -> TemplateModelBundle
* R3.5 constructor arguments resolved from enclosing scopes / argparse defaults: ['ngpu']
* R2.2 data root bound to config['data_path'] (opt.dataroot)
* R3.5 argparse defaults folded in the data pipeline: ['imageSize']
* R3.5 argparse defaults folded in the data pipeline: ['classes', 'imageSize']
* R3.5 data pipeline definitions hoisted: ['classes']
* R2.1 5 dataset candidate(s) emitted (data-flow linked to the training loader)
* R2.3 candidates probed in try/except; synthetic fallback gated on allow_synthetic_data
* R4 synthetic shape (3, 64, 64) / target=binary from: explicit:FakeData=(3, 64, 64); catalogue:CIFAR10=(3, 32, 32),10; model:first-Conv2d.in_channels=3
* R3.1 train_step extracted from the training loop at line 223 (module level)
* R3.7 multi-loss loop: returning errD_real + errD_fake + errG
* R3.3 optimizer/backward/logging call sites scrubbed
* R3.6 in-place label.fill_() rewritten out-of-place
* R3.4 backward slice removed statements not feeding the loss
* R3.5 hoisted from enclosing scopes: ['criterion', 'fake_label', 'nz', 'real_label']
* R6 module-level constant ngf folded for Generator
* R6 module-level constant nz folded for Generator
* R6 module-level constant nc folded for Generator
* R6 module-level constant ndf folded for Discriminator
* R6 module-level constant nc folded for Discriminator
* R1 optimizer constructor detected: Adam(lr=0.0002, betas=(0.5, 0.999)) [torch]
* R6 imports tree-shaken to the names referenced by the generated module
* converter warnings: 5 (scrubbed statements are listed in the file header)

**imagenet_main** (pytorch) — model: `none`, train_step: `extracted`
* R1.2 library model constructor kept verbatim: models.__dict__ (main_worker())
* R2.2 data root bound to config['data_path'] (args.data)
* R3.5 data pipeline definitions hoisted: ['normalize', 'traindir']
* R2.1 1 dataset candidate(s) emitted (data-flow linked to the training loader)
* R2.3 candidates probed in try/except; synthetic fallback gated on allow_synthetic_data
* R4 synthetic shape (3, 224, 224) / target=class[1000] from: explicit:FakeData=(3, 224, 224); classes:FakeData=1000
* R3.1 train_step extracted from the training loop at line 327 in train()
* R3.4 backward slice removed statements not feeding the loss
* R3.5 hoisted from enclosing scopes: ['criterion']
* R1 optimizer constructor detected: SGD(lr=0.1, momentum=0.9, weight_decay=0.0001) [torch]
* R6 imports tree-shaken to the names referenced by the generated module

**mnist_main** (pytorch) — model: `none`, train_step: `extracted`
* R1.1 model class Net (instantiated at main() as model)
* R2.2 data root bound to config['data_path'] (literal root)
* R3.5 data pipeline definitions hoisted: ['transform']
* R2.1 1 dataset candidate(s) emitted (data-flow linked to the training loader)
* R2.3 candidates probed in try/except; synthetic fallback gated on allow_synthetic_data
* R4 synthetic shape (1, 28, 28) / target=class[10] from: catalogue:MNIST=(1, 28, 28),10; model:first-Conv2d.in_channels=1; model:last-Linear.out_features=10; transform:channels=1
* R3.1 train_step extracted from the training loop at line 38 in train()
* R3.3 optimizer/backward/logging call sites scrubbed
* R1 optimizer constructor detected: Adadelta(lr=1.0) [torch]
* R6 imports tree-shaken to the names referenced by the generated module
* converter warnings: 1 (scrubbed statements are listed in the file header)

**image_classification_from_scratch** (tensorflow) — model: `generic`, train_step: `generic`
* R2.4 keras image_dataset_from_directory -> torchvision ImageFolder(+Resize, ToTensor)
* R2.1 1 dataset candidate(s) emitted
* R2.3 candidates probed in try/except; synthetic fallback gated on allow_synthetic_data
* R4 synthetic shape (3, 180, 180) / target=binary from: explicit:keras.Input=(3, 180, 180); classes:keras.Input=1
* R3.8 generic train_step with detected loss BinaryCrossentropy (keras.compile)
* R1 optimizer constructor detected: Adam(lr=0.0003) [keras.optimizers]
* R6 unavailable/irrelevant framework imports dropped: keras, tensorflow
* R6 imports tree-shaken to the names referenced by the generated module
* FALLBACK: generic model: non-linear functional graph (layers applied inside a loop)
* FALLBACK: generic train_step (no training loop)

**lstm_seq2seq** (tensorflow) — model: `generic`, train_step: `generic`
* R2.3 candidates probed in try/except; synthetic fallback gated on allow_synthetic_data
* R4 synthetic shape (16, 32) / target=class[10] from: explicit:keras.Input=(16, 32); default:classes=10
* R3.8 generic train_step with detected loss categorical_crossentropy (keras.compile)
* R1 optimizer constructor detected: RMSprop() [keras.compile]
* R6 unavailable/irrelevant framework imports dropped: keras
* R6 imports tree-shaken to the names referenced by the generated module
* FALLBACK: generic model: multi-input functional graph (4 keras.Input calls)
* FALLBACK: no usable dataset construction found; synthetic data only
* FALLBACK: synthetic input shape could not be fully resolved from the source; defaults used
* FALLBACK: generic train_step (no training loop)
* converter warnings: 1 (scrubbed statements are listed in the file header)

**mnist_convnet** (tensorflow) — model: `none`, train_step: `generic`
* R5.1 keras layer stack -> torch.nn.Sequential (static NHWC->NCHW shape tracking)
* R5.1 keras Input shape (28, 28, 1) -> channels-first (1, 28, 28)
* R5.1 final softmax folded into the loss (model returns logits)
* R2.4 keras.datasets.mnist.load_data() -> torchvision.datasets.MNIST
* R2.1 1 dataset candidate(s) emitted
* R2.3 candidates probed in try/except; synthetic fallback gated on allow_synthetic_data
* R4 synthetic shape (1, 28, 28) / target=class[10] from: explicit:keras.Input=(1, 28, 28); classes:keras.Input=10; catalogue:mnist=(1, 28, 28),10
* R3.8 generic train_step with detected loss categorical_crossentropy (keras.compile)
* R1 optimizer constructor detected: Adam() [keras.compile]
* R6 unavailable/irrelevant framework imports dropped: keras
* R6 imports tree-shaken to the names referenced by the generated module
* FALLBACK: generic train_step (no training loop)

**mednist_tutorial** (monai) — model: `none`, train_step: `extracted`
* R1.2 library model constructor kept verbatim: monai.networks.nets.DenseNet121 (main())
* R3.5 data pipeline definitions hoisted: ['images', 'labels', 'train_transforms']
* R2.1 1 dataset candidate(s) emitted (data-flow linked to the training loader)
* R2.3 candidates probed in try/except; synthetic fallback gated on allow_synthetic_data
* R4 synthetic shape (1, 96, 96, 96) / target=class[2] from: explicit:monai(in_channels,spatial)=(1, 96, 96, 96); classes:monai.out_channels=2
* R3.1 train_step extracted from the training loop at line 95 in main()
* R3.3 optimizer/backward/logging call sites scrubbed
* R3.4 backward slice removed statements not feeding the loss
* R3.5 hoisted from enclosing scopes: ['loss_function']
* R1 optimizer constructor detected: Adam(lr=1e-05) [torch]
* R6 imports tree-shaken to the names referenced by the generated module
* converter warnings: 1 (scrubbed statements are listed in the file header)

**spleen_segmentation_3d** (monai) — model: `none`, train_step: `extracted`
* R1.2 library model constructor kept verbatim: monai.networks.nets.UNet (main())
* R2.2 data root bound to config['data_path'] (tempdir)
* R3.5 data pipeline definitions hoisted: ['images', 'segs', 'train_files', 'train_transforms']
* R2.5 DataLoader collate_fn=list_data_collate preserved (batch structure)
* R2.1 1 dataset candidate(s) emitted (data-flow linked to the training loader)
* R2.3 candidates probed in try/except; synthetic fallback gated on allow_synthetic_data
* R4 synthetic shape (1, 96, 96, 96) / target=segmentation from: explicit:monai(in_channels,spatial)=(1, 96, 96, 96); classes:monai.out_channels=1; batch:dict['img', 'seg']
* R5.3 dict batches preserved with keys ['img', 'seg']
* R3.1 train_step extracted from the training loop at line 130 in main()
* R3.3 optimizer/backward/logging call sites scrubbed
* R3.4 backward slice removed statements not feeding the loss
* R3.5 hoisted from enclosing scopes: ['loss_function']
* R1 optimizer constructor detected: Adam(lr=0.001) [torch]
* R6 imports tree-shaken to the names referenced by the generated module
* converter warnings: 1 (scrubbed statements are listed in the file header)

**backbone_image_classifier** (lightning) — model: `none`, train_step: `extracted`
* R5.2 LightningModule LitClassifier used as the model (hooks stripped)
* R2.2 data root bound to config['data_path'] (DATASETS_PATH)
* R2.1 2 dataset candidate(s) emitted
* R2.3 candidates probed in try/except; synthetic fallback gated on allow_synthetic_data
* R4 synthetic shape (1, 28, 28) / target=class[10] from: catalogue:MNIST=(1, 28, 28),10
* R3.1 train_step extracted from the LightningModule.training_step
* R5.2 base LightningModule -> nn.Module
* R5.2 save_hyperparameters() -> SimpleNamespace hparams
* R5.2 hook training_step() removed
* R5.2 hook validation_step() removed
* R5.2 hook test_step() removed
* R5.2 hook predict_step() removed
* R5.2 hook configure_optimizers() removed
* R1 optimizer constructor detected: Adam() [torch]
* R2.4 lightning.pytorch.demos MNIST -> torchvision.datasets.MNIST
* R6 unavailable/irrelevant framework imports dropped: lightning.pytorch, lightning.pytorch.cli, lightning.pytorch.demos.mnist_datamodule, lightning.pytorch.utilities.imports
* R6 imports tree-shaken to the names referenced by the generated module

**mnist_lite** (lightning) — model: `none`, train_step: `extracted`
* R5.2 LightningModule GAN used as the model (hooks stripped)
* R2.4 lightning demo MNISTDataModule -> torchvision.datasets.MNIST
* R2.1 1 dataset candidate(s) emitted
* R2.3 candidates probed in try/except; synthetic fallback gated on allow_synthetic_data
* R4 synthetic shape (1, 28, 28) / target=binary from: explicit:__init__.img_shape=(1, 28, 28); catalogue:MNISTDataModule=(1, 28, 28),10
* R3.1 train_step extracted from the LightningModule.training_step
* R3.7 multi-loss loop: returning g_loss + d_loss
* R3.3 optimizer/backward/logging call sites scrubbed
* R5.2 base LightningModule -> nn.Module
* R5.2 save_hyperparameters() -> SimpleNamespace hparams
* R5.2 hook training_step() removed
* R5.2 hook configure_optimizers() removed
* R5.2 hook on_train_epoch_end() removed
* R1 optimizer constructor detected: Adam() [torch]
* R6 unavailable/irrelevant framework imports dropped: lightning.pytorch, lightning.pytorch.core, lightning.pytorch.demos.mnist_datamodule, lightning.pytorch.trainer, lightning.pytorch.utilities.imports
* R6 imports tree-shaken to the names referenced by the generated module
* converter warnings: 7 (scrubbed statements are listed in the file header)

**mlp_digits** (sklearn) — model: `surrogate`, train_step: `generic`
* R5.4 sklearn/XGBoost estimator MLPClassifier -> MLP surrogate on (64,)
* R5.4 hidden_layer_sizes=(128, 64) mirrored from MLPClassifier
* R2.4 StandardScaler mirrored as per-feature standardisation
* R2.4 sklearn.datasets.load_digits() -> TensorDataset
* R2.1 1 dataset candidate(s) emitted
* R2.3 candidates probed in try/except; synthetic fallback gated on allow_synthetic_data
* R4 synthetic shape (64,) / target=class[10] from: explicit:sklearn.load_digits=(64,); classes:sklearn.load_digits=10; catalogue:load_digits=(64,),10
* R3.8 generic train_step with detected loss cross_entropy (default)
* R1 optimizer constructor detected: AdamW() [default]
* R6 imports tree-shaken to the names referenced by the generated module
* FALLBACK: surrogate model for a non-neural estimator
* FALLBACK: generic train_step (no training loop)

**svm_iris** (sklearn) — model: `surrogate`, train_step: `generic`
* R5.4 sklearn/XGBoost estimator SVC -> MLP surrogate on (4,)
* R2.4 StandardScaler mirrored as per-feature standardisation
* R2.4 sklearn.datasets.load_iris() -> TensorDataset
* R2.1 1 dataset candidate(s) emitted
* R2.3 candidates probed in try/except; synthetic fallback gated on allow_synthetic_data
* R4 synthetic shape (4,) / target=class[3] from: explicit:sklearn.load_iris=(4,); classes:sklearn.load_iris=3; catalogue:load_iris=(4,),3
* R3.8 generic train_step with detected loss cross_entropy (default)
* R1 optimizer constructor detected: AdamW() [default]
* R6 imports tree-shaken to the names referenced by the generated module
* FALLBACK: surrogate model for a non-neural estimator
* FALLBACK: generic train_step (no training loop)

**xgb_breast_cancer** (xgboost) — model: `surrogate`, train_step: `generic`
* R5.4 sklearn/XGBoost estimator XGBClassifier -> MLP surrogate on (30,)
* R2.4 StandardScaler mirrored as per-feature standardisation
* R2.4 sklearn.datasets.load_breast_cancer() -> TensorDataset
* R2.1 1 dataset candidate(s) emitted
* R2.3 candidates probed in try/except; synthetic fallback gated on allow_synthetic_data
* R4 synthetic shape (30,) / target=class[2] from: explicit:sklearn.load_breast_cancer=(30,); classes:sklearn.load_breast_cancer=2; catalogue:load_breast_cancer=(30,),2
* R3.8 generic train_step with detected loss cross_entropy (default)
* R1 optimizer constructor detected: AdamW() [default]
* R6 imports tree-shaken to the names referenced by the generated module
* FALLBACK: surrogate model for a non-neural estimator
* FALLBACK: generic train_step (no training loop)

## 3. Iteration log (what failed during development and which *general* rule fixed it)

The converter was run, the failure stage/message of every script inspected, and only general rules were changed (no script is referenced by name in the converter). Runs are numbered in the order they happened.

| run | failing script(s) | stage / message | root cause | general rule added or fixed |
|---|---|---|---|---|
| dry conversion | imagenet_main | train_step fell back to generic: `loss statement depends on unresolvable names ['criterion']` | the name-cycle guard of the hoisting resolver keyed on the bare name, so `criterion` (a parameter of `train()`) could not be re-resolved in the caller's scope | cycle guard keyed on (name, scope) — R3.5 parameter -> call-site resolution now works across scopes |
| dry conversion | dcgan_main | synthetic shape (3, 32, 32) although the script resizes to 64 | argparse attributes (`opt.imageSize`) were folded by the rewriter but not by the constant evaluator used for shape inference | attribute-aware constant lookup (`_Lookup.attr_lookup`) shared by every rule that evaluates constants (R3.5/R4) |
| dry conversion | imagenet_main | `traindir = 'imagenet/train'` — `args.data` folded to its literal default instead of `data_path` | constant folding ran before data-root substitution in the data context | in the data context the rewrite path (R2.2) runs before constant folding |
| dry conversion | lstm_seq2seq | `num_encoder_tokens` folded to 0 | `len(input_characters)` evaluated on the initial `set()` while the set is filled later with `.add()` | names mutated in place (`.add/.append/+=/item assignment`) are never constant-folded (R3.5) |
| dry conversion | spleen_segmentation_3d | dict keys `['seg', 'img']` (reversed) | AST walk order is LIFO | subscripts sorted by source position (R4 batch-format inference) |
| run 1 | dcgan_main | preflight/forward_pass: `NameError: name 'weights_init' is not defined` | the closure computation concatenated the emitted sections without separators, so the joined code failed to parse and the closure was empty | sections joined with blank lines; a parse failure now falls back to a token scan (R6 closure) |
| run 2 (fast group) | spleen_segmentation_3d | preflight/forward_pass: `TypeError: list indices must be integers or slices, not str` | the real NIfTI files *were* found under `data_path`; `RandCropByPosNegLabeld(num_samples=4)` returns a list per item and the source used `collate_fn=list_data_collate`, which the generated DataLoader dropped | **R2.5**: the `collate_fn` of the DataLoader that feeds the training loop is followed through the data-flow link and re-emitted |
| run 2 (TF) | image_classification_from_scratch | passes, but `build_optimizer` reported `Adam()` without the learning rate | `keras.optimizers.Adam(3e-4)` was matched by the torch optimizer branch (positional arg dropped) | Keras optimizer constructors are routed to the Keras branch (`lr=` from the first positional / `learning_rate=`) |
| run 2 | all remaining scripts | — | — | no further rule changes; final files regenerated and re-evaluated with the final converter |

Things that were deliberately **not** done because they would be script-specific: hard-coding IXI/PetImages/fra.txt paths, seeding vocabulary sizes for the seq2seq script, or writing a hand-made residual block translator for the Cats-vs-Dogs mini-Xception.

## 4. What the rules could and could not do (honest analysis)

### 4.1 Passes that preserve the source semantics (rules only, no fallback)
* **mnist_main, imagenet_main** — the classic single-model/single-loss PyTorch
  loop. The whole extraction chain worked: model located through
  `train(args, model, ...)` -> `main()`; the loss statement kept verbatim;
  `criterion`/`transform` hoisted from the enclosing function; argparse defaults
  folded (`args.arch -> 'resnet18'`, `args.data -> data_path`); the ImageNet
  folder is absent so the module used the synthetic 3x224x224 stand-in whose
  shape came from the script's own `FakeData(..., (3, 224, 224), 1000, ...)`
  dummy branch. Nothing is approximated.
* **mednist_tutorial, spleen_segmentation_3d** — MONAI network constructors kept
  verbatim; the MONAI transform pipelines were hoisted intact; the segmentation
  loop's dict batches (`batch["img"]`, `batch["seg"]`) and the source's
  `collate_fn=list_data_collate` were preserved (rule R2.5 was added after the
  first run exposed this). Spleen actually trained on the NIfTI volumes found
  under `data_path` (MONAI read them without nibabel); MedNIST used synthetic
  1x96x96x96 volumes because the IXI files are not present.
* **backbone_image_classifier** — Lightning `training_step` body used as-is;
  `LightningModule` -> `nn.Module`; the demo `MNIST` alias resolved to
  torchvision and trained on the real MNIST files under `data_path`.
* **dcgan_main, mnist_lite** — the two GANs. The rule set bundles the two
  networks (R1.3), scrubs the two optimisers (R3.3), rewrites the in-place
  `label.fill_()` that would otherwise break a single combined backward (R3.6),
  and returns the *sum* of the back-propagated losses (R3.7). This is the same
  approximation the LLM-generated structured clients make (one AdamW over both
  networks, D and G losses summed), so it is a faithful *baseline* for the
  harness, but it is not adversarial training in the original sense: with a
  single optimizer the generator's loss also updates the discriminator. Rules
  can detect this situation (two optimisers / two `backward()` calls) but cannot
  express alternating optimisation in a single-loss `train_step` contract.

### 4.2 Passes that rely on a marked fallback
* **mnist_convnet (Keras Sequential)** — the *model* is a genuine translation
  (R5.1, shape-tracked `nn.Sequential`, softmax folded into cross-entropy) and
  the data comes from real MNIST, but the `train_step` is the generic
  `logits = model(x); loss = cross_entropy(...)` because `model.fit()` has no
  loop body to extract. The generic step is exactly what `fit()` would do for a
  compiled classifier, so this pass is semantically honest.
* **mlp_digits, svm_iris, xgb_breast_cancer (sklearn / XGBoost)** — a neural
  surrogate is the only way to satisfy an `nn.Module` contract; the surrogate
  mirrors `hidden_layer_sizes` and the `StandardScaler`, and the harness trains
  on the real toy datasets. `TEMPLATE_FALLBACK = "surrogate"` marks that the
  estimator family (kernel SVM, gradient-boosted trees) was replaced, exactly
  as the LLM structured clients did.
* **image_classification_from_scratch, lstm_seq2seq (Keras functional)** —
  both run through the harness only because of the *generic* fallback
  (`TEMPLATE_FALLBACK = "generic"`): a two-layer MLP on the inferred input
  shape. For the Cats-vs-Dogs script the input shape (3x180x180), the binary
  logit head and the BCE-with-logits loss are all correctly inferred from
  `keras.Input`, `Dense(units=1)` and `BinaryCrossentropy(from_logits=True)`,
  but the mini-Xception body (loop over block sizes, residual `layers.add`) is
  not translated. For the seq2seq script even the input shape is unresolvable
  (vocabulary sizes are computed from a downloaded text file) and the
  two-input encoder/decoder graph is out of scope, so the generated module is a
  placeholder that exercises the pipeline and nothing more. These two passes
  should be reported as "harness-runnable via generic fallback", not as
  successful conversions.

### 4.3 What more rules could fix
* **Functional Keras graphs with static structure** (residual blocks, `for size
  in [...]` loops) could be handled by a symbolic tracer that executes the
  model-building function with shape-only placeholder tensors and records a
  DAG; that is a small compiler rather than a template rule, and it still needs
  a catalogue of layer semantics. It is feasible for the Cats-vs-Dogs script.
* **Data-derived hyper-parameters** (vocabulary sizes, `max_seq_len`) can only
  be resolved by running the data pipeline, which a static converter cannot do
  without the data; a rule could at best expose them as `config` keys with
  documented defaults (the generic fallback already does this).
* **Optimizer-specific semantics** (Adadelta lr=1.0 in `mnist_main`, SGD with
  momentum in `imagenet_main`, two Adams in the GANs) are detected and emitted
  as `build_optimizer()` but ignored by the FL runtime, which owns a single
  AdamW; that is a property of the harness contract, not of the converter.

### 4.4 Structurally beyond rule-based rewriting under this contract
* **Alternating / multi-optimizer training** (DCGAN, Lightning GAN with manual
  optimisation): the contract admits one loss and one optimizer step per
  batch. Summing the losses makes the module runnable but changes the game.
* **Custom `train_step` overrides and callbacks in Keras** (`class MyModel(keras.Model): def train_step`), 
  `tf.function` graphs and `tf.data` pipelines with Python-side preprocessing:
  there is no PyTorch equivalent to rewrite to without executing TensorFlow,
  which is exactly what the environment lacks.
* **Multi-input / multi-output models** (encoder-decoder seq2seq) and
  **teacher forcing**: the batch contract is `(x, y)` or a dict; a rule can
  build the dict, but the model-side graph translation is the blocker.
* **Anything that needs the data to decide the architecture** (vocabulary,
  number of classes read from a folder listing): resolvable only at runtime.
