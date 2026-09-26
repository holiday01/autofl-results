# v1 vs v2 benchmark comparison

v1: `results/benchmark.csv`  
v2: `results/benchmark_v2.csv`

## Per-row stage / e2e comparison

| script | method | v1 stage | v2 stage | v1 e2e | v2 e2e | stage changed | e2e changed | v2 error (first line) |
|---|---|---|---|---|---|---|---|---|
| backbone_image_classifier | ast | preflight/data_load | preflight/import_check | False | False | YES |  | ModuleNotFoundError: No module named 'lightning' |
| dcgan_main | ast | preflight/data_load | preflight/data_load | False | False |  |  | NotImplementedError: build_dataloader: no Dataset class detected — edit the generated file |
| image_classification_from_scratch | ast | preflight/data_load | preflight/import_check | False | False | YES |  | ModuleNotFoundError: No module named 'keras' |
| imagenet_main | ast | preflight/data_load | preflight/data_load | False | False |  |  | NotImplementedError: build_dataloader: no Dataset class detected — edit the generated file |
| lstm_seq2seq | ast | preflight/data_load | preflight/import_check | False | False | YES |  | ModuleNotFoundError: No module named 'keras' |
| mednist_tutorial | ast | preflight/data_load | preflight/data_load | False | False |  |  | NotImplementedError: build_dataloader: no Dataset class detected — edit the generated file |
| mnist_convnet | ast | preflight/data_load | preflight/import_check | False | False | YES |  | ModuleNotFoundError: No module named 'keras' |
| mnist_lite | ast | preflight/data_load | preflight/import_check | False | False | YES |  | ModuleNotFoundError: No module named 'lightning' |
| mnist_main | ast | preflight/data_load | preflight/data_load | False | False |  |  | NotImplementedError: build_dataloader: no Dataset class detected — edit the generated file |
| spleen_segmentation_3d | ast | preflight/data_load | preflight/data_load | False | False |  |  | NotImplementedError: build_dataloader: no Dataset class detected — edit the generated file |
| backbone_image_classifier | zero_shot | preflight/forward_pass | preflight/forward_pass | False | False |  |  | AttributeError: 'NoneType' object has no attribute 'zero_grad' |
| dcgan_main | zero_shot | preflight/data_load | preflight/data_load | False | False |  |  | KeyError: 'dataset' |
| image_classification_from_scratch | zero_shot | preflight/data_load | preflight/import_check | False | False | YES |  | ModuleNotFoundError: No module named 'keras' |
| imagenet_main | zero_shot | preflight/data_load | preflight/data_load | False | False |  |  | FileNotFoundError: [Errno 2] No such file or directory: 'imagenet/train' |
| lstm_seq2seq | zero_shot | preflight/data_load | preflight/import_check | False | False | YES |  | ModuleNotFoundError: No module named 'keras' |
| mednist_tutorial | zero_shot | preflight/data_load | preflight/data_load | False | False |  |  | FileNotFoundError: Caught FileNotFoundError in DataLoader worker process 0. |
| mnist_convnet | zero_shot | preflight/data_load | preflight/import_check | False | False | YES |  | ModuleNotFoundError: No module named 'keras' |
| mnist_lite | zero_shot | preflight/backward_pass | preflight/backward_pass | False | False |  |  | RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn |
| mnist_main | zero_shot | preflight/forward_pass | preflight/forward_pass | False | False |  |  | AttributeError: 'NoneType' object has no attribute 'zero_grad' |
| spleen_segmentation_3d | zero_shot | preflight/data_load | preflight/data_load | False | False |  |  | KeyError: 'data_dir' |
| backbone_image_classifier | few_shot | preflight/backward_pass | preflight/backward_pass | False | False |  |  | RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn |
| dcgan_main | few_shot | preflight/backward_pass | PASS | False | True | YES | YES |  |
| image_classification_from_scratch | few_shot | preflight/data_load | preflight/import_check | False | False | YES |  | ModuleNotFoundError: No module named 'keras' |
| imagenet_main | few_shot | preflight/data_load | preflight/data_load | False | False |  |  | FileNotFoundError: [Errno 2] No such file or directory: './train' |
| lstm_seq2seq | few_shot | preflight/data_load | preflight/import_check | False | False | YES |  | ModuleNotFoundError: No module named 'keras' |
| mednist_tutorial | few_shot | preflight/data_load | preflight/data_load | False | False |  |  | ValueError: num_samples should be a positive integer value, but got num_samples=0 |
| mnist_convnet | few_shot | preflight/data_load | preflight/import_check | False | False | YES |  | ModuleNotFoundError: No module named 'keras' |
| mnist_lite | few_shot | PASS | PASS | True | True |  |  |  |
| mnist_main | few_shot | preflight/backward_pass | PASS | False | True | YES | YES |  |
| spleen_segmentation_3d | few_shot | preflight/data_load | preflight/backward_pass | False | False | YES |  | RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn |
| backbone_image_classifier | structured | PASS | PASS | True | True |  |  |  |
| dcgan_main | structured | PASS | PASS | True | True |  |  |  |
| image_classification_from_scratch | structured | PASS | PASS | True | True |  |  |  |
| imagenet_main | structured | PASS | PASS | True | True |  |  |  |
| lstm_seq2seq | structured | PASS | PASS | True | True |  |  |  |
| mednist_tutorial | structured | PASS | PASS | True | True |  |  |  |
| mnist_convnet | structured | PASS | PASS | True | True |  |  |  |
| mnist_lite | structured | PASS | PASS | True | True |  |  |  |
| mnist_main | structured | PASS | PASS | True | True |  |  |  |
| spleen_segmentation_3d | structured | PASS | PASS | True | True |  |  |  |

Rows with changed error_stage: 14 / 40; rows with changed e2e outcome: 2 / 40

### Rows whose e2e outcome changed

- dcgan_main / few_shot: v1 e2e=False (preflight/backward_pass: RuntimeError: element 0 of tensors does not require grad and does not have a gra) -> v2 e2e=True (: )
- mnist_main / few_shot: v1 e2e=False (preflight/backward_pass: RuntimeError: element 0 of tensors does not require grad and does not have a gra) -> v2 e2e=True (: )

### Strategy x stage-of-first-failure (v1, n=40 rows)

| Strategy | Syntax | Interface | S1 Import | S2 Hardware | S3 Data | S4 Forward | S5 Backward | E2E | Pass |
|---|---|---|---|---|---|---|---|---|---|
| ast | 0 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 0 |
| zero_shot | 0 | 0 | 0 | 0 | 7 | 2 | 1 | 0 | 0 |
| few_shot | 0 | 0 | 0 | 0 | 6 | 0 | 3 | 0 | 1 |
| structured | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 |

### Strategy x stage-of-first-failure (v2, n=40 rows)

| Strategy | Syntax | Interface | S1 Import | S2 Hardware | S3 Data | S4 Forward | S5 Backward | E2E | Pass |
|---|---|---|---|---|---|---|---|---|---|
| ast | 0 | 0 | 5 | 0 | 5 | 0 | 0 | 0 | 0 |
| zero_shot | 0 | 0 | 3 | 0 | 4 | 2 | 1 | 0 | 0 |
| few_shot | 0 | 0 | 3 | 0 | 2 | 0 | 2 | 0 | 3 |
| structured | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 |

## Structured passes: data provenance (v2)

| script | framework | e2e | data_path | data_path_exists | dataset_class | dataset_len | elapsed (s) |
|---|---|---|---|---|---|---|---|
| backbone_image_classifier | lightning | True | . | True | MNIST | 60000 | 11.09 |
| mnist_lite | lightning | True | . | True | MNIST | 60000 | 53.71 |
| mednist_tutorial | monai | True | . | True | _SyntheticMRIDataset | 20 | 14.64 |
| spleen_segmentation_3d | monai | True | . | True | _SyntheticSpleenDataset | 20 | 9.3 |
| dcgan_main | pytorch | True | . | True | CIFAR10 | 50000 | 743.36 |
| imagenet_main | pytorch | True | . | True | TensorDataset | 200 | 5.09 |
| mnist_main | pytorch | True | . | True | MNIST | 60000 | 264.51 |
| image_classification_from_scratch | tensorflow | True | . | True | TensorDataset | 200 | 4.85 |
| lstm_seq2seq | tensorflow | True | . | True | _Seq2SeqDataset | 200 | 1.07 |
| mnist_convnet | tensorflow | True | . | True | MNIST | 60000 | 19.5 |
