## run_simulation: runtime actually used (v2)

- torch_device: cuda
- cuda_available: True
- torch_cuda_device_name: NVIDIA Graphics Device
- nvidia_smi_gpu_name: NVIDIA Graphics Device
- amp_requested: True
- amp_active: True
- torch_version: 2.10.0+cu128
- python_version: 3.11.9
- batch_size: 16
- gradient_accumulation: 1
- cpu_count: 24
- ram_gb: 125.3
- seed: 42; aggregation: fedavg_sample_weighted

### PyTorch MNIST (3 clients, 5 rounds)

v2 partition: iid_disjoint_equal_shards, train split 54000, shards {'site-1': 18000, 'site-2': 18000, 'site-3': 18000}, seed 42
v1 partition: none (each client trained on the full 54000-example split); v1 aggregation: uniform mean

| Round | v1 agg. loss | v1 elapsed (s) | v2 agg. loss (weighted) | v2 site-1 | v2 site-2 | v2 site-3 | v2 elapsed (s) |
|---|---|---|---|---|---|---|---|
| 1 | 0.1993 | 20.0 | 0.3218 | 0.3140 | 0.3389 | 0.3127 | 10.23 |
| 2 | 0.1022 | 19.9 | 0.1494 | 0.1477 | 0.1480 | 0.1524 | 10.64 |
| 3 | 0.0735 | 19.88 | 0.1092 | 0.1116 | 0.1064 | 0.1097 | 8.2 |
| 4 | 0.0574 | 19.89 | 0.0919 | 0.0945 | 0.0910 | 0.0901 | 10.02 |
| 5 | 0.0474 | 19.99 | 0.0772 | 0.0783 | 0.0764 | 0.0770 | 11.6 |

v2 monotonic decrease: True; v2 reduction: 76.0%; v2 total round time: 50.7s
v1 monotonic decrease: True; v1 reduction: 76.2%; v1 total round time: 99.7s

### Lightning GAN (2 clients, 3 rounds)

v2 partition: iid_disjoint_equal_shards, train split 54000, shards {'lgn-1': 27000, 'lgn-2': 27000}, seed 42
v1 partition: none (each client trained on the full 54000-example split); v1 aggregation: uniform mean

| Round | v1 agg. loss | v1 elapsed (s) | v2 agg. loss (weighted) | v2 lgn-1 | v2 lgn-2 | v2 elapsed (s) |
|---|---|---|---|---|---|---|
| 1 | 1.4637 | 15.46 | 1.5181 | 1.5143 | 1.5219 | 8.48 |
| 2 | 1.4096 | 15.6 | 1.4093 | 1.4077 | 1.4108 | 9.42 |
| 3 | 1.4234 | 15.48 | 1.4108 | 1.4119 | 1.4098 | 10.25 |

v2 monotonic decrease: False; v2 reduction: 7.1%; v2 total round time: 28.1s
v1 monotonic decrease: False; v1 reduction: 2.7%; v1 total round time: 46.5s

## run_simulation_non_iid: config (v2)

- version: v2
- aggregation: fedavg_sample_weighted
- subset_note: full dataset (60k)
- batch_size: 16
- use_amp: True
- amp_active: True
- torch_device: cuda
- cuda_available: True
- torch_cuda_device_name: NVIDIA Graphics Device
- nvidia_smi_gpu_name: NVIDIA Graphics Device
- torch_version: 2.10.0+cu128
- seed: 42

### Aggregated training loss per round (v1 uniform mean vs v2 sample-weighted)

| Round | v1 IID | v2 IID | v1 Dir 0.5 | v2 Dir 0.5 | v1 Dir 0.1 | v2 Dir 0.1 | v1 central | v2 central |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.3061 | 0.3184 | 0.2595 | 0.2634 | 0.1263 | 0.1326 | 0.2073 | 0.1997 |
| 2 | 0.1477 | 0.1468 | 0.1301 | 0.1261 | 0.0743 | 0.0771 | 0.1060 | 0.1016 |
| 3 | 0.1047 | 0.1048 | 0.0978 | 0.0964 | 0.0592 | 0.0595 | 0.0823 | 0.0794 |
| 4 | 0.0907 | 0.0897 | 0.0809 | 0.0799 | 0.0532 | 0.0513 | 0.0713 | 0.0680 |
| 5 | 0.0766 | 0.0771 | 0.0724 | 0.0701 | 0.0462 | 0.0476 | 0.0597 | 0.0588 |

### v2 per-client losses and sample counts


**IID** (client n_k: {'site-1': 20000, 'site-2': 20000, 'site-3': 20000})

| Round | agg (weighted) | agg (uniform) | site-1 | site-2 | site-3 | elapsed (s) |
|---|---|---|---|---|---|---|
| 1 | 0.3184 | 0.3184 | 0.3074 | 0.3148 | 0.3330 | 7.33 |
| 2 | 0.1468 | 0.1468 | 0.1472 | 0.1492 | 0.1438 | 7.83 |
| 3 | 0.1048 | 0.1048 | 0.1053 | 0.1038 | 0.1054 | 12.71 |
| 4 | 0.0897 | 0.0897 | 0.0910 | 0.0930 | 0.0851 | 9.69 |
| 5 | 0.0771 | 0.0771 | 0.0794 | 0.0780 | 0.0739 | 6.74 |
monotonic: True; reduction 75.8%; wall 44.3s

**Dir 0.5** (client n_k: {'site-1': 16986, 'site-2': 23249, 'site-3': 19765})

| Round | agg (weighted) | agg (uniform) | site-1 | site-2 | site-3 | elapsed (s) |
|---|---|---|---|---|---|---|
| 1 | 0.2634 | 0.2672 | 0.3064 | 0.2323 | 0.2630 | 7.08 |
| 2 | 0.1261 | 0.1272 | 0.1385 | 0.1176 | 0.1254 | 8.35 |
| 3 | 0.0964 | 0.0970 | 0.1051 | 0.0922 | 0.0937 | 9.66 |
| 4 | 0.0799 | 0.0808 | 0.0898 | 0.0724 | 0.0803 | 8.92 |
| 5 | 0.0701 | 0.0710 | 0.0800 | 0.0633 | 0.0698 | 7.47 |
monotonic: True; reduction 73.4%; wall 41.5s

**Dir 0.1** (client n_k: {'site-1': 18063, 'site-2': 23662, 'site-3': 18275})

| Round | agg (weighted) | agg (uniform) | site-1 | site-2 | site-3 | elapsed (s) |
|---|---|---|---|---|---|---|
| 1 | 0.1326 | 0.1300 | 0.0924 | 0.1576 | 0.1400 | 10.95 |
| 2 | 0.0771 | 0.0756 | 0.0518 | 0.0917 | 0.0833 | 9.72 |
| 3 | 0.0595 | 0.0586 | 0.0419 | 0.0685 | 0.0653 | 11.28 |
| 4 | 0.0513 | 0.0504 | 0.0375 | 0.0600 | 0.0538 | 7.97 |
| 5 | 0.0476 | 0.0468 | 0.0384 | 0.0553 | 0.0466 | 10.25 |
monotonic: True; reduction 64.1%; wall 50.2s

Centralized: wall 41.7s, steps/epoch 3750

## non_iid_accuracy: meta (v2)

- version: v2
- aggregation: fedavg_sample_weighted
- seed: 42
- num_clients: 3
- num_rounds: 5
- torch_device: cuda
- cuda_available: True
- torch_cuda_device_name: NVIDIA Graphics Device
- nvidia_smi_gpu_name: NVIDIA Graphics Device
- use_amp_requested: True
- amp_active: True
- batch_size: 16
- torch_version: 2.10.0+cu128
- total_wall_sec: 160.52
- partition IID: sizes [20000, 20000, 20000]
- partition Dirichlet_0.5: sizes [16986, 23249, 19765]
- partition Dirichlet_0.1: sizes [18063, 23662, 18275]

### Held-out test accuracy per round: v1 overall | v2 overall | v2 macro

| Round | IID v1 | IID v2 overall | IID v2 macro | Dirichlet_0.5 v1 | Dirichlet_0.5 v2 overall | Dirichlet_0.5 v2 macro | Dirichlet_0.1 v1 | Dirichlet_0.1 v2 overall | Dirichlet_0.1 v2 macro |
|---|---|---|---|---|---|---|---|---|---|
| 0 | n/a | 8.42% | 8.61% | n/a | 8.42% | 8.61% | n/a | 8.42% | 8.61% |
| 1 | 97.49% | 97.33% | 97.31% | 93.25% | 93.90% | 93.90% | 86.38% | 74.14% | 74.42% |
| 2 | 98.54% | 98.43% | 98.42% | 97.67% | 96.99% | 96.98% | 88.81% | 78.87% | 79.25% |
| 3 | 98.74% | 98.86% | 98.85% | 98.40% | 98.08% | 98.07% | 89.99% | 85.83% | 86.11% |
| 4 | 98.96% | 99.01% | 99.00% | 98.32% | 98.09% | 98.08% | 90.16% | 87.64% | 87.93% |
| 5 | 98.97% | 98.94% | 98.92% | 98.62% | 98.55% | 98.55% | 90.54% | 87.97% | 88.27% |

### v2 aggregated (sample-weighted) training loss and elapsed per round

| Round | IID loss | IID elapsed (s) | Dirichlet_0.5 loss | Dirichlet_0.5 elapsed (s) | Dirichlet_0.1 loss | Dirichlet_0.1 elapsed (s) |
|---|---|---|---|---|---|---|
| 1 | 0.3217 | 7.78 | 0.2612 | 11.26 | 0.1320 | 7.04 |
| 2 | 0.1430 | 8.97 | 0.1257 | 11.56 | 0.0778 | 11.08 |
| 3 | 0.1050 | 7.67 | 0.0929 | 8.19 | 0.0604 | 11.21 |
| 4 | 0.0897 | 9.89 | 0.0796 | 10.96 | 0.0523 | 13.44 |
| 5 | 0.0769 | 9.36 | 0.0681 | 12.31 | 0.0439 | 9.38 |

v2 total wall-clock: 160.52s

### v2 per-class accuracy at round 5

| Class | IID | Dirichlet_0.5 | Dirichlet_0.1 |
|---|---|---|---|
| 0 | 99.69% | 99.80% | 99.69% |
| 1 | 99.74% | 99.56% | 99.12% |
| 2 | 98.93% | 99.52% | 9.59% |
| 3 | 99.41% | 99.21% | 81.68% |
| 4 | 99.19% | 99.49% | 98.78% |
| 5 | 98.43% | 99.33% | 98.77% |
| 6 | 98.75% | 98.43% | 98.33% |
| 7 | 98.74% | 97.86% | 99.71% |
| 8 | 98.56% | 98.97% | 98.67% |
| 9 | 97.82% | 93.36% | 98.32% |
