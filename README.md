# AutoFL: Experimental Results and Figures (v2.0)

Raw outputs, run notes, and figures for:

> Yen-Jung Chiu and Chao-Chun Chuang. *Schema-Augmented LLM Prompting for Converting ML Training Scripts to Federated Learning Clients.* ACM Transactions on Software Engineering and Methodology (resubmission under review, 2026).

- Concept DOI (always resolves to the latest version): https://doi.org/10.5281/zenodo.20156953
- Companion source-code deposit (converters, validator, runners, generated clients): https://doi.org/10.5281/zenodo.20156961 ([GitHub](https://github.com/holiday01/autofl-system))
- v1.0 (as submitted 2026-05-14): https://doi.org/10.5281/zenodo.20156954

## What changed in v2.0

- v1.0 files are kept unchanged under their original names (`results/benchmark.csv`, `fl_results/simulation_results.json`, ...). They are the exact outputs behind the 2026-05-14 submission.
- `*_v2.*` files and `fl_results/v2/` are the corrected reruns: re-validation of every archived generated client with the corrected validator (missing dependencies are Stage 1 failures; `data_path_exists` records whether real data was present), and FL simulations with sample-weighted FedAvg, disjoint IID shards, and seed 42.
- New experiments of the resubmission: template converter baseline, iterative self-repair, prompt ablation, repeated sampling, frozen expansion benchmark, and source-to-converted differential tests.
- The compiled manuscript PDF is no longer included.

## File map

| File | Contents |
|---|---|
| `results/benchmark.csv` (v1), `results/benchmark_v2.csv` | Primary benchmark: 10 DL scripts x 4 strategies, first failing stage, coverage, `e2e_runnable` |
| `results/benchmark_v2_with_template.csv`, `results/template_baseline.csv` | The same with the template converter as a fifth strategy |
| `results/benchmark_v1_v2_comparison.{csv,md}` | Row-by-row changes between the v1 and v2 validator |
| `results/benchmark_extended.csv` (v1), `results/benchmark_extended_v2.csv` | Non-DL extension (scikit-learn x 2, XGBoost x 1) |
| `results/comparison_multimodel.csv` | Cross-provider six-script subset (Claude, Gemini 2.5 Flash, Llama 3 8B) |
| `results/exemplar_correction*.csv` | Exemplar correction: dry-run patch analysis and real-API Gemini runs (v1 and v2) |
| `results/self_repair_claude.csv` | Iterative self-repair, zero-shot and few-shot bases, rounds 0-3, per-call tokens and cost |
| `results/ablation_claude.csv` | Prompt ablation, 6 conditions x 10 scripts x 2 generations |
| `results/repeated_claude.csv`, `results/repeated_gemini.csv` | Repeated sampling, 5 generations per script and strategy |
| `results/expansion.csv`, `results/expansion_template_baseline.csv`, `results/expansion_manifest.csv` | Frozen expansion benchmark (development, holdout, adversarial) with commit hashes and source checksums |
| `results/differential_test.csv` | Source-to-converted differential tests on seven pairs |
| `results/hardware_sweep.csv` | Batch-size x learning-rate sweep, CPU-only MNIST |
| `results/non_dl_proxy_validation.csv` | PyTorch surrogate vs. native scikit-learn/XGBoost accuracy |
| `results/phase2_tables.tex`, `results/phase2_summary.md` | Tables generated from the CSVs above by `eval/make_phase2_tables.py` |
| `results/*_notes.md`, `results/logs_v2/` | Machine-generated provenance and run logs (config, seed, device, command line) |
| `fl_results/*.json` (v1), `fl_results/v2/*.json` | FL simulation, non-IID loss trajectories, per-round held-out accuracy (overall and macro) |
| `results/figures/`, `results/figures_v2/` | Figures regenerated from the CSV/JSON files (v1 and v2 runs) |
| `figures/` | The figures as they appear in the manuscript |

Rows whose `error_stage` is `generation` record a provider call that failed (for example a rate limit), not a generated client; they are excluded from every rate in the paper.

## Re-deriving statistics

Every script is evaluated under every strategy, so the paper uses the exact McNemar test on discordant pairs:

```python
import pandas as pd
from scipy.stats import binomtest
d = pd.read_csv("results/benchmark_v2_with_template.csv")
ok = d.pivot_table(index="script_name", columns="method", values="e2e_runnable",
                   aggfunc="first").astype(str) == "True"
b = (ok.structured & ~ok.few_shot).sum(); c = (~ok.structured & ok.few_shot).sum()
print(b, c, binomtest(b, b + c, 0.5).pvalue)   # 7 0 0.015625
```

`eval/check_consistency.py` in the source deposit recomputes the paper's numbers from these files.

## License

CC-BY-4.0.

## Citation

```bibtex
@misc{autofl_results_2026,
  title  = {AutoFL: Experimental Results and Figures},
  author = {Chiu, Yen-Jung and Chuang, Chao-Chun},
  year   = {2026},
  doi    = {10.5281/zenodo.20156953},
  note   = {Version 2.0. v1.0: 10.5281/zenodo.20156954}
}
```
