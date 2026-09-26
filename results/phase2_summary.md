# Phase-2 results

## Self-repair

- **few_shot**: 3/10 -> 6/10 -> 7/10 -> 7/10; mean 1.2 repair calls, \$0.19 per script; unresolved: imagenet_main (preflight/backward_pass); lstm_seq2seq (preflight/import_check); image_classification_from_scratch (preflight/import_check)
- **zero_shot**: 0/10 -> 1/10 -> 1/10 -> 2/10; mean 2.5 repair calls, \$0.22 per script; unresolved: mnist_main (preflight/forward_pass); dcgan_main (preflight/forward_pass); mnist_convnet (preflight/import_check); lstm_seq2seq (preflight/import_check); image_classification_from_scratch (preflight/import_check); imagenet_main (preflight/backward_pass); spleen_segmentation_3d (preflight/forward_pass); mednist_tutorial (preflight/forward_pass)

## Prompt ablation

- **Schema in system prompt (reference)**: 17/20
- **Schema in user message**: 19/20
- **Schema plus corrected exemplar**: 18/20
- **Schema, I3 phrased as a prohibition**: 18/20
- **Exemplar plus invariants, no framework rules**: 16/20
- **Framework rules only, no invariants**: 0/20

## Repeated sampling

- **claude**: 10 scripts with the complete 5-sample grid
- **claude / few_shot_corrected**: 27/50; always 4, never 2, mixed 4
- **claude / structured**: 46/50; always 7, never 0, mixed 3
- **gemini**: 3 scripts with the complete 5-sample grid; excluded for incomplete coverage: image_classification_from_scratch, lstm_seq2seq, mnist_convnet
- **gemini / few_shot_corrected**: 10/15; always 2, never 1, mixed 0
- **gemini / structured**: 5/15; always 1, never 2, mixed 0

## Frozen expansion benchmark

- **dev_new** (n=7): template 6/7, zero_shot 0/7, few_shot_corrected 7/7, structured 5/7
- **holdout** (n=21): template 10/21, zero_shot 0/21, few_shot_corrected 8/21, structured 12/21
- McNemar structured vs template: 7+5 discordant, p=0.7744
- McNemar structured vs few_shot_corrected: 7+3 discordant, p=0.3438
- McNemar structured vs zero_shot: 12+0 discordant, p=0.0005
- McNemar template vs few_shot_corrected: 4+2 discordant, p=0.6875
- adversarial mutants (template): 5/5

## Cost per generation

- self-repair | claude | 37 | 43117 | 3689 | 0.111
- ablation | claude | 106 | 30340 | 7346 | 0.144
- repeated-claude | claude | 89 | 29519 | 6277 | 0.120
- repeated-gemini | gemini | 36 | 4478 | 2789 | 0.028
- expansion | claude | 84 | 35842 | 6695 | 0.170
