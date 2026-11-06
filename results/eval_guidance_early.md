| system | changed | exact | exact del | exact repl | BLEU-4 | p_b(det) | p_b(judge) | neutral% |
|---|---|---|---|---|---|---|---|---|
| copy | 0.000 | 0.000 | 0.000 | 0.000 | 90.88 | 0.660 | 0.718 | 0.295 |
| delete_top1 | 1.000 | 0.255 | 0.586 | 0.000 | 89.16 | 0.351 | 0.366 | 0.710 |
| infill_top1_unc | 0.690 | 0.045 | 0.000 | 0.080 | 89.12 | 0.458 | 0.482 | 0.580 |
| infill_top1_excl | 1.000 | 0.050 | 0.000 | 0.088 | 87.53 | 0.435 | 0.458 | 0.595 |
| delete_gold (oracle position) | 1.000 | 0.415 | 0.954 | 0.000 | 93.93 | 0.336 | 0.295 | 0.810 |
| infill_gold_unc (oracle position) | 0.840 | 0.110 | 0.000 | 0.195 | 91.93 | 0.431 | 0.428 | 0.635 |
| infill_gold_excl (oracle position) | 1.000 | 0.125 | 0.000 | 0.221 | 92.10 | 0.411 | 0.399 | 0.665 |
| g200_detdel_none | 1.000 | 0.265 | 0.586 | 0.018 | 81.13 | 0.329 | 0.292 | 0.810 |
| g200_detdel_logit2 | 1.000 | 0.245 | 0.563 | 0.000 | 80.86 | 0.317 | 0.282 | 0.815 |
| human neutral (reference) |  |  |  |  |  | 0.325 | 0.262 | 0.820 |

- n = 200 test sentences (87 delete-only, 113 replace-only).
- p_b(det) is circular for any system that uses the detector (IG word choice or guidance).
- p_b(judge) uses valurank/distilroberta-bias; it is outside the pipeline but trained on WNC-derived data.
- Systems marked 'oracle position' or with length=oracle use the gold answer and are upper bounds.
