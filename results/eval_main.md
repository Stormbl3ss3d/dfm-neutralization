| system | changed | exact | exact del | exact repl | BLEU-4 | BERTScore F1 | p_b(det) | p_b(judge) | neutral% |
|---|---|---|---|---|---|---|---|---|---|
| copy | 0.000 | 0.000 | 0.000 | 0.000 | 91.32 | 0.990 | 0.688 | 0.756 | 0.236 |
| delete_top1 | 1.000 | 0.308 | 0.604 | 0.000 | 90.43 | 0.986 | 0.358 | 0.374 | 0.692 |
| infill_top1_unc | 0.710 | 0.030 | 0.000 | 0.061 | 89.65 | 0.988 | 0.453 | 0.486 | 0.560 |
| infill_top1_excl | 1.000 | 0.032 | 0.000 | 0.066 | 88.20 | 0.986 | 0.434 | 0.461 | 0.584 |
| delete_gold (oracle position) | 1.000 | 0.478 | 0.937 | 0.000 | 94.73 | 0.991 | 0.343 | 0.303 | 0.788 |
| infill_gold_unc (oracle position) | 0.834 | 0.078 | 0.000 | 0.160 | 92.05 | 0.990 | 0.428 | 0.412 | 0.650 |
| infill_gold_excl (oracle position) | 0.998 | 0.086 | 0.000 | 0.176 | 92.11 | 0.990 | 0.412 | 0.388 | 0.674 |
| dfm_src | 0.590 | 0.048 | 0.000 | 0.098 | 83.32 | 0.983 | 0.483 | 0.492 | 0.524 |
| dfm_detdel | 0.994 | 0.296 | 0.573 | 0.008 | 81.66 | 0.981 | 0.335 | 0.316 | 0.756 |
| dfm_oracle (oracle length) | 0.908 | 0.416 | 0.667 | 0.156 | 83.85 | 0.984 | 0.356 | 0.316 | 0.754 |
| human neutral (reference) |  |  |  |  |  |  | 0.343 | 0.272 | 0.806 |

- n = 500 test sentences (255 delete-only, 244 replace-only).
- p_b(det) is circular for any system that uses the detector (IG word choice or guidance).
- p_b(judge) uses valurank/distilroberta-bias; it is outside the pipeline but trained on WNC-derived data.
- Systems marked 'oracle position' or with length=oracle use the gold answer and are upper bounds.
