set -x
CKPT=~/dfm-neutralization/models/checkpoint_v3.pth
IN=~/dfm-neutralization/results/infill_baseline.jsonl
R=~/dfm-neutralization/results
run() { python inference.py --checkpoint_path $CKPT --input_file $IN --batch_size 32 "$@"; }

# 0. baselines with the capitalisation fix (same 500 sentences)
(cd ~/dfm-neutralization && python src/infill_baseline.py --n 500)

# 1. main table: unguided DFM, all 500 sentences
run --length src             --out $R/dfm_src.jsonl
run --length detector_delete --out $R/dfm_detdel.jsonl
run --length oracle          --out $R/dfm_oracle.jsonl

# 2. guidance + NFE sweep, first 200 sentences (guided runs are slower)
run --n 200 --length detector_delete                                     --out $R/g200_detdel_none.jsonl
run --n 200 --length detector_delete --guidance logit  --guidance_scale 2   --out $R/g200_detdel_logit2.jsonl
run --n 200 --length detector_delete --guidance taylor --guidance_scale 10  --out $R/g200_detdel_taylor10.jsonl
run --n 200 --length detector_delete --guidance taylor --guidance_scale 100 --out $R/g200_detdel_taylor100.jsonl
run --n 200 --length detector_delete --steps 32                          --out $R/g200_detdel_nfe32.jsonl

# 3. tables
cd ~/dfm-neutralization
python src/evaluate_outputs.py --dfm results/dfm_src.jsonl results/dfm_detdel.jsonl results/dfm_oracle.jsonl --out results/eval_main.md
python src/evaluate_outputs.py --dfm results/g200_*.jsonl --out results/eval_guidance.md
echo ALL DONE
