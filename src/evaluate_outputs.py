"""Score every system on the same sentences with the same metrics.

  python src/evaluate_outputs.py \
      --baseline results/infill_baseline.jsonl \
      --dfm results/dfm_*.jsonl \
      --out results/eval_table.md

Systems: every method column of the baseline file (copy, delete_top1, ...) plus one system per
DFM jsonl (its `hyp` field). Only ids present in ALL files are scored, so every row of the table
is on the same sentences.

Metrics
  changed    fraction of outputs that differ from the source (copy-input scores BLEU ~91 while
             changing nothing, so BLEU alone cannot tell systems apart)
  exact      exact match with the human neutral sentence, overall / delete-only / replace-only
  BLEU-4     sacrebleu on word-tokenised text (same tokenisation as wnc_prep / infill_baseline)
  BERTScore  F1 against the human reference (skip with --no_bertscore)
  p_b(det)   mean p(biased) from OUR detector. CIRCULAR: the same model picks the words to
             delete/infill and guides the DFM, so it favours those systems by construction.
  p_b(judge) mean p(biased) from a classifier that is NOT used anywhere in the pipeline.
             Default is the public valurank/distilroberta-bias. It was trained on WNC-derived
             data and may have seen these test sentences, so it is a separate judge, not a
             fully independent one. Pass --judge to use another (e.g. one trained on a disjoint
             split with src/train_detector.py --skip 20000 --model bert-base-uncased).
  neutral%   fraction the judge calls neutral (p_biased < 0.5)
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import sacrebleu
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path.home() / "dfm-neutralization"
TOKEN_RE = re.compile(r"\w+|[^\w\s]")
BASELINE_METHODS = ["copy", "delete_top1", "infill_top1_unc", "infill_top1_excl",
                    "delete_gold", "infill_gold_unc", "infill_gold_excl"]
ORACLE_NOTE = {"delete_gold", "infill_gold_unc", "infill_gold_excl"}

ap = argparse.ArgumentParser()
ap.add_argument("--baseline", default=str(ROOT / "results/infill_baseline.jsonl"))
ap.add_argument("--dfm", nargs="*", default=[])
ap.add_argument("--detector", default=str(ROOT / "checkpoints/detector"))
ap.add_argument("--judge", default="valurank/distilroberta-bias")
ap.add_argument("--judge_biased_idx", type=int, default=None, help="override if the judge's labels are generic")
ap.add_argument("--no_judge", action="store_true")
ap.add_argument("--no_bertscore", action="store_true")
ap.add_argument("--bertscore_model", default="roberta-large")
ap.add_argument("--max_len", type=int, default=128)
ap.add_argument("--device", default=None)
ap.add_argument("--out", default=str(ROOT / "results/eval_table.md"))
args = ap.parse_args()
device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

wt = lambda s: " ".join(TOKEN_RE.findall(s))


def load(p):
    return {str(r["id"]): r for r in map(json.loads, open(Path(p).expanduser()))}


base = load(args.baseline)
dfm = {Path(p).stem: load(p) for p in args.dfm}
ids = set(base)
for d in dfm.values():
    ids &= set(d)
ids = sorted(ids)
if not ids:
    raise SystemExit("no shared ids between the files (run inference.py with --input_file = the baseline jsonl)")
print(f"device {device} | scoring {len(ids)} shared sentences")

src = [base[i]["src"] for i in ids]
refs = [wt(base[i]["tgt"]) for i in ids]
kinds = np.array([base[i]["kind"] for i in ids])
systems = {m: [base[i][m] for i in ids] for m in BASELINE_METHODS if m in base[ids[0]]}
for name, d in dfm.items():
    label = name + (" (oracle length)" if d[ids[0]].get("length") == "oracle" else "")
    systems[label] = [d[i]["hyp"] for i in ids]
for name, d in dfm.items():
    r0 = d[ids[0]]
    print(f"{name}: length={r0.get('length')} guidance={r0.get('guidance')} scale={r0.get('guidance_scale')} "
          f"nfe={r0.get('nfe')} seed={r0.get('seed')}")


def biased_index(model, override):
    if override is not None:
        return override
    for i, name in model.config.id2label.items():
        n = str(name).lower()
        if "bias" in n and not any(k in n for k in ("non", "un", "neutral")):
            return int(i)
    if all(str(v).startswith("LABEL_") for v in model.config.id2label.values()):
        return 1
    raise SystemExit(f"cannot tell which label is 'biased' in {model.config.id2label}; pass --judge_biased_idx")


def scorer(path, override=None):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    bi = biased_index(model, override)
    print(f"classifier {path}: labels {model.config.id2label}, using index {bi} as 'biased'")

    @torch.no_grad()
    def p_biased(texts, bs=64):
        out = []
        for k in range(0, len(texts), bs):
            enc = tok(texts[k:k + bs], padding=True, truncation=True, max_length=args.max_len,
                      return_tensors="pt").to(device)
            out += torch.softmax(model(**enc).logits.float(), -1)[:, bi].cpu().tolist()
        return np.array(out)

    return p_biased


det = scorer(args.detector)
judge = None if args.no_judge else scorer(args.judge, args.judge_biased_idx)

bert_f1 = None
if not args.no_bertscore:
    from bert_score import score as bertscore
    bert_f1 = lambda h: bertscore(h, [base[i]["tgt"] for i in ids], model_type=args.bertscore_model,
                                  lang="en", device=device, verbose=False)[2].numpy()

rows = []
ref_row = {"system": "human neutral (reference)", "p_b(det)": det([base[i]["tgt"] for i in ids]).mean()}
if judge:
    pj = judge([base[i]["tgt"] for i in ids])
    ref_row.update({"p_b(judge)": pj.mean(), "neutral%": (pj < 0.5).mean()})
for name, hyps in systems.items():
    h = [wt(x) for x in hyps]
    em = np.array([a == b for a, b in zip(h, refs)])
    row = {"system": name + (" (oracle position)" if name in ORACLE_NOTE else ""),
           "changed": np.mean([a != wt(s) for a, s in zip(h, src)]),
           "exact": em.mean(), "exact del": em[kinds == "delete-only"].mean(),
           "exact repl": em[kinds == "replace-only"].mean(),
           "BLEU-4": sacrebleu.corpus_bleu(h, [refs], force=True).score,
           "p_b(det)": det(hyps).mean()}
    if bert_f1:
        row["BERTScore F1"] = bert_f1(hyps).mean()
    if judge:
        pj = judge(hyps)
        row.update({"p_b(judge)": pj.mean(), "neutral%": (pj < 0.5).mean()})
    rows.append(row)
    print(f"scored {name}", flush=True)
rows.append(ref_row)

cols = ["system", "changed", "exact", "exact del", "exact repl", "BLEU-4", "BERTScore F1",
        "p_b(det)", "p_b(judge)", "neutral%"]
cols = [c for c in cols if any(c in r for r in rows)]
def fmt(c, v):
    if v is None:
        return ""
    if c == "BLEU-4":
        return f"{v:.2f}"
    return f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)


lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
lines += ["| " + " | ".join(fmt(c, r.get(c)) for c in cols) + " |" for r in rows]
n_del, n_rep = int((kinds == "delete-only").sum()), int((kinds == "replace-only").sum())
notes = [f"n = {len(ids)} test sentences ({n_del} delete-only, {n_rep} replace-only).",
         "p_b(det) is circular for any system that uses the detector (IG word choice or guidance).",
         (f"p_b(judge) uses {args.judge}; it is outside the pipeline but trained on WNC-derived data."
          if args.judge == "valurank/distilroberta-bias" else f"p_b(judge) uses {args.judge}, which is not used in the pipeline.")
         if judge else "No separate judge was run.",
         "Systems marked 'oracle position' or with length=oracle use the gold answer and are upper bounds."]
table = "\n".join(lines) + "\n\n" + "\n".join("- " + n for n in notes) + "\n"
print("\n" + table)
Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
Path(args.out).expanduser().write_text(table)
print("saved", args.out)
