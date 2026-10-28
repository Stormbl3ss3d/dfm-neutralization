import json, random, argparse, re
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score
from captum.attr import LayerIntegratedGradients

ROOT = Path.home() / "dfm-neutralization"
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/detector"))
ap.add_argument("--n_attr", type=int, default=300)
ap.add_argument("--device", default=None)
ap.add_argument("--max_len", type=int, default=128)
args = ap.parse_args()
device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
random.seed(0); np.random.seed(0); torch.manual_seed(0)
print("device:", device)

tok = AutoTokenizer.from_pretrained(args.ckpt)
model = AutoModelForSequenceClassification.from_pretrained(args.ckpt).to(device).eval()
test = [json.loads(l) for l in open(ROOT / "data/processed/test.jsonl")]
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


@torch.no_grad()
def p_biased_ids(ids_list, bs=64):
    out = []
    for i in range(0, len(ids_list), bs):
        chunk = ids_list[i:i + bs]
        L = max(len(x) for x in chunk)
        ids = torch.full((len(chunk), L), tok.pad_token_id)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, x in enumerate(chunk):
            ids[j, :len(x)] = torch.tensor(x); att[j, :len(x)] = 1
        logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
        out += torch.softmax(logits.float(), -1)[:, 1].cpu().tolist()
    return out


def encode(texts):
    return [tok(t, truncation=True, max_length=args.max_len)["input_ids"] for t in texts]


src_ids = encode([r["src"] for r in test]); tgt_ids = encode([r["tgt"] for r in test])
ps = np.array(p_biased_ids(src_ids)); pt = np.array(p_biased_ids(tgt_ids))
y = np.r_[np.ones(len(ps)), np.zeros(len(pt))]; p = np.r_[ps, pt]
print(f"\n[clean] acc {((p > 0.5) == y).mean():.3f}  AUC {roc_auc_score(y, p):.3f}")
for k in ["delete-only", "replace-only"]:
    idx = [i for i, r in enumerate(test) if r["kind"] == k]
    print(f"[pairwise acc, {k}] n={len(idx)}: {np.mean(ps[idx] > pt[idx]):.3f}")


def mask_ids(ids, frac, rng):
    ids = list(ids)
    pos = list(range(1, len(ids) - 1))
    for i in rng.sample(pos, int(round(frac * len(pos)))):
        ids[i] = tok.mask_token_id
    return ids


print("\nmask frac | acc | AUC | mean p(biased) on biased / neutral")
for frac in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
    rng = random.Random(1)
    a = np.array(p_biased_ids([mask_ids(x, frac, rng) for x in src_ids]))
    b = np.array(p_biased_ids([mask_ids(x, frac, rng) for x in tgt_ids]))
    pp = np.r_[a, b]
    print(f"{frac:.1f} | {((pp > 0.5) == y).mean():.3f} | {roc_auc_score(y, pp):.3f} | {a.mean():.3f} / {b.mean():.3f}")


def fwd(input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask).logits[:, 1]


lig = LayerIntegratedGradients(fwd, model.roberta.embeddings.word_embeddings)


def word_scores(r):
    enc = tok(r["src"], truncation=True, max_length=args.max_len, return_offsets_mapping=True, return_tensors="pt")
    offs = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"].to(device); att = enc["attention_mask"].to(device)
    base = ids.clone()
    special = (ids == tok.cls_token_id) | (ids == tok.sep_token_id)
    base[~special] = tok.pad_token_id
    attr = lig.attribute(ids, baselines=base, additional_forward_args=(att,), n_steps=20, internal_batch_size=10)
    tok_score = attr.sum(-1)[0].detach().cpu().numpy()
    spans = [(m.start(), m.end()) for m in TOKEN_RE.finditer(r["src"])]
    ws = np.zeros(len(spans))
    for (s, e), sc in zip(offs, tok_score):
        if e <= s:
            continue
        for wi, (a, b) in enumerate(spans):
            if s < b and e > a:
                ws[wi] += sc; break
    return ws


print(f"\nattribution (Integrated Gradients) on {args.n_attr} random test pairs")
res = {"all": [], "delete-only": [], "replace-only": []}
for r in random.sample(test, min(args.n_attr, len(test))):
    if not r["gold_idx"] or r["kind"] not in res:
        continue
    ws = word_scores(r); n = len(ws); gold = set(r["gold_idx"])
    order = list(np.argsort(-ws)); rnd = list(range(n)); random.shuffle(rnd)
    rec = {}
    for k in (1, 3, 5):
        rec[f"top{k}"] = int(any(i in gold for i in order[:k]))
        rec[f"rand{k}"] = int(any(i in gold for i in rnd[:k]))
    res["all"].append(rec); res[r["kind"]].append(rec)
for name, recs in res.items():
    if recs:
        print(f"[{name}] n={len(recs)}  " + "  ".join(
            f"hit@{k}: {np.mean([x[f'top{k}'] for x in recs]):.2f} (random {np.mean([x[f'rand{k}'] for x in recs]):.2f})" for k in (1, 3, 5)))
