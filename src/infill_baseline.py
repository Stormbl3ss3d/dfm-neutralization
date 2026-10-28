import json, random, argparse, re
from pathlib import Path
import numpy as np, torch, sacrebleu
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
from captum.attr import LayerIntegratedGradients

ROOT = Path.home() / "dfm-neutralization"
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/detector"))
ap.add_argument("--mlm", default="roberta-base")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--max_len", type=int, default=128)
ap.add_argument("--device", default=None)
args = ap.parse_args()
device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
random.seed(0); torch.manual_seed(0)
print("device:", device)

TOKEN_RE = re.compile(r"\w+|[^\w\s]")
det_tok = AutoTokenizer.from_pretrained(args.ckpt)
det = AutoModelForSequenceClassification.from_pretrained(args.ckpt).to(device).eval()
mlm_tok = AutoTokenizer.from_pretrained(args.mlm)
mlm = AutoModelForMaskedLM.from_pretrained(args.mlm).to(device).eval()

test = [json.loads(l) for l in open(ROOT / "data/processed/test.jsonl")]
test = random.sample(test, min(args.n, len(test)))


def fwd(input_ids, attention_mask):
    return det(input_ids=input_ids, attention_mask=attention_mask).logits[:, 1]


lig = LayerIntegratedGradients(fwd, det.roberta.embeddings.word_embeddings)


def word_scores(text):
    spans = [(m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
    enc = det_tok(text, truncation=True, max_length=args.max_len, return_offsets_mapping=True, return_tensors="pt")
    offs = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"].to(device); att = enc["attention_mask"].to(device)
    base = ids.clone()
    special = (ids == det_tok.cls_token_id) | (ids == det_tok.sep_token_id)
    base[~special] = det_tok.pad_token_id
    attr = lig.attribute(ids, baselines=base, additional_forward_args=(att,), n_steps=20, internal_batch_size=10)
    tok_score = attr.sum(-1)[0].detach().cpu().numpy()
    ws = np.zeros(len(spans))
    for (s, e), sc in zip(offs, tok_score):
        if e <= s:
            continue
        for wi, (a, b) in enumerate(spans):
            if s < b and e > a:
                ws[wi] += sc; break
    return ws, spans


def apply(text, span, repl):
    s, e = span
    return re.sub(r"\s+", " ", text[:s] + repl + text[e:]).strip()


@torch.no_grad()
def infill(text, span, orig):
    s, e = span
    masked = text[:s] + mlm_tok.mask_token + text[e:]
    enc = mlm_tok(masked, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)
    pos = (enc["input_ids"][0] == mlm_tok.mask_token_id).nonzero()
    if len(pos) == 0:
        return orig, orig
    logits = mlm(**enc).logits[0, pos[0, 0]]
    cands = [mlm_tok.decode([t]).strip().lower() for t in torch.topk(logits, 30).indices.tolist()]
    cands = [c for c in cands if c]
    unc = cands[0] if cands else orig
    excl = next((c for c in cands if c != orig.lower()), orig)
    return unc, excl


@torch.no_grad()
def p_biased(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        enc = det_tok(texts[i:i + bs], padding=True, truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
        out += torch.softmax(det(**enc).logits.float(), -1)[:, 1].cpu().tolist()
    return out


methods = ["copy", "delete_top1", "infill_top1_unc", "infill_top1_excl", "delete_gold", "infill_gold_unc", "infill_gold_excl"]
hyps = {m: [] for m in methods}
hit1, restore_det, restore_gold, ex_rows = [], [], [], []

for n, r in enumerate(test):
    src = r["src"]; sw = words(src) if False else TOKEN_RE.findall(src)
    ws, spans = word_scores(src)
    top = int(np.argmax(ws)); gold = r["gold_idx"][0] if r["gold_idx"] else top
    hit1.append(int(top in set(r["gold_idx"])))
    unc_d, excl_d = infill(src, spans[top], sw[top])
    unc_g, excl_g = infill(src, spans[gold], sw[gold])
    restore_det.append(int(unc_d == sw[top].lower())); restore_gold.append(int(unc_g == sw[gold].lower()))
    hyps["copy"].append(src)
    hyps["delete_top1"].append(apply(src, spans[top], ""))
    hyps["infill_top1_unc"].append(apply(src, spans[top], unc_d))
    hyps["infill_top1_excl"].append(apply(src, spans[top], excl_d))
    hyps["delete_gold"].append(apply(src, spans[gold], ""))
    hyps["infill_gold_unc"].append(apply(src, spans[gold], unc_g))
    hyps["infill_gold_excl"].append(apply(src, spans[gold], excl_g))
    if (n + 1) % 25 == 0:
        print(f"{n+1}/{len(test)} done", flush=True)

refs = [" ".join(r["tgt_words"]) for r in test]
kinds = np.array([r["kind"] for r in test])
print(f"\nn={len(test)}  detector top-1 hits gold word: {np.mean(hit1):.3f}")
print(f"unconstrained infill restores the ORIGINAL word: at detected pos {np.mean(restore_det):.3f}, at gold pos {np.mean(restore_gold):.3f}")
print(f"mean p(biased): source {np.mean(p_biased([r['src'] for r in test])):.3f}, human neutral {np.mean(p_biased([r['tgt'] for r in test])):.3f}\n")
print(f"{'method':18s} {'exact':>6s} {'del-only':>9s} {'repl-only':>9s} {'BLEU-4':>7s} {'p(biased)':>10s}")
for m in methods:
    h = [" ".join(TOKEN_RE.findall(t)) for t in hyps[m]]
    em = np.array([a == b for a, b in zip(h, refs)])
    bleu = sacrebleu.corpus_bleu(h, [refs], force=True).score
    pb = np.mean(p_biased(hyps[m]))
    print(f"{m:18s} {em.mean():6.3f} {em[kinds=='delete-only'].mean():9.3f} {em[kinds=='replace-only'].mean():9.3f} {bleu:7.2f} {pb:10.3f}")

out = ROOT / "results"; out.mkdir(exist_ok=True)
with open(out / "infill_baseline.jsonl", "w") as f:
    for i, r in enumerate(test):
        f.write(json.dumps({"id": r["id"], "kind": r["kind"], "src": r["src"], "tgt": r["tgt"], **{m: hyps[m][i] for m in methods}}) + "\n")
print("\nper-example outputs saved to results/infill_baseline.jsonl")
print("\n--- 6 replace-only examples ---")
for i in [j for j, r in enumerate(test) if r["kind"] == "replace-only"][:6]:
    print("SRC :", test[i]["src"]); print("TGT :", test[i]["tgt"]); print("INFL:", hyps["infill_top1_excl"][i]); print()
