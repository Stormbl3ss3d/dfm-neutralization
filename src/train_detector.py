import json, random, time, argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

ROOT = Path.home() / "dfm-neutralization"
PROC = ROOT / "data/processed"
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="distilroberta-base")
ap.add_argument("--n_pairs", type=int, default=20000)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--max_len", type=int, default=128)
ap.add_argument("--out", default=str(ROOT / "checkpoints/detector"))
args = ap.parse_args()

random.seed(0); torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)


def load(split):
    return [json.loads(l) for l in open(PROC / f"{split}.jsonl")]


train = load("train"); random.shuffle(train); train = train[: args.n_pairs]
data = [(r["src"], 1) for r in train] + [(r["tgt"], 0) for r in train]

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
steps_total = args.epochs * ((len(data) + args.bs - 1) // args.bs)
sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps_total), steps_total)


def enc(texts):
    return tok(texts, padding=True, truncation=True, max_length=args.max_len, return_tensors="pt").to(device)


@torch.no_grad()
def p_biased(texts, bs=64):
    model.eval(); out = []
    for i in range(0, len(texts), bs):
        logits = model(**enc(texts[i:i + bs])).logits
        out += torch.softmax(logits.float(), -1)[:, 1].cpu().tolist()
    return out


def evaluate(split):
    rows = load(split)
    ps, pt = p_biased([r["src"] for r in rows]), p_biased([r["tgt"] for r in rows])
    y = [1] * len(rows) + [0] * len(rows); pred = [int(p > 0.5) for p in ps + pt]
    acc = sum(a == b for a, b in zip(y, pred)) / len(y)
    pair = sum(a > b for a, b in zip(ps, pt)) / len(rows)
    print(f"[{split}] sentence acc {acc:.3f}  F1(biased) {f1_score(y, pred):.3f}  pairwise acc {pair:.3f}")


t0, step = time.time(), 0
for ep in range(args.epochs):
    random.shuffle(data); model.train()
    for i in range(0, len(data), args.bs):
        batch = data[i:i + args.bs]
        x = enc([t for t, _ in batch]); y = torch.tensor([l for _, l in batch], device=device)
        loss = model(**x, labels=y).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(); step += 1
        if step % 100 == 0:
            el = time.time() - t0
            print(f"step {step}/{steps_total} loss {loss.item():.4f} elapsed {el/60:.1f}m eta {(el/step)*(steps_total-step)/60:.1f}m", flush=True)
    evaluate("dev")

evaluate("test")
Path(args.out).mkdir(parents=True, exist_ok=True)
model.save_pretrained(args.out); tok.save_pretrained(args.out)
print("saved to", args.out)
