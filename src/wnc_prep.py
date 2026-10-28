import re, json, difflib, random
from pathlib import Path
from collections import Counter
import sacrebleu

ROOT = Path.home() / "dfm-neutralization"
DATA = ROOT / "data/bias_data/WNC"
OUT = ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)
SPLITS = {"train": "biased.word.train", "dev": "biased.word.dev", "test": "biased.word.test"}
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def words(s):
    return TOKEN_RE.findall(s)


def load(fname):
    rows, bad = [], 0
    with open(DATA / fname, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                bad += 1
                continue
            rows.append({"id": p[0], "src": p[3], "tgt": p[4]})
    return rows, bad


def diff(src, tgt):
    a, b = words(src), words(tgt)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    ops, gold = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        ops.append({"op": tag, "src_span": [i1, i2], "src_text": a[i1:i2], "tgt_text": b[j1:j2]})
        gold.extend(range(i1, i2))
    return a, b, ops, gold


def kind(ops):
    if not ops:
        return "identical"
    t = {o["op"] for o in ops}
    if len(t) > 1:
        return "mixed"
    return {"delete": "delete-only", "insert": "insert-only", "replace": "replace-only"}[t.pop()]


def main():
    ids = {}
    for split, fname in SPLITS.items():
        rows, bad = load(fname)
        kinds, nops, golds, lens, recs = Counter(), Counter(), Counter(), [], []
        for r in rows:
            a, b, ops, gold = diff(r["src"], r["tgt"])
            k = kind(ops)
            kinds[k] += 1
            nops[min(len(ops), 3)] += 1
            golds[min(len(gold), 3)] += 1
            lens.append(len(a))
            recs.append({**r, "src_words": a, "tgt_words": b, "ops": ops, "gold_idx": gold, "kind": k})
        with open(OUT / f"{split}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        ids[split] = {r["id"] for r in rows}
        n = len(rows)
        lens.sort()
        print(f"\n=== {split}: {n} pairs ({bad} malformed lines skipped) ===")
        print("pair types:", {k: f"{100*v/n:.1f}%" for k, v in kinds.most_common()})
        print("edit ops per pair (3 = 3+):", dict(sorted(nops.items())))
        print("gold src tokens per pair (3 = 3+):", dict(sorted(golds.items())))
        print(f"sentence length in words: median {lens[n//2]}, p95 {lens[int(0.95*n)]}, max {lens[-1]}")

        if split in ("dev", "test"):
            refs = [" ".join(r["tgt_words"]) for r in recs]
            copy = [" ".join(r["src_words"]) for r in recs]
            oracle = [" ".join(w for i, w in enumerate(r["src_words"]) if i not in set(r["gold_idx"])) for r in recs]
            for name, hyp in [("copy-input", copy), ("oracle-delete-gold", oracle)]:
                bleu = sacrebleu.corpus_bleu(hyp, [refs]).score
                em = sum(h == r for h, r in zip(hyp, refs)) / n
                print(f"{name:20s} BLEU-4 {bleu:.2f}  exact-match {100*em:.1f}%")

    print("\nid overlap train/dev:", len(ids["train"] & ids["dev"]),
          " train/test:", len(ids["train"] & ids["test"]),
          " dev/test:", len(ids["dev"] & ids["test"]))
    random.seed(0)
    recs = [json.loads(l) for l in open(OUT / "test.jsonl")]
    print("\n--- 6 random test examples ---")
    for r in random.sample(recs, 6):
        print("SRC:", r["src"]); print("TGT:", r["tgt"]); print("OPS:", [(o["op"], o["src_text"], o["tgt_text"]) for o in r["ops"]]); print()


if __name__ == "__main__":
    main()
