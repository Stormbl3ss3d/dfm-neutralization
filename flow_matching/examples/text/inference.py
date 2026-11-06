"""Conditional DFM neutralization with optional classifier guidance (inference only, no retraining).

Two ways to run it:

  # a few sentences from the terminal (prints results)
  python inference.py --checkpoint_path CKPT --input_text "sentence one" "sentence two"

  # the evaluation set (writes a jsonl that src/evaluate_outputs.py scores)
  python inference.py --checkpoint_path CKPT \
      --input_file ~/dfm-neutralization/results/infill_baseline.jsonl \
      --length src --guidance none --out ~/dfm-neutralization/results/dfm_src_none.jsonl

Target length (--length). The model was trained with the target region exactly as long as the
human neutral sentence, and it never learned to emit padding inside that region, so it cannot
shorten a sentence on its own. We therefore choose the length at inference time:
  src             = token length of the biased source (cannot express a deletion)
  detector_delete = token length of the source with the detector's top word removed
                    (needs the `delete_top1` field written by src/infill_baseline.py)
  oracle          = token length of the human neutral reference (uses the gold answer:
                    a diagnostic upper bound, never a system result)

Guidance (--guidance), applied to the denoiser's posterior over clean tokens p(x_1 | x_t) at
every Euler step. The bias classifier only ever sees the generated sentence (<s> target </s>),
never the "Biased: ... Neutral:" prompt, and the objective is log p(neutral).
  none   = plain conditional DFM
  logit  = the original steering: gradient of log p(neutral) w.r.t. the posterior logits,
           L2-normalised per position, added with step --guidance_scale
  taylor = first-order Taylor tilt. For position i and candidate token v,
             log p(neutral | x_i = v) ~ const + g_i . E_v,   g_i = d log p(neutral) / d e_i,
           evaluated at the expected embedding e_i = sum_v p(v) E_v. The posterior becomes
             p~(v) ∝ p(v) * exp(gamma * g_i . E_v),   gamma = --guidance_scale,
           i.e. a first-order approximation of p(x_1) p(neutral | x_1)^gamma per position.
  Both are heuristics on the posterior with a classifier trained on clean text; neither is the
  rate-level guidance of Nisonoff et al. (ICLR 2025). Guidance shifts probabilities; it does not
  guarantee a neutral output.
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from model import Transformer


def biased_label_index(classifier) -> int:
    """Our detector was trained with 1 = biased; fall back to that if labels are generic."""
    for i, name in classifier.config.id2label.items():
        n = str(name).lower()
        if "bias" in n and not any(k in n for k in ("non", "un", "neutral")):
            return int(i)
    return 1


class GuidedWrappedModel(ModelWrapper):
    """Returns p(x_1 | x_t) for the solver, with prompt tokens clamped and optional guidance."""

    def __init__(self, model, prompt_ids, target_mask, starts, budgets, classifier, tokenizer,
                 mode="none", scale=0.0, guide_start=0.0):
        super().__init__(model=model)
        self.prompt_ids, self.target_mask = prompt_ids, target_mask
        self.starts, self.budgets = starts, budgets
        self.classifier, self.mode, self.scale, self.guide_start = classifier, mode, scale, guide_start
        if mode != "none":
            self.E = classifier.get_input_embeddings().weight.detach()  # [V, D]
            self.biased_idx = biased_label_index(classifier)
            ids = torch.tensor([tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id],
                               device=prompt_ids.device)
            self.cls_e, self.sep_e, self.pad_e = self.E[ids]

    def _classifier_input(self, emb):
        """Pack the target region of each row into <s> target </s> <pad>... with an attention mask."""
        B, _, D = emb.shape
        T = int(self.budgets.max())
        ar = torch.arange(T, device=emb.device)
        gather = (self.starts[:, None] + ar[None]).clamp(max=emb.shape[1] - 1)
        valid = ar[None] < self.budgets[:, None]
        tgt = emb.gather(1, gather[..., None].expand(-1, -1, D))
        tgt = torch.where(valid[..., None], tgt, self.pad_e)
        inp = torch.cat([self.cls_e.expand(B, 1, D), tgt, self.pad_e.expand(B, 1, D)], dim=1)
        sep_pos = self.budgets + 1
        sep_slot = F.one_hot(sep_pos, T + 2).bool()[..., None]
        inp = torch.where(sep_slot, self.sep_e, inp)
        att = (torch.arange(T + 2, device=emb.device)[None] <= sep_pos[:, None]).long()
        return inp, att

    def _log_p_neutral(self, emb):
        inp, att = self._classifier_input(emb)
        logits = self.classifier(inputs_embeds=inp, attention_mask=att).logits.float()
        logp = torch.log_softmax(logits, dim=-1)
        neutral = [i for i in range(logp.shape[-1]) if i != self.biased_idx]
        return torch.logsumexp(logp[:, neutral], dim=-1).sum()

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        x_in = torch.where(self.target_mask, x, self.prompt_ids)
        logits = self.model(x_t=x_in, time=t).float()

        if self.mode != "none" and self.scale > 0 and float(t[0]) >= self.guide_start:
            with torch.enable_grad():
                if self.mode == "logit":
                    lg = logits.detach().requires_grad_(True)
                    emb = torch.softmax(lg, -1) @ self.E
                    g = torch.autograd.grad(self._log_p_neutral(emb), lg)[0]
                    step = g / (g.norm(dim=-1, keepdim=True) + 1e-8)
                else:  # taylor
                    emb = (torch.softmax(logits.detach(), -1) @ self.E).requires_grad_(True)
                    g = torch.autograd.grad(self._log_p_neutral(emb), emb)[0]  # [B, L, D]
                    step = g @ self.E.T  # [B, L, V]
                    step = step - step.mean(-1, keepdim=True)
            logits = logits + self.scale * torch.where(self.target_mask[..., None], step, 0.0)

        probs = torch.softmax(logits, dim=-1)
        keep = ~self.target_mask
        probs[keep] = 0.0
        probs[keep, self.prompt_ids[keep]] = 1.0
        return probs


def build_batch(rows, tokenizer, length_mode, max_length, device):
    """Same layout as training: <s> Biased: {src} Neutral: </s> {target} </s> <pad>..."""
    prompt_ids, target_mask, starts, budgets, keep = [], [], [], [], []
    for r in rows:
        prefix = [tokenizer.cls_token_id] + tokenizer(f"Biased: {r['src']} Neutral:",
                                                      add_special_tokens=False)["input_ids"] + [tokenizer.sep_token_id]
        ref_text = {"src": r["src"], "oracle": r.get("tgt"), "detector_delete": r.get("delete_top1")}[length_mode]
        if ref_text is None:
            raise ValueError(f"--length {length_mode} needs field "
                             f"{'tgt' if length_mode == 'oracle' else 'delete_top1'} in the input rows")
        budget = len(tokenizer(" " + ref_text, add_special_tokens=False)["input_ids"])
        budget = min(budget, max_length - 1 - len(prefix))  # mirror training-time truncation
        keep.append(budget > 0)
        budget = max(budget, 1)
        ids = (prefix + [tokenizer.pad_token_id] * budget + [tokenizer.sep_token_id])[:max_length]
        ids += [tokenizer.pad_token_id] * (max_length - len(ids))
        m = [0] * max_length
        for i in range(len(prefix), min(len(prefix) + budget, max_length)):
            m[i] = 1
        prompt_ids.append(ids); target_mask.append(m); starts.append(len(prefix)); budgets.append(budget)
    T = lambda v, dt: torch.tensor(v, dtype=dt, device=device)
    return T(prompt_ids, torch.long), T(target_mask, torch.bool), T(starts, torch.long), T(budgets, torch.long), keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--input_text", nargs="+", default=None)
    ap.add_argument("--input_file", default=None, help="jsonl with src (and tgt, delete_top1, id, kind)")
    ap.add_argument("--n", type=int, default=None, help="only the first n rows of --input_file")
    ap.add_argument("--out", default=None, help="output jsonl (default: print only)")
    ap.add_argument("--length", choices=["src", "detector_delete", "oracle"], default="src")
    ap.add_argument("--guidance", choices=["none", "logit", "taylor"], default="none")
    ap.add_argument("--guidance_scale", type=float, default=0.0)
    ap.add_argument("--guide_start", type=float, default=0.0, help="apply guidance only for t >= this")
    ap.add_argument("--steps", type=int, default=128, help="Euler steps = NFE of the DFM")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tokenizer", default="distilroberta-base")
    ap.add_argument("--classifier_path", default=os.path.expanduser("~/dfm-neutralization/checkpoints/detector"))
    ap.add_argument("--hidden_size", type=int, default=768)
    ap.add_argument("--n_blocks", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--cond_dim", type=int, default=128)
    ap.add_argument("--max_length", type=int, default=128)
    args = ap.parse_args()
    if (args.input_text is None) == (args.input_file is None):
        ap.error("give exactly one of --input_text / --input_file")
    if args.guidance != "none" and args.guidance_scale <= 0:
        ap.error("--guidance logit/taylor needs --guidance_scale > 0")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                          else "mps" if torch.backends.mps.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    print(f"device {device} | length {args.length} | guidance {args.guidance} "
          f"scale {args.guidance_scale} from t>={args.guide_start} | steps {args.steps} | seed {args.seed}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    cfg = OmegaConf.create({"hidden_size": args.hidden_size, "cond_dim": args.cond_dim, "length": args.max_length,
                            "n_blocks": args.n_blocks, "n_heads": args.n_heads, "dropout": 0.0})
    model = Transformer(config=cfg, vocab_size=tokenizer.vocab_size, masked=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device, weights_only=True)["model"])
    model.eval()

    classifier = None
    if args.guidance != "none":
        classifier = AutoModelForSequenceClassification.from_pretrained(args.classifier_path).to(device).eval()
        for p in classifier.parameters():
            p.requires_grad_(False)

    if args.input_file:
        rows = [json.loads(l) for l in open(os.path.expanduser(args.input_file))][: args.n]
    else:
        rows = [{"id": str(i), "src": s} for i, s in enumerate(args.input_text)]

    path = MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=1.0))  # matches config exponent 1
    out_rows, t0 = [], time.time()
    for b in range(0, len(rows), args.batch_size):
        chunk = rows[b: b + args.batch_size]
        prompt_ids, target_mask, starts, budgets, keep = build_batch(
            chunk, tokenizer, args.length, args.max_length, device)
        x_init = torch.randint_like(prompt_ids, high=tokenizer.vocab_size)  # uniform source, as in training
        x_init = torch.where(target_mask, x_init, prompt_ids)
        wrapped = GuidedWrappedModel(model, prompt_ids, target_mask, starts, budgets, classifier, tokenizer,
                                     args.guidance, args.guidance_scale, args.guide_start)
        solver = MixtureDiscreteEulerSolver(model=wrapped, path=path, vocabulary_size=tokenizer.vocab_size)
        with torch.no_grad():
            sample = solver.sample(x_init=x_init, step_size=1.0 / args.steps,
                                   time_grid=torch.tensor([0.0, 1.0]), dtype_categorical=torch.float64)
        for j, r in enumerate(chunk):
            s, n = int(starts[j]), int(budgets[j])
            hyp = tokenizer.decode(sample[j, s: s + n], skip_special_tokens=True).strip() if keep[j] else r["src"]
            out_rows.append({"id": r.get("id"), "kind": r.get("kind"), "src": r["src"], "tgt": r.get("tgt"),
                             "hyp": hyp, "length": args.length, "guidance": args.guidance,
                             "guidance_scale": args.guidance_scale, "guide_start": args.guide_start,
                             "nfe": args.steps, "seed": args.seed, "truncated_prompt": not keep[j]})
        print(f"{len(out_rows)}/{len(rows)} done  ({(time.time() - t0) / len(out_rows):.2f}s/sentence)", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")
        print("saved", args.out)
    for r in out_rows[: 10 if args.input_file else len(out_rows)]:
        print("-" * 60)
        print("BIASED :", r["src"])
        if r.get("tgt"):
            print("HUMAN  :", r["tgt"])
        print("DFM    :", r["hyp"])


if __name__ == "__main__":
    main()
