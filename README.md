# Text Neutralization with Classifier-Guided Discrete Flow Matching

Generative AI course project. Given a Wikipedia sentence with subjective bias, the system finds the biased word with a classifier and rewrites the sentence into a neutral point of view with a conditional **Discrete Flow Matching (DFM)** model. The project's twist is **classifier guidance**: the bias classifier steers the DFM at every sampling step.

Data: the single-word-edit subset of the **Wiki Neutrality Corpus** (WNC; Pryzant et al., AAAI 2020). Generator: adapted from Meta's [`flow_matching`](https://github.com/facebookresearch/flow_matching) text example (Gat et al., NeurIPS 2024).

## Main results

500 WNC test sentences (255 delete-only, 244 replace-only), seed 0, 128 sampling steps. *Exact* = output identical to the human neutral edit. *Judged neutral* = share a separate classifier ([`valurank/distilroberta-bias`](https://huggingface.co/valurank/distilroberta-bias)) calls neutral. Full tables: [`results/eval_main.md`](results/eval_main.md), [`results/eval_guidance.md`](results/eval_guidance.md).

| System | Exact (all) | Exact (delete) | Exact (replace) | BLEU-4 | BERTScore F1 | Judged neutral |
| --- | --- | --- | --- | --- | --- | --- |
| Copy input | 0.0% | 0.0% | 0.0% | 91.32 | 0.990 | 23.6% |
| Delete detector's top word | 30.8% | 60.4% | 0.0% | 90.43 | 0.986 | 69.2% |
| Infill detector's top word (masked LM) | 3.2% | 0.0% | 6.6% | 88.20 | 0.986 | 58.4% |
| **DFM, output length = source** | 4.8% | 0.0% | **9.8%** | 83.32 | 0.983 | 52.4% |
| **DFM, output length = source − detector's word** | **29.6%** | 57.3% | 0.8% | 81.66 | 0.981 | **75.6%** |
| DFM, output length = gold *(oracle)* | 41.6% | 66.7% | 15.6% | 83.85 | 0.984 | 75.4% |
| Human neutral edit | | | | | | 80.6% |

**Findings**

1. **Output length is the bottleneck.** The DFM fills a fixed number of slots and was never trained to leave one empty, so it cannot delete a word on its own. The same model goes from 4.8% (source length) to 29.6% (length set by the detector) to 41.6% (gold length) exact match. With the detector's length it matches, but does not beat, the deletion baseline.
2. **On replacements the DFM beats masked-LM infilling** (9.8% vs 6.6% exact) without being told which word to edit. It rewrites the whole sentence, so it sometimes alters neighbouring words, which lowers BLEU.
3. **Guidance shifts scores slightly but does not add correct edits.** On the first 200 sentences, logit guidance (s = 2) lowers the judge's mean p(biased) from 0.292 to 0.282, while exact match goes from 26.5% to 24.5% (a difference within noise). Sampling with **32 steps instead of 128** gives the same quality at a quarter of the cost.

**Evaluation caveats.** Our own detector picks the word to delete and guides the DFM, so its scores for those systems are circular; neutrality is therefore judged by a separate public classifier. That judge was also trained on WNC-derived data, so it is separate but not fully independent. Both classifiers rank the systems in the same order. Sampling is random: results are for seed 0.

## How it works

| Component | What it does | Code |
| --- | --- | --- |
| Data preparation | Reads `biased.word.{train,dev,test}`, word-diffs each pair into delete / replace edits and gold edited positions | `src/wnc_prep.py` |
| Bias detector | `distilroberta-base` fine-tuned on 20,000 pairs (biased = 1, neutral = 0), 1 epoch. Test: accuracy 0.720, F1 0.704, pairwise accuracy 0.827 | `src/train_detector.py` |
| Attribution | Layer Integrated Gradients (Captum) on the embedding layer; sub-word scores summed into words; top word matches the human edit about half the time | `src/eval_detector.py` |
| Baselines | Copy; delete or masked-LM-infill (roberta-base) the detector's top word or the gold word | `src/infill_baseline.py` |
| Conditional DFM | Sequence `<s> Biased: {src} Neutral: </s> {tgt} </s>`; only target tokens are noised on the mixture path (uniform source, linear schedule), prompt tokens stay clean; cross-entropy on target tokens only. 6-block DiT, hidden size 768, embeddings initialised from distilroberta-base | `flow_matching/examples/text/data/wnc_data.py`, `logic/training.py`, `train.py` |
| Guided sampling | Euler solver; at each step the detector scores the expected generated sentence and the gradient of log p(neutral) is added to the DFM logits (normalised per position, step size s). Output length: `src`, `detector_delete` or `oracle` | `flow_matching/examples/text/inference.py` |
| Evaluation | Scores every system on the same sentence ids: changed rate, exact match by edit type, BLEU-4, BERTScore, p(biased) from the detector and from the separate judge | `src/evaluate_outputs.py` |

This guidance is a heuristic on the clean-token prediction with a classifier trained on clean text. It is not the rate-level guidance of Nisonoff et al. (ICLR 2025), which requires a classifier trained on noisy inputs.

## Repository layout

```
src/                         data prep, detector, attribution, baselines, evaluator
flow_matching/               Meta's flow_matching library (CC BY-NC 4.0)
  examples/text/             conditional DFM: data, training, model, inference.py, run_all.sh
results/                     per-sentence outputs (*.jsonl) and metric tables (*.md)
```

Data (`data/`), checkpoints (`checkpoints/`, `models/`) and training outputs are not in the repository.

## Setup

Python 3.11, PyTorch 2.1 or newer. Runs on Apple Silicon (`mps`) or a single CUDA GPU.

```bash
conda create -n dfm python=3.11
conda activate dfm
pip install torch transformers captum sacrebleu bert-score scikit-learn omegaconf hydra-core einops torchdiffeq
pip install -e flow_matching
```

Download WNC from [rpryzant/neutralizing-bias](https://github.com/rpryzant/neutralizing-bias) and extract it so the files sit in `data/bias_data/WNC/`. The scripts expect the repository at `~/dfm-neutralization`.

## Reproducing the results

```bash
python src/wnc_prep.py
python src/train_detector.py
python src/eval_detector.py
python src/infill_baseline.py --n 500

cd flow_matching/examples/text
python run_train.py
bash run_all.sh

cd ~/dfm-neutralization
python src/evaluate_outputs.py --dfm results/dfm_src.jsonl results/dfm_detdel.jsonl results/dfm_oracle.jsonl --out results/eval_main.md
python src/evaluate_outputs.py --dfm results/g200_detdel_none.jsonl results/g200_detdel_logit2.jsonl results/g200_detdel_nfe32.jsonl --out results/eval_guidance.md
```

`run_all.sh` sets the DFM checkpoint path in its first lines; edit `CKPT` to point at your checkpoint. To try a single sentence:

```bash
cd flow_matching/examples/text
python inference.py --checkpoint_path PATH/TO/checkpoint.pth --length src --input_text "the avalanche lost to the underdog minnesota wild in the playoffs."
```

## Limitations and next steps

- The DFM cannot choose its output length. Training with the target padded to the source length would let it learn to emit padding, and therefore to delete.
- The guidance classifier is trained on clean text and applied to partly uncertain predictions; a classifier trained on noisy inputs (Nisonoff et al.) is the principled alternative.
- Results are from one seed; several seeds and a human spot check would make the comparison more reliable.

## References

- Gat et al. *Discrete Flow Matching.* NeurIPS 2024. [arXiv:2407.15595](https://arxiv.org/abs/2407.15595)
- Pryzant et al. *Automatically Neutralizing Subjective Bias in Text.* AAAI 2020. [arXiv:1911.09709](https://arxiv.org/abs/1911.09709)
- Nisonoff et al. *Unlocking Guidance for Discrete State-Space Diffusion and Flow Models.* ICLR 2025. [arXiv:2406.01572](https://arxiv.org/abs/2406.01572)
- Lipman et al. *Flow Matching Guide and Code.* [arXiv:2412.06264](https://arxiv.org/abs/2412.06264)
- Sundararajan et al. *Axiomatic Attribution for Deep Networks.* ICML 2017. [arXiv:1703.01365](https://arxiv.org/abs/1703.01365)
- Zhang et al. *BERTScore: Evaluating Text Generation with BERT.* ICLR 2020. [arXiv:1904.09675](https://arxiv.org/abs/1904.09675)

## Licences

`flow_matching/` is Meta's code under CC BY-NC 4.0 (see `flow_matching/LICENSE`). WNC is distributed by Pryzant et al. under the MIT licence.
