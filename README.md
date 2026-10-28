# Subjective Bias Neutralization via Conditional Discrete Flow Matching (DFM)

This repository implements a generative style-transfer pipeline to automatically neutralize subjective bias in text (using the **Wikipedia Neutrality Corpus (WNC)**). 

Rather than relying on standard left-to-right autoregressive decoding, this project adapts **Meta AI's Discrete Flow Matching (`flow_matching`)** framework—originally built strictly for *unconditional* text generationto perform **conditional sequence-to-sequence denoising**, paired with an Explainable AI (XAI) bias detector.



##  Summary of Custom Engineering & Contributions

Because this project builds upon the official [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching) library, our custom implementations and architectural modifications are localized to the following modules:

### 1. Explainable Bias Detection & Baselines (100% Custom)
* **`train_detector.py` / `wnc_prep.py`**: Fine-tunes a `distilroberta-base` classifier on WNC pairs (achieving **84.0% pairwise accuracy**) and applies **Captum Layer Integrated Gradients** to isolate human-edited biased tokens (**51.6% Top-1 gold word detection**).

### 2. Conditional Discrete Flow Matching Architecture (Custom & Modified)
Located in `flow_matching/examples/text/`:
* **`data/wnc_data.py` *(Custom Script)***: Custom PyTorch Dataset & DataLoader formatting WNC pairs into `[CLS] Biased: <src> [SEP] Neutral: <tgt> [SEP]` and dynamically computing a binary `target_mask` tensor (`0` for prompt/padding, `1` for target generation tokens).
* **`logic/training.py` *(Modified Core Math)***:
  * **Forward Flow Clamping**: Patched `step()` using `torch.where(target_mask, path_sample.x_t, x_1)` so prompt tokens remain noise-free across all timesteps $t \in [0, 1]$.
  * **Masked Objective Function**: Replaced global mean reduction with unreduced token-level Cross-Entropy filtered by `target_mask`, ensuring gradients only backpropagate from target neutral tokens.
* **`inference.py` *(Custom Script)***: Implements `ConditionalWrappedModel` around the `MixtureDiscreteEulerSolver` to freeze prompt token distributions during 128-step reverse-flow sampling.
* **`train.py`, `logic/state.py`, & `configs/config.yaml` *(Modified Infrastructure)***: Decoupled multi-node SLURM/NCCL Distributed Data Parallel (DDP) wrappers and CUDA-only fused optimizers to support native unified-memory execution on Apple Silicon (`mps`) alongside single-GPU environments.



##  Initial Prototype Results

| Component / Experiment | Metric / Configuration | Result |
| :--- | :--- | :--- |
| **DistilRoBERTa Bias Judge** | Standalone / Pairwise Accuracy | 72.0% / **84.0%** |
| **Integrated Gradients (Captum)** | Top-1 / Top-3 Gold Word Attribution | **51.6%** / >60.0% |
| **Heuristic Deletion Baseline** | Bias Probability Drop | 0.68 $\rightarrow$ 0.35 (BLEU > 90) |
| **Conditional DFM Prototype** | 6L, 384D (~50M params), 5,000 steps | Loss converged: **10.34 --> 4.71** |
