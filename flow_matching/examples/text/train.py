# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import datetime
import os
import torch

from flow_matching.loss import MixturePathGeneralizedKL
from logic import evaluate, flow, generate, training
from logic.state import TrainState
from model import Transformer
from omegaconf import OmegaConf
from torch import optim
from utils import checkpointing, logging

# NEW: Import our custom dataloader
from data.wnc_data import get_wnc_dataloaders


def run_train(rank: int, cfg: OmegaConf) -> None:
    torch.manual_seed(cfg.training.seed + rank)

    # Logging and configuration
    work_dirs = checkpointing.get_work_dirs(work_dir=cfg.work_dir, rank=rank)
    logger = logging.TrainLogger(log_dir=work_dirs.root, rank=rank, cfg=cfg)
    logger.info(work_dirs)
    logger.info(cfg)

    # NEW: Safely detect device (CUDA vs MPS vs CPU)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    logger.log_devices(device=device, logger=logger)

    # NEW: Load our custom WNC Data
    train_path = os.path.expanduser("~/dfm-neutralization/data/processed/train.jsonl")
    dev_path = os.path.expanduser("~/dfm-neutralization/data/processed/dev.jsonl")
    
    # We pass cfg parameters to match their expected batch sizes and sequence lengths
    train_iter, eval_iter, tokenizer = get_wnc_dataloaders(
        train_path=train_path,
        dev_path=dev_path,
        tokenizer_name="distilroberta-base",
        batch_size=cfg.training.batch_size,
        max_length=cfg.model.length
    )
    # We need an infinite iterator for their train loop structure
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x
    train_iterator = iter(cycle(train_iter))
    eval_iterator = iter(cycle(eval_iter))

    vocab_size = tokenizer.vocab_size

    source_distribution = flow.get_source_distribution(
        source_distribution=cfg.flow.source_distribution, vocab_size=vocab_size
    )

    # Model initialization
    model = Transformer(
        config=cfg.model, vocab_size=vocab_size, masked=source_distribution.masked
    ).to(device)

    num_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters in the model: {num_parameters}")

    # NEW: Removed DDP wrap. Just log the standard model.
    logger.info(model)

    # Optimizer initialization
    # NEW: MPS doesn't support fused=True, so we disable it if we aren't on CUDA
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
        fused=cfg.optim.fused if torch.cuda.is_available() else False,
    )
    logger.info(f"Optimizer: {optimizer}")
    
    # NEW: GradScaler is for CUDA AMP. Disable it gracefully on MPS to prevent crashes.
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    logger.info(f"Scaler: {scaler}")

    # Train state
    state = TrainState(model=model, optimizer=optimizer, step=1, data_state=None)
    state.restore_checkpoint(ckpt_dir=work_dirs.checkpoint, device=device, rank=rank)

    if cfg.model.compile and torch.cuda.is_available():
        # torch.compile has limited support on MPS right now, safer to only enable on CUDA
        state.compile_model()
        torch.set_float32_matmul_precision("high")

    # Flow matching
    path = flow.get_path(
        scheduler_type=cfg.flow.scheduler_type, exponent=cfg.flow.exponent
    )
    loss_fn = flow.get_loss_function(loss_function=cfg.flow.loss_function, path=path)
    time_epsilon = 1e-3 if isinstance(loss_fn, MixturePathGeneralizedKL) else 0.0

    num_train_steps = cfg.optim.n_iters
    logger.info(f"Starting training loop at step {state.step}.")

    train_loss_values = []

    while state.step <= num_train_steps:
        loss = training.step(
            loss_fn=loss_fn,
            path=path,
            state=state,
            scaler=scaler,
            iterator=train_iterator,
            optim_params=cfg.optim,
            device=device,
            source_distribution=source_distribution,
            logger=logger,
            training=True,
            time_epsilon=time_epsilon,
        )

        train_loss_values.append(loss)

        # Train logging
        if state.step % cfg.logging.log_freq == 0:
            agg_train_loss_values = torch.tensor(train_loss_values, device=device).mean()
            # NEW: Removed dist.all_reduce
            logger.log_metric(
                value=agg_train_loss_values, name="Loss", stage="Train", step=state.step
            )
            train_loss_values = []

        # Checkpoint
        if state.step % cfg.training.snapshot == 0:
            logger.info("Saving checkpoint...", step=state.step)
            state.save_checkpoint(ckpt_dir=work_dirs.checkpoint, rank=rank)

        # Evaluation loss
        if state.step % cfg.training.eval_freq == 0:
            logger.info("Evaluating loss...", step=state.step)
            eval_loss = training.step(
                state=state,
                loss_fn=loss_fn,
                path=path,
                scaler=scaler,
                iterator=eval_iterator,
                device=device,
                source_distribution=source_distribution,
                logger=logger,
                training=False,
                time_epsilon=time_epsilon,
            )
            # NEW: Removed dist.all_reduce
            logger.log_metric(
                value=eval_loss.item(), name="Loss", stage="Evaluation", step=state.step
            )

        state.step = state.step + 1

    if state.step == num_train_steps:
        logger.info("Saving checkpoint...", step=state.step)
        state.save_checkpoint(ckpt_dir=work_dirs.checkpoint, rank=rank)

    logger.finish()

# NEW: We completely bypass the multi-process setup and just run the script directly.
def run_mp_training(rank: int, world_size: int, cfg: OmegaConf, port: int) -> None:
    # Single GPU/MPS bypass
    run_train(rank=0, cfg=cfg)

