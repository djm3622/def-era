"""Training utilities for PARADIS-backed diffusion denoisers."""

import os
import time
from typing import Optional

import torch
import torch.optim as optim
from accelerate import Accelerator
from omegaconf import DictConfig
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model.utility import save_training_state


def _diffusion_step(
    batch,
    model: Module,
    criterion: Module,
    alpha_bar: torch.Tensor,
    condition_dropout: float,
) -> torch.Tensor:
    clean_states, forcings, constants, noise, timesteps = batch
    timesteps = timesteps.squeeze(-1).long()
    alpha_bar_t = alpha_bar[timesteps].view(-1, 1, 1, 1)

    null_mask = torch.rand(
        clean_states.shape[0], device=clean_states.device
    ) < condition_dropout
    condition = torch.where(
        null_mask.view(-1, 1, 1, 1),
        torch.zeros_like(clean_states),
        clean_states,
    )

    noisy_state = (
        torch.sqrt(alpha_bar_t) * clean_states
        + torch.sqrt(1 - alpha_bar_t) * noise
    )
    noise_pred = model(
        condition=condition,
        noisy_state=noisy_state,
        timesteps=timesteps,
        forcings=forcings,
        constants=constants,
        return_dict=False,
    )[0]

    return criterion(noise_pred, noise)


@torch.no_grad()
def _validation_loss(
    valid: DataLoader,
    model: Module,
    criterion: Module,
    alpha_bar: torch.Tensor,
    condition_dropout: float,
    accelerator: Accelerator,
    max_batches: Optional[int] = None,
) -> Optional[float]:
    if valid is None or max_batches == 0:
        return None

    model.eval()
    losses = []
    for batch_idx, batch in enumerate(valid):
        if max_batches is not None and batch_idx >= max_batches:
            break
        loss = _diffusion_step(batch, model, criterion, alpha_bar, condition_dropout)
        gathered = accelerator.gather(loss.detach().view(1)).mean()
        losses.append(gathered)

    model.train()
    if not losses:
        return None
    return torch.stack(losses).mean().item()


def training_loop(
    accelerator: Accelerator,
    train: DataLoader,
    valid: DataLoader,
    model: Module,
    epochs: int,
    criterion: Module,
    save_path: str,
    optimizer: optim.Optimizer,
    scheduler: Optional[object],
    t_timesteps: int,
    condition_dropout: float = 0.1,
    validation_batches: Optional[int] = 4,
    loading_bar: bool = True,
    epoch_start: int = 0,
    config: DictConfig = {},
) -> None:
    accelerator.print(f"Rank: {accelerator.process_index}")
    accelerator.print(f"Train dataset size: {len(train.dataset)}")
    accelerator.print(
        f"Number of workers: {train.num_workers if hasattr(train, 'num_workers') else 'N/A'}"
    )

    beta_start = float(config.dataset.get("beta_start", 1e-4))
    beta_end = float(config.dataset.get("beta_end", 0.02))
    beta = torch.linspace(beta_start, beta_end, t_timesteps).to(accelerator.device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0).to(accelerator.device)

    os.makedirs(save_path, exist_ok=True)

    model.train()
    for epoch in range(epoch_start, epochs):
        train_loss = 0.0
        start_time = time.time()

        train_bar = tqdm(
            train,
            desc=f"PARADIS diffusion epoch {epoch + 1}",
            leave=False,
            disable=not (loading_bar and accelerator.is_main_process),
            mininterval=1.0,
        )

        for train_batch in train_bar:
            with accelerator.accumulate(model):
                loss = _diffusion_step(
                    train_batch,
                    model,
                    criterion,
                    alpha_bar,
                    condition_dropout,
                )
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                train_loss += loss.item()

            if loading_bar:
                lr = optimizer.param_groups[0]["lr"]
                train_bar.set_postfix(train_loss=loss.item(), lr=lr)

        train_loss /= max(len(train), 1)
        gathered_train_loss = accelerator.gather(
            torch.tensor([train_loss], device=accelerator.device)
        ).mean().item()
        valid_loss = _validation_loss(
            valid,
            model,
            criterion,
            alpha_bar,
            condition_dropout,
            accelerator,
            max_batches=validation_batches,
        )

        elapsed = time.time() - start_time
        accelerator.print(
            f"Epoch {epoch + 1}/{epochs}, train_loss={gathered_train_loss:.6f}, "
            f"valid_loss={valid_loss if valid_loss is not None else 'n/a'}, "
            f"elapsed={elapsed:.2f}s"
        )

        accelerator.wait_for_everyone()
        save_training_state(
            accelerator=accelerator,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=os.path.join(save_path, "states"),
        )
        accelerator.wait_for_everyone()
