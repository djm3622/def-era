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
from utils.wandb_helper import log_losses


def _clean_states_from_batch(batch) -> torch.Tensor:
    """Extract clean states from current or legacy diffusion batch layouts."""

    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (tuple, list)) and batch:
        return batch[0]
    raise TypeError(f"Unsupported diffusion batch type: {type(batch)!r}")


def _feature_names(config: DictConfig) -> list[str]:
    """Return channel names in the same order as the stacked dataset."""

    names = []
    for variable in config.features.base.atmospheric:
        for level in config.features.pressure_levels:
            names.append(f"{variable}_h{int(level)}")
    names.extend(str(variable) for variable in config.features.base.surface)
    return names


def _standardize_spatial(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
    return (x - mean) / std


@torch.no_grad()
def _sample_with_cfg_ddim(
    model: Module,
    condition: torch.Tensor,
    constants: Optional[torch.Tensor],
    alpha_bar: torch.Tensor,
    t_timesteps: int,
    guidance_scale: float,
    num_steps: int,
    eta: float,
    corrector: bool,
    generator: Optional[torch.Generator],
    loading_bar: bool,
    accelerator: Accelerator,
) -> torch.Tensor:
    """Generate denoised examples from a fixed validation condition batch."""

    num_steps = max(1, min(int(num_steps), int(t_timesteps)))
    samples = torch.randn(
        condition.shape,
        device=condition.device,
        dtype=condition.dtype,
        generator=generator,
    )
    step_indices = torch.linspace(
        t_timesteps - 1,
        0,
        num_steps,
        dtype=torch.long,
        device=condition.device,
    )

    sample_bar = tqdm(
        enumerate(step_indices),
        total=len(step_indices),
        desc="Saving PARADIS diffusion samples",
        leave=False,
        disable=not (loading_bar and accelerator.is_main_process),
        mininterval=1.0,
    )

    for step_offset, step in sample_bar:
        batch_size = condition.shape[0]
        timesteps = torch.full(
            (batch_size,),
            int(step.item()),
            dtype=torch.long,
            device=condition.device,
        )
        double_timesteps = torch.cat([timesteps, timesteps], dim=0)
        double_condition = torch.cat([condition, torch.zeros_like(condition)], dim=0)
        double_constants = (
            torch.cat([constants, constants], dim=0)
            if constants is not None
            else None
        )
        double_samples = torch.cat([samples, samples], dim=0)

        noise_pred = model(
            condition=double_condition,
            noisy_state=double_samples,
            timesteps=double_timesteps,
            constants=double_constants,
            return_dict=False,
        )[0]
        noise_pred_cond, noise_pred_uncond = torch.chunk(noise_pred, 2, dim=0)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        alpha_bar_t = alpha_bar[step].view(1, 1, 1, 1)
        if step_offset + 1 < len(step_indices):
            prev_step = step_indices[step_offset + 1]
            alpha_bar_prev = alpha_bar[prev_step].view(1, 1, 1, 1)
        else:
            alpha_bar_prev = torch.ones_like(alpha_bar_t)

        predicted_x0 = (
            samples - torch.sqrt(1.0 - alpha_bar_t) * noise_pred
        ) / torch.sqrt(alpha_bar_t)
        variance = (
            (1.0 - alpha_bar_prev)
            / (1.0 - alpha_bar_t)
            * (1.0 - alpha_bar_t / alpha_bar_prev)
        ).clamp_min(0.0)
        sigma_t = float(eta) * torch.sqrt(variance)

        if eta > 0.0 and step_offset + 1 < len(step_indices):
            noise = torch.randn(
                samples.shape,
                device=samples.device,
                dtype=samples.dtype,
                generator=generator,
            )
        else:
            noise = torch.zeros_like(samples)

        direction_scale = (1.0 - alpha_bar_prev - sigma_t.square()).clamp_min(0.0)
        samples = (
            torch.sqrt(alpha_bar_prev) * predicted_x0
            + torch.sqrt(direction_scale) * noise_pred
            + sigma_t * noise
        )

        if corrector:
            samples = _standardize_spatial(samples)

    return samples


@torch.no_grad()
def _save_validation_samples(
    valid: DataLoader,
    model: Module,
    alpha_bar: torch.Tensor,
    t_timesteps: int,
    save_path: str,
    epoch: int,
    epochs: int,
    config: DictConfig,
    accelerator: Accelerator,
    loading_bar: bool,
) -> None:
    sample_cfg = config.get("sampling", {})
    if not sample_cfg or not bool(sample_cfg.get("enabled", False)):
        return

    epoch_number = epoch + 1
    interval = max(1, int(sample_cfg.get("interval", 1)))
    should_sample = (
        epoch_number == 1
        or epoch_number == epochs
        or epoch_number % interval == 0
    )
    if not should_sample:
        return

    sample_dir = os.path.join(save_path, str(sample_cfg.get("save_dir", "samples")))
    if accelerator.is_main_process:
        os.makedirs(sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    model.eval()
    clean_states = _clean_states_from_batch(next(iter(valid)))
    num_samples = min(int(sample_cfg.get("num_samples", 4)), clean_states.shape[0])
    clean_states = clean_states[:num_samples]

    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(int(sample_cfg.get("seed", 1000)))

    samples = _sample_with_cfg_ddim(
        model=model,
        condition=clean_states,
        constants=None,
        alpha_bar=alpha_bar,
        t_timesteps=t_timesteps,
        guidance_scale=float(sample_cfg.get("guidance_scale", 1.0)),
        num_steps=int(sample_cfg.get("num_steps", 50)),
        eta=float(sample_cfg.get("eta", 0.0)),
        corrector=bool(sample_cfg.get("corrector", True)),
        generator=generator,
        loading_bar=loading_bar,
        accelerator=accelerator,
    )
    model.train()

    if accelerator.is_main_process:
        payload = {
            "epoch": epoch,
            "epoch_number": epoch_number,
            "samples": samples.detach().float().cpu(),
            "condition": clean_states.detach().float().cpu(),
            "target": clean_states.detach().float().cpu(),
            "feature_names": _feature_names(config),
            "sampling": {
                "sampler": "DDIM",
                "num_steps": int(sample_cfg.get("num_steps", 50)),
                "guidance_scale": float(sample_cfg.get("guidance_scale", 1.0)),
                "eta": float(sample_cfg.get("eta", 0.0)),
                "corrector": bool(sample_cfg.get("corrector", True)),
                "seed": int(sample_cfg.get("seed", 1000)),
            },
        }
        output_file = os.path.join(sample_dir, f"epoch_{epoch_number:06d}.pt")
        accelerator.save(payload, output_file)

    accelerator.wait_for_everyone()


def _diffusion_step(
    batch,
    model: Module,
    criterion: Module,
    alpha_bar: torch.Tensor,
    condition_dropout: float,
) -> torch.Tensor:
    clean_states = _clean_states_from_batch(batch)
    noise = torch.randn_like(clean_states)
    timesteps = torch.randint(
        0,
        alpha_bar.numel(),
        (clean_states.shape[0],),
        device=clean_states.device,
    )
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
        constants=None,
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
        loss = _diffusion_step(batch, model, criterion, alpha_bar, 0.0)
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
        data_wait_time = 0.0
        step_time = 0.0
        batch_count = 0
        start_time = time.time()

        train_bar = tqdm(
            train,
            desc=f"PARADIS diffusion epoch {epoch + 1}",
            leave=False,
            disable=not (loading_bar and accelerator.is_main_process),
            mininterval=1.0,
        )

        train_iter = iter(train_bar)
        while True:
            fetch_start = time.time()
            try:
                train_batch = next(train_iter)
            except StopIteration:
                break
            data_wait_time += time.time() - fetch_start

            step_start = time.time()
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
                optimizer.zero_grad(set_to_none=True)
                train_loss += loss.item()
            step_time += time.time() - step_start
            batch_count += 1

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
        timing = torch.tensor(
            [data_wait_time, step_time, float(batch_count)],
            device=accelerator.device,
        )
        gathered_timing = accelerator.gather(timing).view(-1, 3).mean(dim=0)
        mean_batches = max(float(gathered_timing[2].item()), 1.0)
        mean_data_wait = float(gathered_timing[0].item())
        mean_step_time = float(gathered_timing[1].item())
        accelerator.print(
            f"Epoch {epoch + 1}/{epochs}, train_loss={gathered_train_loss:.6f}, "
            f"valid_loss={valid_loss if valid_loss is not None else 'n/a'}, "
            f"elapsed={elapsed:.2f}s, data_wait={mean_data_wait:.2f}s "
            f"({mean_data_wait / mean_batches:.4f}s/batch), "
            f"step_time={mean_step_time:.2f}s "
            f"({mean_step_time / mean_batches:.4f}s/batch), "
            f"batches={int(mean_batches)}"
        )

        if accelerator.is_main_process:
            log_losses(
                train_loss=gathered_train_loss,
                valid_loss=valid_loss,
                step=epoch,
            )

        accelerator.wait_for_everyone()
        _save_validation_samples(
            valid=valid,
            model=model,
            alpha_bar=alpha_bar,
            t_timesteps=t_timesteps,
            save_path=save_path,
            epoch=epoch,
            epochs=epochs,
            config=config,
            accelerator=accelerator,
            loading_bar=loading_bar,
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
