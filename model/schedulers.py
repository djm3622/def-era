import torch.optim as optim
from omegaconf import DictConfig


def get_onecycle_lr(
    optimizer: optim.Optimizer,
    max_lr: float,
    epoches: int,
    steps_per_epoch: int,
    pct_start: float = 0.1,
    div_factor: float = 25,
    final_div_factor: float = 1e4,
    cfg: DictConfig | None = None,
) -> optim.lr_scheduler:
    scheduler_cfg = {} if cfg is None else cfg
    pct_start = float(scheduler_cfg.get("pct_start", pct_start))
    div_factor = float(scheduler_cfg.get("div_factor", div_factor))
    final_div_factor = float(
        scheduler_cfg.get("final_div_factor", final_div_factor)
    )

    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epoches,
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
    )
