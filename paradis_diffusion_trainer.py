import logging
from contextlib import suppress

import hydra
from accelerate import Accelerator
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import data.era5_paradis_diffusion_dataset as data
import model.optimizers as optimizers
import model.paradis_diffusion.model as paradis_model
import model.paradis_diffusion.train as training
import model.schedulers as schedulers
import model.utility as model_utility
import model.objectives.diffusion_loss as loss
import utils.utility as utility
import utils.wandb_helper as wbhelp


def _loader_kwargs(cfg: DictConfig, shuffle: bool) -> dict:
    workers = int(cfg.distributed_training.workers)
    kwargs = {
        "batch_size": int(cfg.distributed_training.total_batch_size),
        "shuffle": shuffle,
        "num_workers": workers,
        "drop_last": True,
        "pin_memory": True,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


@hydra.main(version_base=None, config_path="_config", config_name="paradis_diffusion")
def main(cfg: DictConfig) -> None:
    save_path = cfg.experiment.save_path

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.distributed_training.grad_accumulate,
        mixed_precision=cfg.distributed_training.get("mixed_precision", "fp16"),
    )

    wandb_initialized = False
    completed = False
    try:
        utility.set_random_seeds(int(cfg.training.seed))

        if accelerator.is_main_process:
            utility.validate_and_create_save_path(
                cfg.experiment.save_path,
                cfg.experiment.experiment_name,
            )
            wbhelp.init_wandb(
                project_name=cfg.experiment.project_name,
                run_name=cfg.experiment.experiment_name,
                config_class=cfg,
                save_path=save_path,
            )
            wandb_initialized = True

        train_dataset = data.ERA5ParadisDiffusionDataset(
            root_dir=cfg.dataset.root_dir,
            start_date=cfg.training.dataset.start_date,
            end_date=cfg.training.dataset.end_date,
            timesteps=cfg.dataset.timestep,
            cfg=cfg,
        )
        valid_dataset = data.ERA5ParadisDiffusionDataset(
            root_dir=cfg.dataset.root_dir,
            start_date=cfg.training.validation_dataset.start_date,
            end_date=cfg.training.validation_dataset.end_date,
            timesteps=cfg.dataset.timestep,
            cfg=cfg,
        )

        sample_state, sample_constants, _, _ = train_dataset[0]
        channels, _, _ = sample_state.shape

        diffusion_model = paradis_model.get_paradis_diffusion_model(
            state_channels=channels,
            static_channels=sample_constants.shape[0],
            lat=train_dataset.lat,
            lon=train_dataset.lon,
            cfg=cfg,
        )
        if accelerator.is_main_process:
            wbhelp.save_model_architecture(diffusion_model, cfg.experiment.save_path)

        if cfg.experiment.from_checkpoint is not None:
            model_utility.load_model_weights(diffusion_model, cfg.experiment.from_checkpoint)

        optimizer = optimizers.get_adamw(diffusion_model, cfg.optimization.lr)

        train_dl = DataLoader(train_dataset, **_loader_kwargs(cfg, shuffle=True))
        valid_dl = DataLoader(valid_dataset, **_loader_kwargs(cfg, shuffle=False))

        scheduler = schedulers.get_onecycle_lr(
            optimizer,
            cfg.optimization.max_lr,
            cfg.training_info.epochs,
            len(train_dl),
        )

        train_dl, valid_dl, diffusion_model, optimizer, scheduler = accelerator.prepare(
            train_dl,
            valid_dl,
            diffusion_model,
            optimizer,
            scheduler,
        )

        epoch_start = None
        if cfg.experiment.from_state is not None:
            epoch_start = model_utility.load_training_state(
                accelerator,
                cfg.experiment.from_state,
                diffusion_model,
                optimizer,
                scheduler,
            )
            accelerator.print(f"State loaded; resuming from epoch {epoch_start}.")

        criterion = loss.get_diffusion_loss()

        training.training_loop(
            accelerator=accelerator,
            train=train_dl,
            valid=valid_dl,
            model=diffusion_model,
            epochs=cfg.training_info.epochs,
            criterion=criterion,
            save_path=save_path,
            optimizer=optimizer,
            scheduler=scheduler,
            t_timesteps=cfg.dataset.timestep,
            condition_dropout=cfg.paradis_diffusion.condition_dropout,
            validation_batches=cfg.training_info.validation_batches,
            loading_bar=True,
            epoch_start=0 if epoch_start is None else epoch_start,
            config=cfg,
        )

        completed = True
    finally:
        if completed:
            try:
                if accelerator.is_main_process and wandb_initialized:
                    wbhelp.finish_run()
            finally:
                accelerator.end_training()
        else:
            if accelerator.is_main_process and wandb_initialized:
                with suppress(Exception):
                    wbhelp.finish_run()
            with suppress(Exception):
                accelerator.end_training()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
