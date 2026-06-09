"""Training-loop runner with `--max_training_steps` support.

Ported from the fork's `diffsynth/diffusion/runner.py`. Token-AE specific
optimizer plumbing is removed; the `max_training_steps` early-stop is kept
because the release exposes that arg.
"""

import os

import torch
from accelerate import Accelerator
from tqdm import tqdm

from diffsynth.diffusion import ModelLogger
from diffsynth.diffusion.training_module import DiffusionTrainingModule


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args=None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs

    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers,
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler,
    )

    max_training_steps = getattr(args, "max_training_steps", None) if args is not None else None
    global_step = 0

    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            if max_training_steps is not None and global_step >= max_training_steps:
                break
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                scheduler.step()
                global_step += 1
        if max_training_steps is not None and global_step >= max_training_steps:
            break
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args=None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers,
    )
    model, dataloader = accelerator.prepare(model, dataloader)

    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(folder, f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
