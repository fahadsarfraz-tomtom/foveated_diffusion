"""Foveated Diffusion — training entry point.

Dispatches on ``--pipeline``:
  - ``image``: FLUX2 foveated training (foveated flow-matching SFT, no token AE).
  - ``video``: Wan2.1 T2V foveated training (random_path default, saliency
    via ``--foveation_from_video``).
"""

import os
import sys

# Disable sageattention: in some envs the installed binary is built against a
# different libtorch ABI and crashes at import. We fall back to PyTorch attention.
sys.modules["sageattention"] = None

import accelerate

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, ImageCropAndResize
from diffsynth.diffusion import ModelLogger

from src.diffsynth_fov import (
    WandbModelLogger,
    WanVideoWandbModelLogger,
    launch_data_process_task,
    launch_training_task,
)
from src.training import (
    Flux2FoveatedImageTrainingModule,
    WanFoveatedVideoTrainingModule,
    build_parser,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_LAUNCHER_MAP = {
    "sft:data_process": launch_data_process_task,
    "direct_distill:data_process": launch_data_process_task,
    "sft": launch_training_task,
    "sft:train": launch_training_task,
    "direct_distill": launch_training_task,
    "direct_distill:train": launch_training_task,
}


def _make_image_dataset(args):
    return UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        ),
    )


def _make_video_dataset(args):
    return UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
    )


def _make_image_logger(args):
    if not args.use_wandb:
        return ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)
    return WandbModelLogger(
        output_path=args.output_path,
        project_name=args.wandb_project,
        run_name=args.wandb_run_name,
        config=vars(args),
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        validation_prompts=args.validation_prompts or [],
        validation_steps=args.validation_steps,
        num_validation_images=args.num_validation_images,
        validation_kwargs={
            "height": args.validation_height,
            "width": args.validation_width,
            "rand_device": "cuda",
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed,
            "decode_mode": args.decode_mode,
            "prediction_type": args.prediction_type,
            "lr_downsample_factor": args.lr_downsample_factor,
        },
    )


def _make_video_logger(args):
    # WanVideoWandbModelLogger always prints per-step loss to stdout. WandB
    # upload is gated on --use_wandb. Sample-video validation runs every
    # `--log_video_steps` steps and lands on disk under {output_path}/samples/
    # (plus wandb.Video upload when WandB is on).
    return WanVideoWandbModelLogger(
        output_path=args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_config=vars(args),
        log_video_steps=args.log_video_steps,
        sample_prompt=args.sample_prompt,
        sample_foveation_trajectory=args.sample_foveation_trajectory,
        sample_num_inference_steps=args.sample_num_inference_steps,
        sample_seed=args.sample_seed,
        sample_cfg_scale=args.sample_cfg_scale,
        sample_height=args.height,
        sample_width=args.width,
        sample_num_frames=args.num_frames,
    )


def main():
    args = build_parser().parse_args()

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(
            find_unused_parameters=args.find_unused_parameters,
        )],
    )

    if args.pipeline == "image":
        dataset = _make_image_dataset(args)
        model = Flux2FoveatedImageTrainingModule(
            model_paths=args.model_paths,
            model_id_with_origin_paths=args.model_id_with_origin_paths,
            tokenizer_path=args.tokenizer_path,
            trainable_models=args.trainable_models,
            lora_base_model=args.lora_base_model,
            lora_target_modules=args.lora_target_modules,
            lora_rank=args.lora_rank,
            lora_checkpoint=args.lora_checkpoint,
            preset_lora_path=args.preset_lora_path,
            preset_lora_model=args.preset_lora_model,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
            extra_inputs=args.extra_inputs,
            fp8_models=args.fp8_models,
            offload_models=args.offload_models,
            task=args.task,
            device=accelerator.device,
            prediction_type=args.prediction_type,
            is_foveated_pipeline=args.is_foveated_pipeline,
            foveated_training_mode=args.foveated_training_mode,
            lr_downsample_factor=args.lr_downsample_factor,
        )
        logger = _make_image_logger(args)
    else:  # video
        dataset = _make_video_dataset(args)
        model = WanFoveatedVideoTrainingModule(
            model_paths=args.model_paths,
            model_id_with_origin_paths=args.model_id_with_origin_paths,
            tokenizer_path=args.tokenizer_path,
            trainable_models=args.trainable_models,
            lora_base_model=args.lora_base_model,
            lora_target_modules=args.lora_target_modules,
            lora_rank=args.lora_rank,
            lora_checkpoint=args.lora_checkpoint,
            preset_lora_path=args.preset_lora_path,
            preset_lora_model=args.preset_lora_model,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
            extra_inputs=args.extra_inputs,
            fp8_models=args.fp8_models,
            offload_models=args.offload_models,
            task=args.task,
            device=accelerator.device,
            max_timestep_boundary=args.max_timestep_boundary,
            min_timestep_boundary=args.min_timestep_boundary,
            foveation_from_video=args.foveation_from_video,
        )
        logger = _make_video_logger(args)

    _LAUNCHER_MAP[args.task](accelerator, dataset, model, logger, args=args)


if __name__ == "__main__":
    main()
