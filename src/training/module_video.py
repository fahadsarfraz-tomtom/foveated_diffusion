"""Training module for the foveated Wan video pipeline (LoRA or full fine-tune).

Mirrors ``Flux2FoveatedImageTrainingModule`` from the image side but wraps
``WanFoveatedVideoPipeline`` instead. Loss tasks supported here:

  - ``sft`` / ``sft:train``: foveated flow-matching SFT
  - ``sft:data_process``: passthrough for data-cache jobs

When ``foveation_from_video=True``, the pipeline's ``WanFoveatedInputVideoEmbedder``
runs DeepGaze on each input video and populates ``state`` for the
loss. Otherwise the loss samples a ``random_path`` per step.
"""

import torch

from diffsynth.diffusion import DiffusionTrainingModule

from ..diffsynth_fov.loss_video import FoveatedWanFlowMatchSFTLoss
from ..diffsynth_fov.pipeline_video import (
    WanFoveatedVideoPipeline,
    default_wan_model_configs,
    default_wan_tokenizer_config,
)


class WanFoveatedVideoTrainingModule(DiffusionTrainingModule):
    """Wraps a foveated Wan pipeline with LoRA / full-finetune trainable modules."""

    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank: int = 32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task: str = "sft",
        max_timestep_boundary: float = 1.0,
        min_timestep_boundary: float = 0.0,
        foveation_from_video: bool = False,
    ):
        super().__init__()
        model_configs = self.parse_model_configs(
            model_paths, model_id_with_origin_paths,
            fp8_models=fp8_models, offload_models=offload_models, device=device,
        )
        if not model_configs:
            model_configs = default_wan_model_configs()
        tokenizer_config = (
            self.parse_path_or_model_id(tokenizer_path, default_value=default_wan_tokenizer_config())
            if hasattr(self, "parse_path_or_model_id")
            else default_wan_tokenizer_config()
        )

        self.pipe = WanFoveatedVideoPipeline.from_pretrained_foveated(
            torch_dtype=torch.bfloat16, device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            foveation_from_video=foveation_from_video,
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Make the foveation flag explicit on the pipeline so the loss /
        # validation paths can dispatch off it.
        self.pipe.foveation = True
        self.pipe.foveation_from_video = foveation_from_video

        parts = []
        if trainable_models:
            parts.extend(trainable_models.split(","))
        if lora_base_model is not None:
            parts.append(lora_base_model)
        effective_trainable = ",".join(parts) if parts else None
        self.switch_pipe_to_training_mode(
            self.pipe, effective_trainable,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, sh, po, _ne: FoveatedWanFlowMatchSFTLoss(pipe, **sh, **po),
            "sft:train": lambda pipe, sh, po, _ne: FoveatedWanFlowMatchSFTLoss(pipe, **sh, **po),
        }

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        # Standard Wan T2V keys + the foveation-specific knobs.
        first = data["video"][0]
        inputs_shared = {
            "input_video": data["video"],
            "height": first.size[1], "width": first.size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1, "cfg_merge": False,
            "sigma_shift": 5.0,
            "tiled": False,
            "tile_size": (30, 52), "tile_stride": (15, 26),
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            # T2V: every non-text-non-video modality is None.
            "input_image": None, "end_image": None,
            "denoising_strength": 1.0,
            "control_video": None, "reference_image": None,
            "camera_control_direction": None, "camera_control_speed": None, "camera_control_origin": None,
            "vace_video": None, "vace_video_mask": None, "vace_reference_image": None, "vace_scale": 1,
            "motion_bucket_id": None, "longcat_video": None,
            "sliding_window_size": None, "sliding_window_stride": None,
            "input_audio": None, "audio_sample_rate": None, "s2v_pose_video": None,
            "audio_embeds": None, "s2v_pose_latents": None, "motion_video": None,
            "animate_pose_video": None, "animate_face_video": None,
            "animate_inpaint_video": None, "animate_mask_video": None,
            "vap_video": None,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return self.task_to_loss[self.task](self.pipe, *inputs)
