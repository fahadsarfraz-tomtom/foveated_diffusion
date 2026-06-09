"""Training module for the foveated FLUX2 DiT (LoRA or full fine-tune).

Loss tasks supported here:
  - "sft" / "sft:train": foveated flow-matching SFT
  - "direct_distill" / "direct_distill:train": direct flow-matching distillation
  - "sft:data_process" / "direct_distill:data_process": passthrough for data-cache jobs
"""

import torch

from diffsynth.diffusion import DiffusionTrainingModule, DirectDistillLoss
from diffsynth.pipelines.flux2_image import ModelConfig

from ..diffsynth_fov import Flux2FoveatedImagePipeline, FoveatedFlowMatchSFTLoss


class Flux2FoveatedImageTrainingModule(DiffusionTrainingModule):
    """Wraps a foveated FLUX2 pipeline with LoRA / full-finetune trainable modules."""

    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        prediction_type="clean",
        is_foveated_pipeline=True,
        foveated_training_mode="random",
        lr_downsample_factor=2,
    ):
        super().__init__()
        model_configs = self.parse_model_configs(
            model_paths, model_id_with_origin_paths,
            fp8_models=fp8_models, offload_models=offload_models, device=device,
        )
        tokenizer_config = self.parse_path_or_model_id(
            tokenizer_path,
            default_value=ModelConfig(
                model_id="black-forest-labs/FLUX.2-klein-base-4B",
                origin_file_pattern="tokenizer/",
            ),
        )
        self.pipe = Flux2FoveatedImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        self.pipe.is_foveated_pipeline = is_foveated_pipeline
        assert foveated_training_mode in ("fixed", "random", "saliency", "bbox"), \
            "foveated_training_mode must be one of: fixed, random, saliency, bbox"
        self.pipe.foveated_training_mode = foveated_training_mode
        self.foveated_training_mode = foveated_training_mode
        self.lr_downsample_factor = lr_downsample_factor
        print(f"[TrainingModule] foveated={is_foveated_pipeline}  "
              f"mode={foveated_training_mode}  lr_factor={lr_downsample_factor}")

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
        self.prediction_type = prediction_type

        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, sh, po, _ne: FoveatedFlowMatchSFTLoss(pipe, **sh, **po),
            "sft:train": lambda pipe, sh, po, _ne: FoveatedFlowMatchSFTLoss(pipe, **sh, **po),
            "direct_distill": lambda pipe, sh, po, _ne: DirectDistillLoss(pipe, **sh, **po),
            "direct_distill:train": lambda pipe, sh, po, _ne: DirectDistillLoss(pipe, **sh, **po),
        }

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            "input_image": data["image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            "embedded_guidance": 1.0,
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "prediction_type": self.prediction_type,
            "foveated_training_mode": self.foveated_training_mode,
            "lr_downsample_factor": self.lr_downsample_factor,
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
