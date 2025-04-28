import torch
import torch
from diffusers import UNet2DConditionModel, LCMScheduler, StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, DDIMScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import sys
sys.path.append('.')
from .utils import call_sdxl

StableDiffusionXLPipeline.__call__ = call_sdxl

def load_sdxl_models(distillation_type='dmd', weights_dtype=torch.bfloat16, device='cuda:0'):
    
    basemodel_id = "stabilityai/stable-diffusion-xl-base-1.0"

    base_unet = UNet2DConditionModel.from_pretrained(basemodel_id, subfolder="unet").to(device, weights_dtype)
    
    pipe = StableDiffusionXLPipeline.from_pretrained(basemodel_id, unet=base_unet, torch_dtype=weights_dtype, use_safetensors=True)
    
    
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    base_scheduler = pipe.scheduler
    
    if distillation_type == 'dmd':
        
        repo_name = "tianweiy/DMD2"
        ckpt_name = "dmd2_sdxl_4step_unet_fp16.bin"

        distilled_unet = UNet2DConditionModel.from_config(basemodel_id, subfolder="unet").to(device, weights_dtype)
        distilled_unet.load_state_dict(torch.load(hf_hub_download(repo_name, ckpt_name), weights_only=True))
        distilled_scheduler =  LCMScheduler.from_config(pipe.scheduler.config)


    elif distillation_type == 'lightning':
        
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!
        
        # Load model.
        distilled_unet = UNet2DConditionModel.from_config(basemodel_id, subfolder="unet").to(device, weights_dtype)
        distilled_unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
        # Ensure sampler uses "trailing" timesteps.
        distilled_scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        base_scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    elif distillation_type == 'turbo':
        
        distilled_unet = UNet2DConditionModel.from_pretrained("stabilityai/sdxl-turbo", subfolder="unet", torch_dtype=weights_dtype).to(device, weights_dtype)
        distilled_scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        base_scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        
    elif distillation_type == 'lcm':
        
        distilled_unet = UNet2DConditionModel.from_pretrained("latent-consistency/lcm-sdxl", torch_dtype=weights_dtype).to(device, weights_dtype)
        distilled_scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    else:
        raise Exception("Distillation Type is not recognised. Available options ('dmd', 'turbo', 'lightning', 'lcm')")
    pipe.to(device).to(weights_dtype)
    return pipe, base_unet, base_scheduler, distilled_unet, distilled_scheduler



def load_pipe(distillation_type='dmd', weights_dtype=torch.bfloat16, device='cuda:0'):
    
    basemodel_id = "stabilityai/stable-diffusion-xl-base-1.0"
    
    if distillation_type is None:
        base_unet = UNet2DConditionModel.from_pretrained(basemodel_id, subfolder="unet").to(device, weights_dtype)
        pipe = StableDiffusionXLPipeline.from_pretrained(basemodel_id, unet=base_unet, torch_dtype=weights_dtype, use_safetensors=True)
        pipe.to(device).to(weights_dtype)
        return pipe


    
    if distillation_type == 'dmd':
        
        repo_name = "tianweiy/DMD2"
        ckpt_name = "dmd2_sdxl_4step_unet_fp16.bin"

        distilled_unet = UNet2DConditionModel.from_config(basemodel_id, subfolder="unet").to(device, weights_dtype)
        distilled_unet.load_state_dict(torch.load(hf_hub_download(repo_name, ckpt_name), weights_only=True))
        pipe = StableDiffusionXLPipeline.from_pretrained(basemodel_id, unet=distilled_unet, torch_dtype=weights_dtype, use_safetensors=True)
        distilled_scheduler =  LCMScheduler.from_config(pipe.scheduler.config)


    elif distillation_type == 'lightning':
        
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!
        
        # Load model.
        distilled_unet = UNet2DConditionModel.from_config(basemodel_id, subfolder="unet").to(device, weights_dtype)
        distilled_unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
        pipe = StableDiffusionXLPipeline.from_pretrained(basemodel_id, unet=distilled_unet, torch_dtype=weights_dtype, use_safetensors=True)
        # Ensure sampler uses "trailing" timesteps.
        distilled_scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        base_scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    elif distillation_type == 'turbo':
        
        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=weights_dtype, use_safetensors=True)
        pipe.to(device).to(weights_dtype) 
        return pipe
        
    elif distillation_type == 'lcm':
        
        distilled_unet = UNet2DConditionModel.from_pretrained("latent-consistency/lcm-sdxl", torch_dtype=weights_dtype).to(device, weights_dtype)
        pipe = StableDiffusionXLPipeline.from_pretrained(basemodel_id, unet=distilled_unet, torch_dtype=weights_dtype, use_safetensors=True)
        distilled_scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    else:
        raise Exception("Distillation Type is not recognised. Available options ('dmd', 'turbo', 'lightning', 'lcm')")

    
    pipe.scheduler = distilled_scheduler
    pipe.to(device).to(weights_dtype) 
    return pipe