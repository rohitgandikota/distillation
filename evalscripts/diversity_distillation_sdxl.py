import torch
import os
import time
import pandas as pd
from tqdm.auto import tqdm
import argparse

import sys
sys.path.append('.')
from utils.load_util import load_sdxl_models

def generate_improved_images(prompts_path, save_path, 
                    
                    distillation_type = "dmd", 
                    
                    run_base_till_timestep = None,
                    run_distilled_from_timestep = 1,
                    
                    distilled_num_inference_steps = 4,
                    base_num_inference_steps = 20,
                    
                    base_guidance_scale = 7,
                    distilled_guidance_scale = 0,
                    
                    from_case=0, 
                    till_case=1000000, 
                    
                    weights_dtype=torch.bfloat16, 
                    device='cuda:0',
                    num_images_per_prompt=1,
                             
                    same_scheduler= False):

    # load the base unet and distilled unet and sdxl pipeline
    pipe, base_unet, base_scheduler, distilled_unet, distilled_scheduler = load_sdxl_models(distillation_type=distillation_type, 
                                                                                        weights_dtype=weights_dtype, 
                                                                                        device=device)


    if same_scheduler:
        base_scheduler = distilled_scheduler
    # set the timesteps for the model
    base_scheduler.set_timesteps(base_num_inference_steps)
    distilled_scheduler.set_timesteps(distilled_num_inference_steps)
    
    # automatically figure out what is the natural point to turn off the base model
    if run_base_till_timestep is None:
        # check the timestep from which you need to run the model
        distilled_timestep = distilled_scheduler.timesteps[run_distilled_from_timestep]
    
        # check the closest timestep in basemodel
        base_timesteps = abs(base_scheduler.timesteps - distilled_timestep)
        run_base_till_timestep = base_timesteps.argmin()

    
    
    df = pd.read_csv(prompts_path)
    prompts = df.prompt
    seeds = df.evaluation_seed
    case_numbers = df.case_number
    
    folder_path = f'{save_path}/'
    os.makedirs(folder_path, exist_ok=True)
    
    total_time = 0

    pipe.set_progress_bar_config(disable=True)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt = [str(row.prompt)]*num_images_per_prompt
        seed = row.evaluation_seed
        case_number = row.case_number
    
        if not (case_number>=from_case and case_number<=till_case):
            continue
    
        if os.path.exists(f"{folder_path}/{case_number}_0.png"):
            continue
    
        generator = torch.manual_seed(seed)    # Seed generator to create the inital latent noise
    
        # first use base model
        pipe.unet = base_unet
        pipe.scheduler = base_scheduler
        
        base_latents = pipe(prompt=prompt, from_timestep=0, till_timestep=run_base_till_timestep, guidance_scale = base_guidance_scale,
                            num_inference_steps=base_num_inference_steps, output_type='latent')
    
        # switch to distilled model
        pipe.unet = distilled_unet
        pipe.scheduler = distilled_scheduler
        start_time = time.perf_counter()
        pil_images = pipe(prompt=prompt, start_latents=base_latents, guidance_scale=distilled_guidance_scale, 
                      from_timestep=run_distilled_from_timestep, till_timestep=None, num_inference_steps=distilled_num_inference_steps)
        end_time = time.perf_counter()
        runtime = end_time - start_time
        total_time += runtime
        for num, im in enumerate(pil_images):
            im.save(f"{folder_path}/{case_number}_{num}.png")
    print(f"Total Runtime: {total_time:.4f} seconds")



if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImprovedDistillationImages',
                    description = 'Generate Images using Improved Distillation Method for Diffusion Models')

    parser.add_argument('--distillation_type', help='type of distilled model (e.g., dmd, turbo, lcm, lightning)', 
                       type=str, required=False, default='dmd')
    
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, 
                       required=False, default="data/coco_30k.csv")
    
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=False, default='eval-images/')
    
    parser.add_argument('--device', help='cuda device to run on', type=str, 
                       required=False, default='cuda:0')
    
    parser.add_argument('--distilled_guidance_scale', help='guidance scale for distilled model', 
                       type=float, required=False, default=0)
    
    parser.add_argument('--base_guidance_scale', help='guidance scale for base model', 
                       type=float, required=False, default=5)
    
    parser.add_argument('--till_case', help='continue generating till case_number', 
                       type=int, required=False, default=9999)
    
    parser.add_argument('--from_case', help='continue generating from case_number', 
                       type=int, required=False, default=0)
    
    parser.add_argument('--base_num_inference_steps', help='number of inference steps for base model', 
                       type=int, required=False, default=4)
    
    parser.add_argument('--distilled_num_inference_steps', help='number of inference steps for distilled model', 
                       type=int, required=False, default=4)
    
    parser.add_argument('--run_base_till_timestep', help='timestep to stop base model', 
                       type=int, required=False, default=None)
    
    parser.add_argument('--run_distilled_from_timestep', help='timestep to start distilled model', 
                       type=int, required=False, default=1)
    
    parser.add_argument('--dtype', help='data type for weights (fp16, fp32, bf16)', 
                       type=str, required=False, default='bf16')
    
    parser.add_argument('--num_images_per_prompt', help='number of images to generate per prompt', 
                       type=int, required=False, default=1)
    
    parser.add_argument('--exp_name', help='if you want a customname for savefolder', type=str, 
                   required=False, default=None)

    parser.add_argument('--same_scheduler', help='if you want to have both base and distilled same schedulers', type=str, 
                   required=False, default='True')
    
    args = parser.parse_args()
    
    if args.dtype == 'bf16':
        weights_dtype = torch.bfloat16
    elif args.dtype == 'fp16':
        weights_dtype = torch.float16
    elif args.dtype == 'fp32':
        weights_dtype = torch.float32
    else:
        raise Exception(f'Dtype {args.dtype} is not implemented. select between "fp16", "fp32", bf16"')

    # Create descriptive folder name
    descriptive_name = (
        f"{args.save_path}/{args.distillation_type}_"
        f"bg{args.base_guidance_scale}_dg{args.distilled_guidance_scale}_"
        f"bn{args.base_num_inference_steps}_dn{args.distilled_num_inference_steps}_"
        f"bts{args.run_base_till_timestep}_dts{args.run_distilled_from_timestep}"
    )

    if args.exp_name is not None:
        descriptive_name =  f"{args.save_path}/{args.exp_name}/"

    generate_improved_images(
        distillation_type=args.distillation_type,
        prompts_path=args.prompts_path,
        save_path=descriptive_name,
        run_base_till_timestep=args.run_base_till_timestep,
        run_distilled_from_timestep=args.run_distilled_from_timestep,
        distilled_num_inference_steps=args.distilled_num_inference_steps,
        base_num_inference_steps=args.base_num_inference_steps,
        base_guidance_scale=args.base_guidance_scale,
        distilled_guidance_scale=args.distilled_guidance_scale,
        from_case=args.from_case,
        till_case=args.till_case,
        weights_dtype=weights_dtype,
        device=args.device,
        num_images_per_prompt=args.num_images_per_prompt,
        same_scheduler=eval(args.same_scheduler),
    )
