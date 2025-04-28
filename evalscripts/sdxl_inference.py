import torch
import os
import time
import pandas as pd
from tqdm.auto import tqdm
import argparse

import sys
sys.path.append('.')
from utils.load_util import load_pipe

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
                    num_images_per_prompt=1):

    # load the base unet and distilled unet and sdxl pipeline
    pipe = load_pipe(distillation_type=distillation_type, 
                            weights_dtype=weights_dtype, 
                            device=device)

    df = pd.read_csv(prompts_path)
    prompts = df.prompt
    seeds = df.evaluation_seed
    case_numbers = df.case_number
    
    folder_path = f'{save_path}/'
    os.makedirs(folder_path, exist_ok=True)
    
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
        pil_images = pipe(prompt=prompt, guidance_scale=distilled_guidance_scale, 
                          num_inference_steps=distilled_num_inference_steps)
        
        for num, im in enumerate(pil_images):
            im.save(f"{folder_path}/{case_number}_{num}.png")



if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImprovedDistillationImages',
                    description = 'Generate Images using Improved Distillation Method for Diffusion Models')

    parser.add_argument('--distillation_type', help='type of distilled model (e.g., dmd, turbo, lcm, lightning)', 
                       type=str, required=False, default=None)
    
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, 
                       required=False, default="coco_30k.csv")
    
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=False, default='eval-images/')
    
    parser.add_argument('--device', help='cuda device to run on', type=str, 
                       required=False, default='cuda:0')
    
    parser.add_argument('--distilled_guidance_scale', help='guidance scale for distilled model', 
                       type=float, required=False, default=0)
    
    parser.add_argument('--base_guidance_scale', help='guidance scale for base model', 
                       type=float, required=False, default=7)
    
    parser.add_argument('--till_case', help='continue generating till case_number', 
                       type=int, required=False, default=1000000)
    
    parser.add_argument('--from_case', help='continue generating from case_number', 
                       type=int, required=False, default=0)
    
    parser.add_argument('--base_num_inference_steps', help='number of inference steps for base model', 
                       type=int, required=False, default=8)
    
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
        num_images_per_prompt=args.num_images_per_prompt
    )
