import os

import time
import argparse
from typing import List, Tuple
import math

import torch
import numpy as np
from PIL import Image
from einops import rearrange
from tqdm import tqdm

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

# ---------------------------
# utils
# ---------------------------
def get_noise(
    num_samples: int,
    channels: int,
    height: int,
    width: int,
    device: torch.device,
    seed: int,
):
    """
    returns: (B, C, H, W) 
    """
    return torch.randn(
        num_samples,
        channels,
        height // 8,
        width // 8,
        device=device,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

# ---------------------------
# Core inference
# ---------------------------
class Engine:

    def __init__(
        self, 
        device: str = 'cuda',
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
    ):
        self.device = device
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        self.noise_scheduler = DDIMScheduler.from_pretrained(self.pretrained_model_name_or_path, subfolder="scheduler")

        self.vae = AutoencoderKL.from_pretrained(self.pretrained_model_name_or_path, subfolder="vae").to(
            self.device,
        ).eval()
        self.unet = UNet2DConditionModel.from_pretrained(self.pretrained_model_name_or_path, subfolder="unet").to(
            self.device,
        ).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(self.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.pretrained_model_name_or_path, subfolder="text_encoder").to(
            self.device,
        ).eval()

        # Sample: Load fine-tuned weights
        #print("Loading finetuned weights...")
        #state_dict = torch.load(os.path.join(self.pretrained_model_name_or_path, "your_dir", "model.pt"), map_location=self.device)
        #self.unet.load_state_dict(state_dict)
    
    @torch.inference_mode()
    def encode_text(self, text: list[str]) -> torch.Tensor:
        """Encode text to CLIP features
        Args:
            text (List[str]): List of text prompts to encode.
        Returns:
            torch.Tensor: Encoded text features. Shape is (B, max_length, D_clip).
        """

        # Tokenize text
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = text_inputs.to(self.device) # (B, max_length)
        text_embeddings = self.text_encoder(text_inputs.input_ids)[0] # (B, max_length, D_clip)

        return text_embeddings  
    
    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str],
        height: int,
        width: int,
        seed: int,
        num_steps: int,
        guidance: float,
    ):

        # ----------------------------
        #  Conditioning
        # ----------------------------
        text_embeddings = self.encode_text(prompts)  # (B, max_length, D_clip)
        uncond_embeddings = self.encode_text([""] * len(prompts))  # (B, max_length, D_clip)
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)  # (2*B, max_length, D_clip)

        # ----------------------------
        # initial noise
        # ----------------------------
        x = get_noise(len(prompts), 4, height, width, device=self.device, seed=seed)
        self.noise_scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps

        # ----------------------------
        # Denoise in latent space
        # ----------------------------
        for i, t in enumerate(timesteps):
            # for cfg
            batch_x = torch.cat([x] * 2, dim=0)  # (2*B, C, H, W)
            t_expanded = torch.tensor([t] * batch_x.shape[0], device=self.device)

            # predict noise
            noise_pred = self.unet(batch_x, t_expanded, encoder_hidden_states=text_embeddings).sample  # (2*B, C, H, W)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

            if guidance > 1.0:
                noise_pred = noise_pred_uncond + guidance * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond

            # compute x_t -> x_t-1
            x = self.noise_scheduler.step(noise_pred, t, x).prev_sample
        
        # ----------------------------
        # Decode to pixel space
        # ----------------------------
        x = x / self.vae.config.scaling_factor
        xpix = self.vae.decode(x).sample  # (B, C=3, H, W)

        # Post-process to uint8 frames (B, H, W, 3)
        xpix = xpix.clamp_(-1, 1).add_(1.0).mul_(255.0 / 2.0)
        xpix = xpix.to(torch.uint8)
        xpix = rearrange(xpix, 'b c h w -> b h w c')  # (B, H, W, 3)
        frames = [Image.fromarray(x.cpu().numpy()) for x in xpix] # List of PIL images

        return frames

# ---------------------------
# CLI
# ---------------------------
def main():

    # -------------------------
    # Parse command-line arguments
    # -------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument('--prompt_file', type=str, required=True)
    parser.add_argument('--pretrained_model_name_or_path', type=str,default="./stable-diffusion-v1-5")

    parser.add_argument('--width', type=int, default=512, help='Output width (multiple of 16)')
    parser.add_argument('--height', type=int, default=512, help='Output height (multiple of 16)')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of diffusion steps')
    parser.add_argument('--guidance', type=float, default=7.5, help='Guidance')
    parser.add_argument('--seed', type=int, default=42, help='Global seed (per-line)')

    parser.add_argument('--output_dir', type=str, default=f"./output/", help='Output directory')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.output_dir, exist_ok=True)
    out_tmpl = os.path.join(args.output_dir, 'seq_{idx}.jpg')

    # ---------------
    # Initialize the engine
    # ---------------
    print("Loading engine...")
    engine = Engine(
        device=device,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
    )
    print("Engine loaded.")

    # ---------------
    # Read the prompt file
    # ---------------
    print(f"Reading prompt file: {args.prompt_file}")
    dataset = []
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(line.strip())

    # ---------------
    # Process each line in the dataset
    # ----------------
    for i, prompts in enumerate(dataset):
        # Batch size 1

        # generate
        frames = engine.generate(
            prompts=prompts,
            height=args.height,
            width=args.width,
            seed=args.seed,
            num_steps=args.num_steps,
            guidance=args.guidance,
        )

        # Save outputs
        strip_path = out_tmpl.format(idx=i)
        frames[0].save(
            strip_path,
            format='JPEG',
            quality=95,
        )
        print(f"Saved sequence {i} to: {strip_path}")

if __name__ == '__main__':
    main()
