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

from train import load_mmditx
from sd3_infer import ClipG, ClipL, T5XXL, VAE
from other_impls import SD3Tokenizer
from sd3_impls import SD3LatentFormat, ModelSamplingDiscreteFlow

# ---------------------------
# utils
# ---------------------------
def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    """
    returns: (B, 16, H/8, W/8) 
    """
    return torch.randn(
        num_samples,
        16,
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

# ---------------------------
# Core inference
# ---------------------------
class SD3Engine:

    @torch.inference_mode()
    def __init__(
        self, 
        ckpt_path: str, #safetensors
        device: str = 'cuda',
    ):
        self.device = device

        # load MMDiTx
        sd3_path = ckpt_path
        prefix = ""
        self.model = load_mmditx(
            sd3_path,
            prefix,
            is_grad=False).to(device)
        self.model.eval()
        self.sampling = ModelSamplingDiscreteFlow(shift=3.0) # sd3.5 default

        # load VAE
        model_base_folder = "./models"
        ae_files = os.path.join(model_base_folder, "sd3.5_medium.safetensors")
        with torch.no_grad():
            self.ae = VAE(ae_files)
        
        # load text encoders
        text_model_folder = os.path.join(model_base_folder, "text_encoders")
        with torch.no_grad():
            self.t5 = T5XXL(text_model_folder, dtype=torch.bfloat16)
            self.clip_l = ClipL(text_model_folder)
            self.clip_g = ClipG(text_model_folder)
            self.tokenizer = SD3Tokenizer() 

        # Frozen parameters
        for m in [self.ae, self.t5, self.clip_l, self.clip_g]:
            m.model.eval().to(device)
            for p in m.model.parameters():
                p.requires_grad = False

    @torch.inference_mode()
    def decode_latent(self, x):
        z = self.ae.model.decode(x)  # (B, 16, H/8, W/8)
        z = SD3LatentFormat().process_out(z)  # scale factor
        return z

    @torch.inference_mode()
    def get_cond(self, prompts):
        """
        prompts: (B,)
        return: context (B, N = 77+ 77, C), pooled (B, C'), on cpu
        """

        if isinstance(prompts, str):
            prompts = [prompts]

        # --- Tokenize (batched) ---
        tokens_list = [self.tokenizer.tokenize_with_weights(p) for p in prompts] 
        l_batch = [tok["l"][0] for tok in tokens_list]  # B*[[(token_id, weight)]] -> [B*[(token_id, weight)]]
        g_batch = [tok["g"][0] for tok in tokens_list]
        t5_batch = [tok["t5xxl"][0] for tok in tokens_list]

        # --- Encode (batched) ---
        # [B, 77, C_*], [B, C_*'] on cpu
        l_out, l_pooled = self.clip_l.model.encode_token_weights(l_batch)
        g_out, g_pooled = self.clip_g.model.encode_token_weights(g_batch)
        t5_out, _ = self.t5.model.encode_token_weights(t5_batch)

        # --- Concatenate ---
        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        context = torch.cat([lg_out, t5_out], dim=-2)
        pooled = torch.cat((l_pooled, g_pooled), dim=-1)

        # --- Return ---
        return context, pooled
    
    def get_sigmas(self, sampling, steps):
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        timesteps = torch.linspace(start, end, steps)
        sigs = []
        for x in range(len(timesteps)):
            ts = timesteps[x]
            sigs.append(sampling.sigma(ts))
        sigs += [0.0]
        return torch.tensor(sigs, device=self.device, dtype=torch.bfloat16)
    
    @torch.inference_mode()
    def denoise_latent(
        self,
        x,
        cond,
        pooled,
        uncond,
        uncond_pooled,
        num_steps: int,
        guidance: float,
    ):
        """
        Denoise in latent-token space using the SD3 model.
        returns: (B, 16, H/8, W/8)
        """

        # make sigmas [1...0]
        sigmas = self.get_sigmas(self.sampling, num_steps)

        # Euler steps
        for i in tqdm(range(len(sigmas) - 1)):
            
            # time
            sigma_hat = sigmas[i] 
            next_sigma = sigmas[i + 1] 
            timestep = self.sampling.timestep(sigma_hat)
            timesteps = timestep.repeat(x.shape[0]) # (B,)

            # cond と uncond を結合
            batch_x = torch.cat([x, x], dim=0)  # (B, 16, H/8, W/8) -> (2*B, 16, H/8, W/8)
            batch_cond = torch.cat([cond, uncond], dim=0) # (B, N, C) -> (2*B, N, C)
            batch_pooled = torch.cat([pooled, uncond_pooled], dim=0) # (B, C') -> (2*B, C')

            # モデルに一度に渡す
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch_output = self.model(
                    batch_x,
                    timesteps.repeat(2),  # timesteps を2回繰り返す
                    batch_pooled,
                    batch_cond,
                ) # (2*B, 16, H/8, W/8)

            # (B, 16, H/8, W/8) に分割
            cond_latent, uncond_latent = torch.chunk(batch_output, 2, dim=0)

            # classifier-free guidance
            if guidance == 1.0:
                c_out = cond_latent
            else:
                c_out = uncond_latent + guidance * (cond_latent - uncond_latent)

            # Euler step
            x = x + (next_sigma - sigma_hat) * c_out
        
        # (B, 16, H/8, W/8)
        return x
    
    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str], # (B,)
        height: int,
        width: int,
        seed: int,
        num_steps: int,
        guidance: float,
    ):

        # Seed & noise
        B = len(prompts)
        if B == 0:
            raise ValueError("prompts must contain at least one text.")
        if B == 1:
            prompts = [prompts[0] for _ in range(1)]  # keep 1 frame case explicit

        # get initial noise for all frames at once (shape: (B, 16, H/8, W/8))
        if seed is None:
            seed = 42
        x = get_noise(B, height, width, device=self.device, dtype=torch.bfloat16, seed=seed)

        # Prepare conditioning
        cond, pooled = self.get_cond(prompts)
        cond = cond.to(torch.bfloat16).to(self.device)
        pooled = pooled.to(torch.bfloat16).to(self.device)

        # Prepare for classifier-free guidance
        uncond_prompts = [""] * B
        uncond, uncond_pooled = self.get_cond(uncond_prompts)
        uncond = uncond.to(torch.bfloat16).to(self.device)
        uncond_pooled = uncond_pooled.to(torch.bfloat16).to(self.device)

        # Denoise
        xlat = self.denoise_latent(
            x,
            cond,
            pooled,
            uncond,
            uncond_pooled,
            num_steps=num_steps,
            guidance=guidance,
        )

        # Decode to pixel space
        xpix = self.decode_latent(xlat)

        # Post-process to uint8 frames (B, H, W, 3)
        xpix = xpix.clamp_(-1, 1).add_(1.0).mul_(255.0 / 2.0)
        xpix = xpix.to(torch.uint8)
        xpix = rearrange(xpix, 'b c h w -> b h w c', b=B)
        frames = [Image.fromarray(x.cpu().numpy()) for x in xpix]

        return frames

# ---------------------------
# CLI
# ---------------------------
def main():

    # -------------------------
    # Parse command-line arguments
    # -------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to fine-tuned FluxTmp checkpoint (.pt)')
    parser.add_argument('--prompt_file', type=str, required=True)

    parser.add_argument('--width', type=int, default=512, help='Output width (multiple of 16)')
    parser.add_argument('--height', type=int, default=512, help='Output height (multiple of 16)')
    parser.add_argument('--num_steps', type=int, default=None,
                        help='Sampling steps; default 4 for schnell, 50 otherwise')
    parser.add_argument('--guidance', type=float, default=5.0, help='Guidance (distilled)')
    parser.add_argument('--seed', type=int, default=None, help='Global seed (per-line)')

    parser.add_argument('--output_dir', type=str, default=f"./output/{time.strftime('%Y%m%d_%H%M')}", help='Output directory')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.num_steps is None:
        args.num_steps = 50

    os.makedirs(args.output_dir, exist_ok=True)
    out_tmpl = os.path.join(args.output_dir, 'seq_{idx}.jpg')

    # ---------------
    # Initialize the SD3 engine
    # ---------------
    print("Loading SD3 engine...")
    engine = SD3Engine(
        ckpt_path=args.ckpt_path,
        device=device
    )
    print("SD3 engine loaded.")

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
    # ---------------
    for i, prompt in enumerate(dataset):
        # Batch size 1

        # generate sequence
        frames = engine.generate(
            prompts=[prompt],
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

    print(f"All sequences saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
