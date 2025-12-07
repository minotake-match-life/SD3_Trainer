# train.py
import os

import re
import tqdm
import math
import torch
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file

from datetime import datetime
from einops import rearrange, repeat
from safetensors import safe_open

from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import glob
from torch.utils.data import DataLoader

# === SD3 ===
from sd3_infer import load_into, ClipG, ClipL, T5XXL, VAE
from mmditx import MMDiTX
from sd3_impls import SD3LatentFormat
from other_impls import SD3Tokenizer

# ---------------------------
# ユーティリティ
# ---------------------------
class SimpleImageDataset(Dataset):
    """
    dataset_root 配下の画像ファイル(*.png, *.jpg, *.jpeg, *.webp)を全部読み、
    [-1, 1] 正規化した (C, H, W) テンソルと、簡単なプロンプトを返すだけのサンプル用 Dataset。
    """
    def __init__(self, root, image_size=512):
        super().__init__()
        self.root = root
        self.paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
            self.paths.extend(sorted(glob.glob(os.path.join(root, ext))))

        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {root}")

        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(image_size),
            T.ToTensor(),                          # [0, 1]
            T.Normalize([0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5])          # [-1, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)  # (C, H, W), [-1, 1]
        prompt = f"A photo of {os.path.basename(path)}" # sample prompt

        return img, prompt

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0
        self.count = 0
    @property
    def value(self):
        return self.sum / max(1, self.count)
    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n

def sample_t_logit_normal(batch_size, mu=0.0, sigma=1.0, eps=1e-5, device="cuda"):
    u = torch.randn(batch_size, device=device) * sigma + mu
    t = torch.sigmoid(u)
    return t.clamp(eps, 1.0 - eps)  # avoid exact 0/1

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def load_mmditx(filename, prefix="model.diffusion_model.", is_grad=True):
    with safe_open(filename, framework="pt", device="cpu") as file:
        patch_size = file.get_tensor(f"{prefix}x_embedder.proj.weight").shape[2]
        depth = file.get_tensor(f"{prefix}x_embedder.proj.weight").shape[0] // 64
        num_patches = file.get_tensor(f"{prefix}pos_embed").shape[1]
        pos_embed_max_size = round(math.sqrt(num_patches))
        adm_in_channels = file.get_tensor(f"{prefix}y_embedder.mlp.0.weight").shape[1]
        context_shape = file.get_tensor(f"{prefix}context_embedder.weight").shape
        """
        all shapes:
            patch_size: 2
            depth: 24
            num_patches: 147456 = 384**2
            pos_embed_max_size: 384
            adm_in_channels: 2048
            context_shape: torch.Size([1536, 4096]) 
        """
        qk_norm = (
            "rms"
            if f"{prefix}joint_blocks.0.context_block.attn.ln_k.weight" in file.keys()
            else None
        )
        x_block_self_attn_layers = sorted(
            [
                int(key.split(".x_block.attn2.ln_k.weight")[0].split(".")[-1])
                for key in list(
                    filter(
                        re.compile(".*.x_block.attn2.ln_k.weight").match, file.keys()
                    )
                )
            ]
        )
        context_embedder_config = {
            "target": "torch.nn.Linear",
            "params": {
                "in_features": context_shape[1],
                "out_features": context_shape[0],
            },
        }
        model = MMDiTX(
            input_size=None,
            pos_embed_scaling_factor=None,
            pos_embed_offset=None,
            pos_embed_max_size=pos_embed_max_size,
            patch_size=patch_size,
            in_channels=16,
            depth=depth,
            num_patches=num_patches,
            adm_in_channels=adm_in_channels,
            context_embedder_config=context_embedder_config,
            qk_norm=qk_norm,
            x_block_self_attn_layers=x_block_self_attn_layers,
            dtype=torch.bfloat16,
            device="cuda",
        )
        model = model.to(torch.bfloat16).to("cuda")
        load_into(file, model, prefix=prefix, device="cuda", dtype=torch.bfloat16, grad=is_grad)

    return model

# ---------------------------
# マルチGPUエントリ
# ---------------------------
def main(args):
   # ---- replace the env detection block in main(args) ----
    def _get_env_int(keys, default):
        for k in keys:
            v = os.environ.get(k)
            if v is not None and str(v).isdigit():
                return int(v)
        return default

    ngpus_per_node = torch.cuda.device_count()  # (=1想定)
    node_count = _get_env_int(["OMPI_COMM_WORLD_SIZE", "PBS_NP", "SLURM_NPROCS"], 1)
    node_rank  = _get_env_int(["OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID"], 0)

    # ★ PBSのJOBID or 共有ファイルを優先的に使う（ランダム生成をやめる）
    job_id = os.environ.get("PBS_JOBID") or os.environ.get("OMPI_COMM_WORLD_JOBID") or "0"

    # ★ PBSジョブで作った共有ファイルをそのまま使う（全ランクで同一パス）
    dist_file_env = os.environ.get("DIST_FILE")
    if dist_file_env is not None:
        dist_url = f"file://{dist_file_env}"
    else:
        # フォールバック（共有FS上に置く）
        workdir = os.environ.get("PBS_O_WORKDIR", os.getcwd())
        dist_url = f"file://{os.path.join(workdir, 'distfile.' + str(job_id))}"

    print(f"[rank?] node_count={node_count} ngpus_per_node={ngpus_per_node} node_rank={node_rank} dist_url={dist_url}", flush=True)

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=({
        "ngpus_per_node": ngpus_per_node,
        "node_count": node_count,
        "node_rank": node_rank,
        "dist_url": dist_url,
        "job_id": job_id
    }, args))

# ---------------------------
# メイン
# ---------------------------
def main_worker(local_rank, cluster_args, args):
    print(f"Hi from local rank {local_rank}!", flush=True)

    # ----------------------------
    # configure distributed training
    # ----------------------------
    world_size = cluster_args["node_count"] * cluster_args["ngpus_per_node"]
    global_rank = cluster_args["node_rank"] * cluster_args["ngpus_per_node"] + local_rank
    dist.init_process_group(
        backend="nccl",
        init_method=cluster_args["dist_url"],
        world_size=world_size,
        rank=global_rank,
    )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if global_rank == 0:
        if args.exp_name is not None:
            store_dir = "./logs/" + args.exp_name + "/" + datetime.strftime(datetime.now(), "%Y-%m-%d_%H%M%S")
        else: 
            store_dir = "./logs/train/" + datetime.strftime(datetime.now(), "%Y-%m-%d_%H%M%S")
        for k, v in sorted(vars(args).items(), key=lambda x: x[0]):
            print(f"# {k}: {v}")
        print(f"# effective_batch_size: {world_size * args.local_batch_size}", flush=True)

    # ---------------------------
    # MODELs (SD3)
    # ---------------------------
    print("Load diffusion transformer", flush=True)
    # load diffusion transformer
    if args.pretrained_ckpt is None:
        sd3_path = "./models/sd3.5_medium.safetensors"
        prefix = "model.diffusion_model."
    else:
        sd3_path = args.pretrained_ckpt
        prefix = ""
    model = load_mmditx(sd3_path, prefix).to(device)
    if global_rank == 0:
        print(f"Model loaded with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters", flush=True)
    
    # load VAE
    model_base_folder = "./models"
    ae_files = os.path.join(model_base_folder, "sd3.5_medium.safetensors")
    with torch.no_grad():
        ae = VAE(ae_files)
    def get_latent(x, ae):
        """
        x must be in [-1, 1]
        encode has autocast bf16
        """
        # encoder has reparam trick
        z = ae.model.encode(x)  # (B, 16, H/8, W/8)
        z = SD3LatentFormat().process_in(z)  # scale factor
        return z
    
    # load text encoders
    text_model_folder = os.path.join(model_base_folder, "text_encoders")
    with torch.no_grad():
        t5 = T5XXL(text_model_folder, dtype=torch.bfloat16)
        clip_l = ClipL(text_model_folder)
        clip_g = ClipG(text_model_folder)
        tokenizer = SD3Tokenizer() 
    def get_cond(prompts, tokenizer, clip_l, clip_g, t5):
        """
        prompts: (B,)
        return: context (B, N, C), pooled (B, C'), on cpu
        """

        if isinstance(prompts, str):
            prompts = [prompts]

        # --- Tokenize (batched) ---
        tokens_list = [tokenizer.tokenize_with_weights(p) for p in prompts]
        l_batch = [tok["l"][0] for tok in tokens_list]        # List[ List[(token, weight)] ]
        g_batch = [tok["g"][0] for tok in tokens_list]
        t5_batch = [tok["t5xxl"][0] for tok in tokens_list]

        # --- Encode (batched) ---
        # [B, 77, C_*], [B, C_*'] on cpu
        l_out, l_pooled = clip_l.model.encode_token_weights(l_batch)
        g_out, g_pooled = clip_g.model.encode_token_weights(g_batch)
        t5_out, _ = t5.model.encode_token_weights(t5_batch)

        # --- Concatenate ---
        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        context = torch.cat([lg_out, t5_out], dim=-2)
        pooled = torch.cat((l_pooled, g_pooled), dim=-1)

        # --- Return ---
        return context, pooled

    # Frozen parameters
    for m in [ae, t5, clip_l, clip_g]:
        m.model.eval().to(device) # bf16
        for p in m.model.parameters():
            p.requires_grad = False

    # distribute the model
    model.train()
    model_parallel = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], find_unused_parameters=False
    )
    if global_rank == 0:
        print(f"Model distributed to gpu {global_rank}!", flush=True)
    
    # ---------------------------
    # DATALOADER
    # ---------------------------
    n_epochs = args.epochs
    save_every_n_epochs = args.save_every

    train_ds = SimpleImageDataset(args.dataset_root, image_size=512)
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, shuffle=True, drop_last=True
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.local_batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    # ---------------------------
    # OPTIMIZER
    # ---------------------------
    optim = torch.optim.AdamW(model_parallel.module.parameters(), lr=args.lr)
    if args.pretrained_ckpt is None:
        past_epochs = 0
    else:
        past_epochs = int(args.pretrained_ckpt.split("_")[-1].split(".")[0])
    loss_meter = AverageMeter()

    # ---------------------------
    # TRAIN LOOP
    # ---------------------------
    for epoch in range(past_epochs + 1, n_epochs + 1):
        if world_size > 1 and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        iterator = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}") if global_rank == 0 else train_loader
            
        for frames, prompts in iterator:
            # frames: (B,C,H,W), [-1,1] prompts: List[str] length B
            
            # prompt dropout
            if random.random() < 0.15:
                prompts = [""] * len(prompts)

            # frames: (B, C, H, W) -> フレームをバッチにまとめる
            B, C, H, W = frames.shape
            frames = frames.to(device, non_blocking=True) # [-1, 1] 

            # AE encode -> latent x, frames must be in [-1, 1] range
            with torch.no_grad():
                x = get_latent(frames, ae)

            # ノイズとv_star
            z = torch.randn_like(x) # (B, 16, H/8, W/8)
            v_star = z - x

            # 中間表現 x_t
            t = sample_t_logit_normal(B, mu=0.0, sigma=1.0, device=device) #logit(0,1)
            t_ = t.view(-1, 1, 1, 1) # sigmas [0,1]
            t = t * 1000 # timesteps [0, 1000]
            x_t = (1.0 - t_) * x + t_ * z

            # Patchトークン化 & guidance準備
            with torch.no_grad():
                context, y = get_cond(prompts, tokenizer, clip_l, clip_g, t5)
                context = context.to(device, non_blocking=True, dtype=torch.bfloat16)  # (B, N, C)
                y = y.to(device, non_blocking=True, dtype=torch.bfloat16)

            # 順伝播 (predict v_star)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model_parallel(
                    x_t,
                    t, # (B,) in [0, 1000]
                    y,
                    context,
                )
                # 損失計算
                loss = torch.nn.functional.mse_loss(pred, v_star, reduction="mean")
            
            # 勾配計算とパラメータ更新
            optim.zero_grad(set_to_none=True)
            loss.backward() # DistributedDataParallel does gradient averaging
            optim.step()
            loss_meter.update(loss.item())

        # ロギング & セーブ
        if global_rank == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss_meter.value:.6f}", flush=True)
            os.makedirs(store_dir, exist_ok=True)
            with open(os.path.join(store_dir, "losses.txt"), "a") as f:
                f.write(f"{epoch:03d}{loss_meter.value:12.6f}\n")
            if epoch % save_every_n_epochs == 0:
                save_file(model_parallel.module.state_dict(), os.path.join(store_dir, f"train_{epoch:03d}.safetensors"))
            loss_meter.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_batch_size", type=int, default=4) 
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--exp_name", type=str, default=None)

    # SD3
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--pretrained_ckpt", type=str, default=None, help="Path to a pretrained checkpoint to load")

    args = parser.parse_args()

    main(args)
