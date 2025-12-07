import os
import tqdm
import torch
import random
import socket
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp

from datetime import datetime
from einops import rearrange, repeat

from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import glob
from torch.utils.data import DataLoader

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

# ---------------------------
# ユーティリティクラス
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
        store_dir = "./logs/" + datetime.strftime(datetime.now(), "%Y-%m-%d_%H%M%S")

    ###############
    # DATASET
    ###############
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

    ###############
    # MODEL
    ###############

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    model = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # freeze parameters of models to save more memory
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
   
    model = model.train().to(device)
    model_parallel = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)

    vae = vae.eval().to(device)
    text_encoder = text_encoder.eval().to(device)

    ###############
    # OPTIMIZER
    ###############
    learning_rate = 2e-5
    parameters2train = model_parallel.module.parameters()
    optim = torch.optim.AdamW(parameters2train, lr=learning_rate)
    loss_metric = AverageMeter()

    for epoch in range(1, n_epochs + 1):
        
        if world_size > 1 and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        iterator = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}") if global_rank == 0 else train_loader

        for frames, prompts in iterator:
            # frames: (B,C,H,W), [-1,1] prompts: List[str] length B

            actual_batch_size = frames.shape[0]
            frames = frames.to(device)

            if random.random() < 0.1:
                prompts = [""] * actual_batch_size

            with torch.no_grad():
                latents = vae.encode(frames).latent_dist.sample() * vae.config.scaling_factor
                text_input_ids = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids
                text_emb = text_encoder(text_input_ids.to(device))[0]

            t = torch.randint(0, noise_scheduler.num_train_timesteps, (actual_batch_size,), device=device).long()
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, t)

            model_output = model_parallel(noisy_latents, t, text_emb).sample

            loss = torch.nn.functional.mse_loss(noise, model_output, reduction='mean')

            optim.zero_grad()
            loss.backward()  # DistributedDataParallel does gradient averaging, i.e. loss is x-times smaller when trained on more GPUs
            optim.step()
            loss_metric.update(loss.item())

        if global_rank == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss_metric.value:.4f}", flush=True)

            os.makedirs(store_dir, exist_ok=True)
            with open(os.path.join(store_dir, "losses.txt"), "a") as f:
                f.write(f"{epoch:03d}{loss_metric.value:12.4f}\n")
            loss_metric.reset()

            if epoch % save_every_n_epochs == 0:
                torch.save(model_parallel.module.state_dict(), os.path.join(store_dir, f"model_{epoch:03d}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str,default="./stable-diffusion-v1-5")

    main(parser.parse_args())
