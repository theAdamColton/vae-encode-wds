from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import io
from pathlib import Path
import torch
import numpy as np
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import jsonargparse
from torchvision.transforms import Resize
import webdataset as wds
import pandas as pd
from PIL import Image


def pil2torch(b):
    im = Image.open(io.BytesIO(b)).convert("RGB")
    im = np.asarray(im)
    im = torch.from_numpy(im)
    im = im.movedim(-1, 0)
    return im


class ResizeToMult(torch.nn.Module):
    def __init__(self, mult):
        super().__init__()
        self.mult = mult

    def forward(self, im):
        c, h, w = im.shape
        nh = int(round(h / self.mult) * self.mult)
        nw = int(round(w / self.mult) * self.mult)
        im = Resize((nh, nw))(im)
        return im


@torch.inference_mode()
def main(
    vae_url: str,
    data_url: str,
    out_path: Path,
    device: str = "cuda",
    dtype=torch.bfloat16,
    image_column_name: str = "jpg",
    enable_tile_thresh: int = 1024,
    downscale_factor: int = 8,
):
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        vae_url,
        torch_dtype=dtype,
    ).to(device=device, dtype=dtype)
    # vae.enable_slicing()
    vae.enable_tiling()
    vae.decoder = None

    shards = list(wds.SimpleShardList(data_url))

    out_path.mkdir(exist_ok=True)
    progress_path = out_path / "progress.csv"

    if progress_path.exists():
        progress = pd.read_csv(progress_path)
        completed_shard_names = set(progress.shard_name)
    else:
        completed_shard_names = set()

    for shard in shards:
        shard = shard["url"]
        ds = (
            wds.WebDataset(shard, shardshuffle=False)
            .rename(image=image_column_name)
            .map_dict(image=pil2torch)
            .map_dict(image=ResizeToMult(downscale_factor))
        )
        ds = DataLoader(ds, num_workers=1, batch_size=None, collate_fn=None)

        in_shard_path = Path(shard)

        shard_name = in_shard_path.name
        if shard_name in completed_shard_names:
            print("Skipping", shard_name)
            continue

        print("Processing", shard_name)

        out_shard_path = out_path / shard_name
        with wds.TarWriter(str(out_shard_path)) as writer:
            for row in tqdm(ds):
                image = row.pop("image")
                image = image.to(dtype=dtype, device=device) / 255
                image = image * 2 - 1

                _, h, w = image.shape

                latent = vae.encode(image.unsqueeze(0)).latent_dist.mean.squeeze(0)

                _, lh, lw = latent.shape

                assert h // lh == downscale_factor
                assert w // lw == downscale_factor

                latent = latent.to(torch.float16).cpu().numpy()
                row["latent.npy"] = latent

                writer.write(row)

        d = pd.DataFrame([{"shard_name": shard_name}])
        d.to_csv(progress_path, mode="a")


if __name__ == "__main__":
    jsonargparse.CLI(main)
