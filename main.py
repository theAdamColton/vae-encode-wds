from tqdm import tqdm
import numpy as np
import io
from pathlib import Path
import torch
import numpy as np
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import jsonargparse
import webdataset as wds
import pandas as pd
from PIL import Image


def pil2torch(b):
    im = Image.open(io.BytesIO(b)).convert("RGB")
    im = np.asarray(im)
    im = torch.from_numpy(im)
    im = im.movedim(-1, 0)
    return im


@torch.inference_mode()
def main(
    vae_url: str,
    data_url: str,
    out_path: Path,
    device="cuda",
    dtype=torch.bfloat16,
    image_column_name="jpg",
):
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        vae_url,
        torch_dtype=dtype,
    ).to(device=device, dtype=dtype)
    vae.decoder = None

    shards = list(wds.SimpleShardList(data_url))

    out_path.mkdir(exist_ok=True)
    progress_path = out_path / "progress.csv"
    progress = pd.read_csv(progress_path)
    completed_shard_names = set(progress.shard_name)

    for shard in shards:
        ds = wds.WebDataset(shard, shardshuffle=False)

        in_shard_path = Path(shard)

        shard_name = in_shard_path.name
        if shard_name in completed_shard_names:
            print("Skipping", shard_name)
            continue

        out_shard_path = out_path / in_shard_path.name
        with wds.TarWriter(str(out_shard_path)) as writer:
            for row in tqdm(ds):
                image = row.pop(image_column_name)
                image = pil2torch(image)
                image = image.to(dtype) / 255
                image = image * 2 - 1
                latent = vae.encode(image.unsqueeze(0)).latent_dist.mean.squeeze(0)
                latent = latent.to(torch.float16).cpu().numpy()
                row["latent.npy"] = latent

                writer.write(row)

        d = pd.DataFrame([{"shard_name": shard_name}])
        d.to_csv(progress_path, mode="a")


if __name__ == "__main__":
    jsonargparse.CLI(main)